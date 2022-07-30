#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/InitAllDialects.h"
#include "vast/mutate/mutator.hpp"
#include "vast/Dialect/Dialects.hpp"
#include "vast/Dialect/HighLevel/Passes.hpp"
#include "llvm/Support/Debug.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/ADT/StringSet.h"
#include <filesystem>
#include <string>

using namespace std;
using namespace mlir;

llvm::cl::OptionCategory MlirTvCategory("mlir-tv options", "");
llvm::cl::OptionCategory MLIR_MUTATE_CAT("mlir-mutate-tv options", "");

llvm::cl::opt<string> filename_src(llvm::cl::Positional,
                                   llvm::cl::desc("first-mlir-file"),
                                   llvm::cl::Required,
                                   llvm::cl::value_desc("filename"),
                                   llvm::cl::cat(MLIR_MUTATE_CAT));

llvm::cl::opt<string> outputFolder(llvm::cl::Positional,
                                   llvm::cl::desc("<outputFileFolder>"),
                                   llvm::cl::Required,
                                   llvm::cl::value_desc("folder"),
                                   llvm::cl::cat(MLIR_MUTATE_CAT));

llvm::cl::opt<unsigned>
    arg_smt_to("smt-to",
               llvm::cl::desc("Timeout for SMT queries (default=30000)"),
               llvm::cl::init(30000), llvm::cl::value_desc("ms"),
               llvm::cl::cat(MLIR_MUTATE_CAT));

llvm::cl::opt<bool>
    arg_verbose("verbose", llvm::cl::desc("Be verbose about what's going on"),
                llvm::cl::Hidden, llvm::cl::init(false),
                llvm::cl::cat(MLIR_MUTATE_CAT));

llvm::cl::opt<long long> randomSeed(
    "s",
    llvm::cl::value_desc("specify the seed of the random number generator"),
    llvm::cl::cat(MLIR_MUTATE_CAT),
    llvm::cl::desc("specify the seed of the random number generator"),
    llvm::cl::init(-1));

llvm::cl::opt<int>
    numCopy("n", llvm::cl::value_desc("number of copies of test files"),
            llvm::cl::desc("specify number of copies of test files"),
            llvm::cl::cat(MLIR_MUTATE_CAT), llvm::cl::init(-1));

llvm::cl::opt<int>
    timeElapsed("t", llvm::cl::value_desc("seconds of the mutator should run"),
                llvm::cl::cat(MLIR_MUTATE_CAT),
                llvm::cl::desc("specify seconds of the mutator should run"),
                llvm::cl::init(-1));

llvm::cl::opt<bool>
    testMode("test",
             llvm::cl::value_desc("mutation file and verify its syntax, without calling alive2"),
             llvm::cl::desc("mutation file and verify its syntax, without calling alive2"),
             llvm::cl::cat(MLIR_MUTATE_CAT));

filesystem::path inputPath, outputPath;

bool isValidOutputPath(), isValidInputPath();
int runOnce(std::shared_ptr<mlir::ModuleOp> src_ptr, Mutator &mutator);
void programEnd(),
    timeMode(unique_ptr<llvm::MemoryBuffer> srcBuffer, MLIRContext *context),
    copyMode(unique_ptr<llvm::MemoryBuffer> srcBuffer, MLIRContext *context);
std::string getOutputFile(int ith, bool isOptimized = false);

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::PrettyStackTraceProgram X(argc, argv);
  llvm::EnableDebugBuffering = true;  

  llvm::cl::ParseCommandLineOptions(argc, argv);
  if (outputFolder.back() != '/')
    outputFolder += '/';

  MLIRContext context;
  DialectRegistry registry;
  mlir::registerAllDialects(context);
  vast::registerAllDialects(registry);    
  context.appendDialectRegistry(registry);
  //context.allowUnregisteredDialects();

  if (!isValidInputPath()) {
    llvm::errs() << "Invalid input file!\n";
    return 1;
  }

  if (!isValidOutputPath()) {
    llvm::errs() << "Invalid output path!\n";
    return 1;
  }

  string errorMessage;
  auto src_file = openInputFile(filename_src, &errorMessage);
  if (!src_file) {
    llvm::errs() << errorMessage << "\n";
    return 66;
  }
  if (randomSeed >= 0) {
    util::Random::setSeed((unsigned)randomSeed);
  }
  llvm::errs() << "current random seed: " << util::Random::getSeed() << "\n";
  if(numCopy>0){
    copyMode(std::move(src_file),&context);
  }else if(timeElapsed>0){
    timeMode(std::move(src_file),&context);
  }
  programEnd();

  return 0;
}

bool isValidInputPath() {
  bool result = filesystem::status(string(filename_src)).type() ==
                filesystem::file_type::regular;
  if (result) {
    inputPath = filesystem::path(string(filename_src));
  }
  return result;
}

bool isValidOutputPath() {
  bool result = filesystem::status(string(outputFolder)).type() ==
                filesystem::file_type::directory;
  if (result) {
    outputPath = filesystem::path(string(outputFolder));
  }
  return result;
}

void timeMode(unique_ptr<llvm::MemoryBuffer> srcBuffer, MLIRContext *context) {
  llvm::SourceMgr src_sourceMgr;
  src_sourceMgr.AddNewSourceBuffer(move(srcBuffer), llvm::SMLoc());

  auto ir_before = parseSourceFile<ModuleOp>(src_sourceMgr, context);
  if (!ir_before) {
    llvm::errs() << "Cannot parse source file\n";
    return;
  }

  std::shared_ptr<mlir::ModuleOp> src_ptr =
      std::make_shared<mlir::ModuleOp>(ir_before.release());
  Mutator mutator(*src_ptr);
  bool init=mutator.init();
  if(!init){
    llvm::errs()<<"Cannot find any location to mutate!\n";
    return;
  }

  std::chrono::duration<double> sum = std::chrono::duration<double>::zero();
  int cnt = 1;
  while (sum.count() < timeElapsed) {
    auto t_start = std::chrono::high_resolution_clock::now();
    int res = runOnce(src_ptr, mutator);

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cur = t_end - t_start;
    if (arg_verbose) {
      mutator.saveModule(getOutputFile(cnt));
      llvm::outs() << "Generted " + to_string(cnt) + "th copies in " +
                          to_string((cur).count()) + " seconds\n";
    }
    sum += cur;
    ++cnt;
  }
}

void copyMode(unique_ptr<llvm::MemoryBuffer> srcBuffer, MLIRContext *context) {
  llvm::SourceMgr src_sourceMgr;
  src_sourceMgr.AddNewSourceBuffer(move(srcBuffer), llvm::SMLoc());

  auto ir_before = parseSourceFile<mlir::ModuleOp>(src_sourceMgr, context);
  if (!ir_before) {
    llvm::errs() << "Cannot parse source file\n";
    return;
  }

  std::shared_ptr<mlir::ModuleOp> src_ptr =
      std::make_shared<mlir::ModuleOp>(ir_before.release());
  Mutator mutator(*src_ptr);
  bool init=mutator.init();
  if(!init){
    llvm::errs()<<"Cannot find any location to mutate!\n";
    return;
  }
  int cnt = 0;
  for (int i = 0; i < numCopy; ++i) {
    int res = runOnce(src_ptr, mutator);

    if (arg_verbose) {
      mutator.saveModule(getOutputFile(cnt));
      llvm::outs() << "Generted " + to_string(cnt) + "th copies\n";
    }
    ++cnt;
  }
}

std::string getOutputFile(int ith, bool isOptimized) {
  static std::string templateName = std::string(outputFolder) + inputPath.stem().string();
  return templateName + to_string(ith) + (isOptimized ? "-opt.mlir" : ".mlir");
}

int runOnce(std::shared_ptr<mlir::ModuleOp> src_ptr, Mutator &mutator) {
  mutator.mutateOnce();
  if(testMode){
    return 0;
  }

  std::shared_ptr<mlir::ModuleOp> tgt_ptr = mutator.getCopy();
  llvm::StringSet<> set;
  set.insert(mutator.getCurrentFunction());
  return 0;
  //thsi should be sent into a verify function.
  //return validate(src_ptr, tgt_ptr, set);
}

void programEnd() { llvm::errs() << "Program End.\n"; }
