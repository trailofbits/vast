#include "vast/mutate/mutatorUtil.hpp"

using namespace util;

std::random_device Random::rd;
std::uniform_int_distribution<int> Random::intDist(0, INT32_MAX);
std::uniform_real_distribution<double> Random::dblDist(0.0, 1.0);
unsigned Random::seed(rd());
std::mt19937 Random::mt(Random::seed);

const std::string OperationStateGenerator::varName = "MLIR-MUTATE-GEN-VAR";
int OperationStateGenerator::cnt = 0;

mlir::BlockArgument util::addFunctionParameter(mlir::FuncOp op, mlir::Type ty) {
  llvm::SmallVector<mlir::Type> inputs, outputs;
  mlir::Region &region = op.getBody();
  std::for_each(op.args_begin(), op.args_end(),
                [&inputs](auto arg) { inputs.push_back(arg.getType()); });
  inputs.push_back(ty);
  mlir::FunctionType newFnTy =
      mlir::FunctionType::get(op.getContext(), inputs, op.getResultTypes());
  // op.setFunctionTypeAttr(tyAttr);
  op.setType(newFnTy);
  region.addArgument(ty,
                     OperationStateGenerator::getNewLocation(op.getContext()));

  return region.getArguments().back();
}

mlir::OperationState
OperationStateGenerator::getNewOperationState(mlir::MLIRContext *context,
                                              std::string name) {
  if (name.empty()) {
    name = varName + std::to_string(cnt);
  }
  mlir::OperationState state(getNewLocation(context),
                             mlir::OperationName(name, context));
  return state;
}

mlir::Location
OperationStateGenerator::getNewLocation(mlir::MLIRContext *context) {
  mlir::StringAttr attr =
      mlir::StringAttr::get(context, varName + std::to_string(cnt));
  cnt++;
  mlir::NameLoc loc = mlir::NameLoc::get(attr);
  return loc;
}