#pragma once
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/FileUtilities.h"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/mutate/mutatorUtil.hpp"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include <iterator>
/*
Mutations without any knowledge about the dialect:
1. switch the order of arguments
2. replace the usage to a value
    a. with function parameter (maybe new added)
    b. with another same type value
    c. with a new generated instruction (this knowledge is collected from
context)
3. shuffle unrelated contiguous instructions
4. random move an instruction. Invalid values needs to be fixed (see option 2)
*/

class FunctionMutator;

class Mutation {
protected:
  std::shared_ptr<FunctionMutator> mutator;

public:
  Mutation(std::shared_ptr<FunctionMutator> mutator) : mutator(mutator){};
  virtual ~Mutation(){};
  std::shared_ptr<FunctionMutator> getFunctionMutator() const {
    return mutator;
  }
  void setFunctionMutator(std::shared_ptr<FunctionMutator> mutator) {
    this->mutator = mutator;
  }
  virtual bool shouldMutate() = 0;
  virtual void mutate() = 0;
  virtual void debug() = 0;
  virtual void reset() = 0;
};

class ReorderArgumentMutation : public Mutation {
  llvm::DenseMap<mlir::Type, llvm::SmallVector<size_t>> tyToPos;
  bool reordered;

public:
  ReorderArgumentMutation(std::shared_ptr<FunctionMutator> mutator)
      : Mutation(mutator), reordered(false){};
  bool shouldMutate() override;
  void mutate() override;
  void debug() override;
  void reset() override;
};

class ReplaceValueMutation : public Mutation {
  bool replaced;
  void replaceOperand(mlir::Operation &op, size_t pos);

public:
  ReplaceValueMutation(std::shared_ptr<FunctionMutator> mutator)
      : Mutation(mutator), replaced(false){};
  bool shouldMutate() override;
  void mutate() override;
  void debug() override;
  void reset() override;
};
/*
 *
 */
class RandomMoveMutation : public Mutation {
  bool moved;
  void moveForward(mlir::Operation &op, mlir::Operation &target);
  void moveBackward(mlir::Operation &op, mlir::Operation &target);
  void fixValuesInOperand(mlir::Operation& op,llvm::DenseMap<mlir::Value, bool>& valSet);

public:
  RandomMoveMutation(std::shared_ptr<FunctionMutator> mutator)
      : Mutation(mutator), moved(false){};
  bool shouldMutate() override;
  void mutate() override;
  void debug() override;
  void reset() override;
};

class FunctionMutator;


/*
This mutator needs to update dominance info so it needs a FunctionMutator
*/
class FunctionMutatorIterator {
  std::shared_ptr<FunctionMutator> func;
  using RegionIterator = llvm::MutableArrayRef<mlir::Region>::iterator;
  RegionIterator region_it, region_it_end;
  using BlockIterator = llvm::simple_ilist<mlir::Block>::iterator;
  BlockIterator block_it;
  using OperationIterator = llvm::simple_ilist<mlir::Operation>::iterator;
  OperationIterator op_it;
  size_t operCnt,curPos;
  bool isOpEnd() { return op_it == block_it->getOperations().end(); }
  bool isBlockEnd() { return block_it == region_it->getBlocks().end(); }
  bool isRegionEnd() {
    return region_it == region_it_end;
  }

  void nextRegion();

  void nextBlock();

  void nextOperation();
public:
  FunctionMutatorIterator(std::shared_ptr<FunctionMutator> func, RegionIterator region_it, BlockIterator block_it,
                          OperationIterator op_it);
  FunctionMutatorIterator(std::shared_ptr<FunctionMutator> func, mlir::Operation& operation);
  FunctionMutatorIterator():func(nullptr){};
  OperationIterator getOperationIterator() { return op_it; }
  mlir::Operation& getOperation(){return *op_it;}
  BlockIterator getBlockIterator() { return block_it; }
  mlir::Block& getBlock(){return *block_it;}
  RegionIterator getRegionIterator() { return region_it; }
  mlir::Region& getRegion(){return *region_it;}
  bool isEnd() { return isRegionEnd(); }
  size_t getCurrentPos()const{return curPos;}
  std::shared_ptr<FunctionMutator> getFunctionMutator(){return func;}
  void next() {
    assert(!isEnd() && "cannot next because current iterator meets end");
    nextOperation();
  }
};

class FunctionMutator {
  friend ReorderArgumentMutation;
  friend ReplaceValueMutation;
  friend RandomMoveMutation;

  friend FunctionMutatorIterator;

  mlir::Value getRandomValue(mlir::Type ty);
  mlir::Value getRandomDominatedValue(mlir::Type ty);
  mlir::Value getRandomExtraValue(mlir::Type ty);
  mlir::Value addFunctionArgument(mlir::Type ty);
  mlir::Value getOrInsertRandomValue(mlir::Type ty, bool forceAdd = false);
  void moveToNextOperaion();
  void moveToNextMutant();

  FunctionMutatorIterator funcIt, funcItInTmp;
  std::vector<FunctionMutatorIterator> funcItStack;
  mlir::FuncOp curFunc;
  std::shared_ptr<mlir::ModuleOp> tmpCopy;
  mlir::BlockAndValueMapping &bavMap;
  llvm::DenseMap<mlir::Operation *, mlir::Operation *>& opMap;
  llvm::SmallVector<std::unique_ptr<Mutation>> mutations;
  std::vector<llvm::SmallVector<mlir::Value>> values;
  llvm::SmallVector<mlir::Value (FunctionMutator::*)(mlir::Type)> valueFuncs;
  llvm::SmallVector<mlir::Value> extraValues;

  static bool canMutate(mlir::Operation &op) {
    if (op.getNumOperands() != 0) {
      return true;
    }
    for (auto rit = op.getRegions().begin(); rit != op.getRegions().end();
         ++rit) {
      for (auto bit = rit->getBlocks().begin(); bit != rit->getBlocks().end();
           ++bit) {
        for (auto op_it = bit->getOperations().begin();
             op_it != bit->getOperations().end(); ++op_it) {
          if (canMutate(*op_it)) {
            return true;
          }
        }
      }
    }
    return false;
  }

public:
  FunctionMutator(mlir::FuncOp curFunc, mlir::BlockAndValueMapping &bavMap, llvm::DenseMap<mlir::Operation *, mlir::Operation *>& opMap);
  void mutate();
  void resetCopy(std::shared_ptr<mlir::ModuleOp> tmpCopy);
  static bool canMutate(mlir::FuncOp &func);
  void init(std::shared_ptr<FunctionMutator> mutator);
  llvm::StringRef getFunctionName() { return curFunc.getName(); }
};

class Mutator {
  mlir::ModuleOp &module;
  std::shared_ptr<mlir::ModuleOp> tmpCopy;
  mlir::BlockAndValueMapping bavMap;

  mlir::OpBuilder opBuilder;
  llvm::SmallVector<std::shared_ptr<FunctionMutator>> functionMutators;
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> opMap;

  void recopy();
  void setOpMap();
  size_t curPos;
  llvm::StringRef curFunction;

public:
  Mutator(mlir::ModuleOp &module)
      : module(module), tmpCopy(nullptr), opBuilder(module.getContext()),
        curPos(0){};
  Mutator(mlir::ModuleOp &&module)
      : module(module), tmpCopy(nullptr), opBuilder(module.getContext()),
        curPos(0){};
  bool init() {
    for (auto module_it = module.begin(); module_it != module.end();
         ++module_it) {
      if (mlir::isa<vast::hl::TranslationUnitOp>(module_it)) {
        mlir::Block &block = module_it->getRegion(0).getBlocks().front();
        for (auto fit = block.begin(); fit != block.end(); ++fit) {
          if (mlir::isa<mlir::FuncOp>(fit)) {
            mlir::FuncOp func = mlir::dyn_cast<mlir::FuncOp>(*fit);
            if (llvm::hasSingleElement(func.getRegion()) &&
                FunctionMutator::canMutate(func)) {
              functionMutators.push_back(
                  std::make_shared<FunctionMutator>(func, bavMap, opMap));
              functionMutators.back()->init(functionMutators.back());
            }
          }
        }
      }
    }
    return !functionMutators.empty();
  }
  mlir::ModuleOp &getOrigin() const { return module; }
  std::shared_ptr<mlir::ModuleOp> getCopy() const { return tmpCopy; }
  void setCopy(std::shared_ptr<mlir::ModuleOp> copy) {
    tmpCopy = copy;
    for (auto &mutators : functionMutators) {
      mutators->resetCopy(tmpCopy);
    }
  }
  llvm::StringRef getCurrentFunction() { return curFunction; };
  void mutateOnce();
  void test();
  void saveModule(const std::string &outputFileName);
};