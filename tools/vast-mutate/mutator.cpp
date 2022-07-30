#include "vast/mutate/mutator.hpp"
#define endl llvm::errs() << "\n"

using namespace util;

bool ReorderArgumentMutation::shouldMutate() {
  if (!reordered) {
    mlir::Operation &op = *mutator->opitInTmp;
    if (op.getNumOperands() > 1) {
      for (size_t i = 0; i < op.getNumOperands(); ++i) {
        for (size_t j = i + 1; j < op.getNumOperands(); ++j) {
          if (op.getOperand(i).getType() == op.getOperand(j).getType()) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

void ReorderArgumentMutation::mutate() {
  mlir::Operation &op = *mutator->opitInTmp;
  for (size_t i = 0; i < op.getNumOperands(); ++i) {
    mlir::Type ty = op.getOperand(i).getType();
    if (tyToPos.find(ty) == tyToPos.end()) {
      tyToPos.insert(std::make_pair(ty, llvm::SmallVector<size_t>()));
    }
    tyToPos.find(ty)->second.push_back(i);
  }
  for (auto mapit = tyToPos.begin(); mapit != tyToPos.end(); ++mapit) {
    if (mapit->second.size() > 1) {
      llvm::SmallVector<mlir::Value> argVec;
      for (size_t i : mapit->second) {
        argVec.push_back(op.getOperand(i));
      }
      llvm::SmallVector<mlir::Value> shuffled =
          util::getShuffledArray<mlir::Value>(argVec);
      size_t idx = 0;
      for (size_t i : mapit->second) {
        op.setOperand(i, shuffled[idx++]);
      }
    }
  }
  reordered = true;
}

void ReorderArgumentMutation::debug() {
  llvm::errs() << "debug in reorder\n";
  llvm::errs() << "==========\n";
  llvm::errs() << mutator->mutations.size();
  llvm::errs() << "==========\n";
  mutator->opit->print(llvm::errs());
  endl;
}

void ReorderArgumentMutation::reset() {
  reordered = false;
  tyToPos.clear();
}

void ReplaceValueMutation::replaceOperand(mlir::Operation &op, size_t pos) {
  bool forceadd = Random::getRandomBool();
  mlir::Value newVal, oldVal = op.getOperand(pos);
  if (forceadd) {
    // because getRandomValue can get the old value as well
    // try 3 times, if 3 times are the same old val, we should add a new one.
    for (size_t i = 0; i < 3; ++i) {
      newVal = mutator->getRandomValue(oldVal.getType());
      if (newVal != oldVal && newVal != mlir::Value()) {
        break;
      }
    }
  }
  if (newVal == mlir::Value() || newVal == oldVal) {
    newVal = mutator->addFunctionArgument(oldVal.getType());
  }
  op.setOperand(pos, newVal);
}

bool ReplaceValueMutation::shouldMutate() {
  return !replaced && mutator->opitInTmp->getNumOperands() > 0;
}

void ReplaceValueMutation::mutate() {
  mlir::Operation &op = *mutator->opitInTmp;
  // try to mutate every argument in the operation.
  for (size_t i = 0; i < op.getNumOperands(); ++i) {
    if (Random::getRandomBool()) {
      replaceOperand(op, i);
    }
  }
  replaced = true;
}

void ReplaceValueMutation::debug() {
  llvm::errs() << "debug in replace\n";
  endl;
}

void ReplaceValueMutation::reset() { replaced = false; }

void RandomMoveMutation::moveForward(mlir::Operation &op,
                                     mlir::Operation &target) {
  assert(target.isBeforeInBlock(&op) &&
         "tartet should be before than moved op");
  llvm::DenseMap<mlir::Value, llvm::SmallVector<size_t>> valSet;
  llvm::SmallVector<size_t> fixPos;
  std::vector<llvm::SmallVector<mlir::Value>> valBackup;
  for (size_t i = 0; i < op.getOperands().size(); ++i) {
    if(valSet.find(op.getOperand(i))==valSet.end()){
      valSet.insert(std::make_pair(op.getOperand(i),llvm::SmallVector<size_t>()));      
    }
    valSet.find(op.getOperand(i))->second.push_back(i);
  }
  for (auto it = --op.getIterator(); it != target.getIterator(); --it) {
    for (auto resIt = it->getResults().begin(); resIt != it->getResults().end();
         ++resIt) {
      if (auto valIt = valSet.find(*resIt); valIt != valSet.end()) {
        for(size_t i:valIt->second){
          fixPos.push_back(i);
        }
      }
    }
    valBackup.push_back(mutator->values.back());
    mutator->values.pop_back();
  }
  // fix values at target pos
  for (auto resIt = target.getResults().begin();
       resIt != target.getResults().end(); ++resIt) {
    if (auto valIt = valSet.find(*resIt); valIt != valSet.end()) {
      for(size_t i:valIt->second){
        fixPos.push_back(i);
      }
    }
  }
  valBackup.push_back(mutator->values.back());
  mutator->values.pop_back();

  for (size_t i : fixPos) {
    mlir::Value val=mutator->getOrInsertRandomValue(op.getOperand(i).getType());
    op.setOperand(i,
                  val);
  }
  op.moveBefore(&target);
  mutator->opitInTmp=op.getIterator();

  while (!valBackup.empty()) {
    mutator->values.push_back(valBackup.back());
    valBackup.pop_back();
  }
}

void RandomMoveMutation::moveBackward(mlir::Operation &op,
                                      mlir::Operation &target) {
  assert(op.isBeforeInBlock(&target) && "op should be before to target");
  llvm::DenseMap<mlir::Value, bool> valSet;
  for (auto resIt = op.getResults().begin(); resIt != op.getResults().end();
       ++resIt) {
    valSet.insert(std::make_pair(*resIt, true));
  }
  for (auto it = ++op.getIterator(); it!=op.getBlock()->getOperations().end()&&it!= target.getIterator(); ++it) {
    for (size_t i = 0; i < it->getNumOperands(); ++i) {
      if (valSet.find(it->getOperand(i)) != valSet.end()) {
        it->setOperand(
            i, mutator->getOrInsertRandomValue(it->getOperand(i).getType()));
      }
    }
  }
  op.moveBefore(&target);
  mutator->opitInTmp=op.getIterator();
}

bool RandomMoveMutation::shouldMutate() {
  mlir::Operation &op = *mutator->opitInTmp;
  mlir::Block *b = op.getBlock();
  return b->getOperations().size() > 2 && b->getOperations().size()-1!= mutator-> curPos;
}

void RandomMoveMutation::mutate() {
  mlir::Operation &op = *mutator->opitInTmp;
  mlir::Block *b = op.getBlock();
  bool canMoveForward = (op.getIterator() != b->begin()),
       canMoveBackward = (op.getIterator() != --(b->end()));
  assert((canMoveForward || canMoveBackward) &&
         "at least one direction should be available");
  bool randomChoice = Random::getRandomBool(); // true for forward;
  size_t targetPos;
  if (canMoveForward) {
    if (!canMoveBackward || randomChoice) {
      targetPos = Random::getRandomInt() % mutator->curPos;
    }
  }
  if (canMoveBackward) {
    if (!canMoveForward || !randomChoice) {
      targetPos = mutator->curPos + 1 +
                  Random::getRandomInt() %
                      (b->getOperations().size() -1 - mutator->curPos);
    }
  }
  mlir::Block::iterator targetIt = b->begin();
  for (size_t i = 0; i < targetPos; ++i, ++targetIt)
    ;
  if (targetPos < mutator->curPos) {
    moveForward(op, *targetIt);
  } else {
    moveBackward(op, *targetIt);
  }
  moved=true;
}

void RandomMoveMutation::debug() {
  llvm::errs() << "debug in move\n";
  endl;
}

void RandomMoveMutation::reset() { moved = false; }

bool FunctionMutator::canMutate(mlir::FuncOp &func) {
  for(auto op_it=func.getBody().getBlocks().front().begin();op_it!=func.getBody().getBlocks().front().end();++op_it){
    llvm::errs()<<"\n"<<mlir::isa<vast::hl::WhileOp>(*op_it)<<"==============\n";
    op_it->print(llvm::errs());
    llvm::errs()<<"============\n";
    if(op_it->getNumOperands()!=0&&op_it->getNumResults()!=0){
      return true;
    }
  }
  return false;
  /*return std::any_of(
      func.getBody().getBlocks().front().begin(),
      func.getBody().getBlocks().front().end(), [](mlir::Operation &op) {
        return op.getNumOperands() != 0 && op.getNumResults() != 0;
      });*/
}

void FunctionMutator::init(std::shared_ptr<FunctionMutator> mutator) {
  mutations.push_back(std::make_unique<ReorderArgumentMutation>(mutator));
  mutations.push_back(std::make_unique<ReplaceValueMutation>(mutator));
  mutations.push_back(std::make_unique<RandomMoveMutation>(mutator));
}

FunctionMutator::FunctionMutator(mlir::FuncOp curFunc,
                                 mlir::BlockAndValueMapping &bavMap)
    : curFunc(curFunc), bavMap(bavMap), curPos(0),
      valueFuncs({&FunctionMutator::getRandomDominatedValue,
                  &FunctionMutator::getRandomExtraValue}) {
  values.push_back(llvm::SmallVector<mlir::Value>());
  for (auto argit = curFunc.args_begin(); argit != curFunc.args_end();
       ++argit) {
    values.back().push_back(*argit);
  }

  opit = curFunc.getBody().getBlocks().front().begin();
  moveToNextValidInstruction();
}

mlir::Value FunctionMutator::getRandomDominatedValue(mlir::Type ty) {
  for (auto &res : values) {
    mlir::Value val = util::findRanomInArray<mlir::Value, mlir::Type>(
        res, ty,
        [](mlir::Value val, mlir::Type ty) { return val.getType() == ty; },
        mlir::Value());
    if (val != mlir::Value()) {
      return bavMap.lookup(val);
    }
  }
  return mlir::Value();
}

mlir::Value FunctionMutator::getRandomExtraValue(mlir::Type ty) {
  return util::findRanomInArray<mlir::Value, mlir::Type>(
      extraValues, ty,
      [](mlir::Value val, mlir::Type ty) { return val.getType() == ty; },
      mlir::Value());
}

void FunctionMutator::mutate() {
  for (size_t i = 0; i < mutations.size(); ++i) {
    if (mutations[i]->shouldMutate()) {
      mutations[i]->reset();
      mutations[i]->mutate();
    }
  }
  moveToNextValidInstruction();
}

void FunctionMutator::resetCopy(std::shared_ptr<mlir::ModuleOp> tmpCopy) {
  this->tmpCopy = tmpCopy;
  mlir::Value res = opit->getResult(0);
  assert(bavMap.contains(res) && "copy should be same!\n");
  mlir::Value tmpres = bavMap.lookup(res);
  opitInTmp = tmpres.getDefiningOp()->getIterator();
  extraValues.clear();
}

mlir::Value FunctionMutator::addFunctionArgument(mlir::Type ty) {
  mlir::Operation *op = opitInTmp->getParentOp();
  assert(mlir::isa<mlir::FuncOp>(*op) && "operation should be a FuncOp");
  mlir::FuncOp funcInTmp = mlir::dyn_cast<mlir::FuncOp>(*op);
  mlir::Value val = util::addFunctionParameter(funcInTmp, ty);
  extraValues.push_back(val);
  return val;
}

mlir::Value FunctionMutator::getRandomValue(mlir::Type ty) {
  mlir::Value result;
  size_t sz = valueFuncs.size();
  for (size_t i = 0, pos = Random::getRandomInt() % sz; i < sz; ++i, ++pos) {
    if (pos == sz) {
      pos = 0;
    }
    result = (this->*valueFuncs[pos])(ty);
    if (result != mlir::Value()) {
      break;
    }
  }
  return result;
}

mlir::Value FunctionMutator::getOrInsertRandomValue(mlir::Type ty,
                                                    bool forceAdd) {
  mlir::Value result;
  if (!forceAdd) {
    result = getRandomValue(ty);
    if (result != mlir::Value()) {
      return result;
    }
  }
  return addFunctionArgument(ty);
}

void FunctionMutator::moveToNextInstruction() {
  values.push_back(llvm::SmallVector<mlir::Value>());
  for (auto val : opit->getResults()) {
    values.back().push_back(val);
  }
  ++opit;
  ++curPos;
  if (opit == curFunc.getBody().getBlocks().front().end()) {
    opit = curFunc.getBody().getBlocks().front().begin();
    values.resize(1);
    extraValues.clear();
    curPos = 0;
  }
}

void FunctionMutator::moveToNextValidInstruction() {
  moveToNextInstruction();
  while (opit->getNumOperands() == 0 || opit->getNumResults() == 0) {
    moveToNextInstruction();
  }
}

void Mutator::test() {
}

void Mutator::recopy() {
  bavMap.clear();
  mlir::Operation *copy = module.getOperation()->clone(bavMap);
  assert(mlir::isa<mlir::ModuleOp>(*copy) && "should be a moudle");
  // tmpCopy = mlir::dyn_cast<mlir::ModuleOp>(*copy);
  tmpCopy =
      std::make_shared<mlir::ModuleOp>(mlir::dyn_cast<mlir::ModuleOp>(*copy));
  assert(tmpCopy != nullptr && "copy should not be null");
  for (auto &mutators : functionMutators) {
    mutators->resetCopy(tmpCopy);
  }
}

void Mutator::mutateOnce() {
  recopy();
  functionMutators[curPos]->mutate();
  curFunction=functionMutators[curPos]->getFunctionName();
  ++curPos;
  if(curPos==functionMutators.size()){
    curPos=0;
  }
}

void Mutator::saveModule(const std::string& outputFileName){
  std::error_code ec;
  llvm::raw_fd_ostream fout(outputFileName, ec);
  fout << "//Current seed: "<<util::Random::getSeed()<<"\n";
  fout << "//============tmp copy============\n";
  tmpCopy->print(fout);
  fout.close();
  llvm::errs()<<"file wrote to "<<outputFileName<<"\n";
}

#undef endl