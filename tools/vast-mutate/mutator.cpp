#include "vast/mutate/mutator.hpp"
#define endl llvm::errs() << "\n"

using namespace util;

bool ReorderArgumentMutation::shouldMutate()
{
  if (!reordered)
  {
    mlir::Operation &op = *mutator->opitInTmp;
    if (op.getNumOperands() > 1)
    {
      for (size_t i = 0; i < op.getNumOperands(); ++i)
      {
        for (size_t j = i + 1; j < op.getNumOperands(); ++j)
        {
          if (op.getOperand(i).getType() == op.getOperand(j).getType())
          {
            return true;
          }
        }
      }
    }
  }
  return false;
}

void ReorderArgumentMutation::mutate()
{
  mlir::Operation &op = *mutator->opitInTmp;
  for (size_t i = 0; i < op.getNumOperands(); ++i)
  {
    mlir::Type ty = op.getOperand(i).getType();
    if (tyToPos.find(ty) == tyToPos.end())
    {
      tyToPos.insert(std::make_pair(ty, llvm::SmallVector<size_t>()));
    }
    tyToPos.find(ty)->second.push_back(i);
  }
  for (auto mapit = tyToPos.begin(); mapit != tyToPos.end(); ++mapit)
  {
    if (mapit->second.size() > 1)
    {
      llvm::SmallVector<mlir::Value> argVec;
      for (size_t i : mapit->second)
      {
        argVec.push_back(op.getOperand(i));
      }
      llvm::SmallVector<mlir::Value> shuffled =
          util::getShuffledArray<mlir::Value>(argVec);
      size_t idx = 0;
      for (size_t i : mapit->second)
      {
        op.setOperand(i, shuffled[idx++]);
      }
    }
  }
  reordered = true;
}

void ReorderArgumentMutation::debug()
{
  llvm::errs() << "debug in reorder\n";
  llvm::errs() << "==========\n";
  llvm::errs() << mutator->mutations.size();
  llvm::errs() << "==========\n";
  mutator->opit->print(llvm::errs());
  endl;
}

void ReorderArgumentMutation::reset()
{
  reordered = false;
  tyToPos.clear();
}

void ReplaceValueMutation::replaceOperand(mlir::Operation &op, size_t pos)
{
  bool forceadd = Random::getRandomBool();
  mlir::Value newVal, oldVal = op.getOperand(pos);
  if (forceadd)
  {
    // because getRandomValue can get the old value as well
    // try 3 times, if 3 times are the same old val, we should add a new one.
    for (size_t i = 0; i < 3; ++i)
    {
      newVal = mutator->getRandomValue(oldVal.getType());
      if (newVal != oldVal && newVal != mlir::Value())
      {
        break;
      }
    }
  }
  if (newVal == mlir::Value() || newVal == oldVal)
  {
    newVal = mutator->addFunctionArgument(oldVal.getType());
  }
  op.setOperand(pos, newVal);
}

bool ReplaceValueMutation::shouldMutate()
{
  return !replaced && mutator->opitInTmp->getNumOperands() > 0;
}

void ReplaceValueMutation::mutate()
{
  mlir::Operation &op = *mutator->opitInTmp;
  // try to mutate every argument in the operation.
  for (size_t i = 0; i < op.getNumOperands(); ++i)
  {
    if (Random::getRandomBool())
    {
      replaceOperand(op, i);
    }
  }
  replaced = true;
}

void ReplaceValueMutation::debug()
{
  llvm::errs() << "debug in replace\n";
  endl;
}

void ReplaceValueMutation::reset() { replaced = false; }

void RandomMoveMutation::moveForward(mlir::Operation &op,
                                     mlir::Operation &target)
{
  assert(target.isBeforeInBlock(&op) &&
         "tartet should be before than moved op");
  llvm::DenseMap<mlir::Value, llvm::SmallVector<size_t>> valSet;
  llvm::SmallVector<size_t> fixPos;
  std::vector<llvm::SmallVector<mlir::Value>> valBackup;
  for (size_t i = 0; i < op.getOperands().size(); ++i)
  {
    if (valSet.find(op.getOperand(i)) == valSet.end())
    {
      valSet.insert(
          std::make_pair(op.getOperand(i), llvm::SmallVector<size_t>()));
    }
    valSet.find(op.getOperand(i))->second.push_back(i);
  }
  for (auto it = --op.getIterator(); it != target.getIterator(); --it)
  {
    for (auto resIt = it->getResults().begin(); resIt != it->getResults().end();
         ++resIt)
    {
      if (auto valIt = valSet.find(*resIt); valIt != valSet.end())
      {
        for (size_t i : valIt->second)
        {
          fixPos.push_back(i);
        }
      }
    }
    valBackup.push_back(mutator->values.back());
    mutator->values.pop_back();
  }
  // fix values at target pos
  for (auto resIt = target.getResults().begin();
       resIt != target.getResults().end(); ++resIt)
  {
    if (auto valIt = valSet.find(*resIt); valIt != valSet.end())
    {
      for (size_t i : valIt->second)
      {
        fixPos.push_back(i);
      }
    }
  }
  valBackup.push_back(mutator->values.back());
  mutator->values.pop_back();

  for (size_t i : fixPos)
  {
    mlir::Value val =
        mutator->getOrInsertRandomValue(op.getOperand(i).getType());
    op.setOperand(i, val);
  }
  op.moveBefore(&target);
  mutator->opitInTmp = op.getIterator();

  while (!valBackup.empty())
  {
    mutator->values.push_back(valBackup.back());
    valBackup.pop_back();
  }
}

void RandomMoveMutation::moveBackward(mlir::Operation &op,
                                      mlir::Operation &target)
{
  assert(op.isBeforeInBlock(&target) && "op should be before to target");
  llvm::DenseMap<mlir::Value, bool> valSet;
  for (auto resIt = op.getResults().begin(); resIt != op.getResults().end();
       ++resIt)
  {
    valSet.insert(std::make_pair(*resIt, true));
  }
  for (auto it = ++op.getIterator();
       it != op.getBlock()->getOperations().end() && it != target.getIterator();
       ++it)
  {
    for (size_t i = 0; i < it->getNumOperands(); ++i)
    {
      if (valSet.find(it->getOperand(i)) != valSet.end())
      {
        it->setOperand(
            i, mutator->getOrInsertRandomValue(it->getOperand(i).getType()));
      }
    }
  }
  op.moveBefore(&target);
  mutator->opitInTmp = op.getIterator();
}

bool RandomMoveMutation::shouldMutate()
{
  mlir::Operation &op = *mutator->opitInTmp;
  mlir::Block *b = op.getBlock();
  return b->getOperations().size() > 2 &&
         b->getOperations().size() - 1 != mutator->funcIt.getCurrentPos();
}

void RandomMoveMutation::mutate()
{
  mlir::Operation &op = *mutator->opitInTmp;
  mlir::Block *b = op.getBlock();
  bool canMoveForward = (op.getIterator() != b->begin()),
       canMoveBackward = (op.getIterator() != --(b->end()));
  assert((canMoveForward || canMoveBackward) &&
         "at least one direction should be available");
  bool randomChoice = Random::getRandomBool(); // true for forward;
  size_t targetPos;
  if (canMoveForward)
  {
    if (!canMoveBackward || randomChoice)
    {
      targetPos = Random::getRandomInt() % mutator->funcIt.getCurrentPos();
    }
  }
  if (canMoveBackward)
  {
    if (!canMoveForward || !randomChoice)
    {
      targetPos = mutator->funcIt.getCurrentPos() + 1 +
                  Random::getRandomInt() %
                      (b->getOperations().size() - 1 - mutator->funcIt.getCurrentPos());
    }
  }
  mlir::Block::iterator targetIt = b->begin();
  for (size_t i = 0; i < targetPos; ++i, ++targetIt)
    ;
  if (targetPos < mutator->funcIt.getCurrentPos())
  {
    moveForward(op, *targetIt);
  }
  else
  {
    moveBackward(op, *targetIt);
  }
  moved = true;
}

void RandomMoveMutation::debug()
{
  llvm::errs() << "debug in move\n";
  endl;
}

void RandomMoveMutation::reset() { moved = false; }

FunctionMutatorIterator::FunctionMutatorIterator(std::shared_ptr<FunctionMutator> func, RegionIterator region_it, BlockIterator block_it,
                                                 OperationIterator op_it)
    : func(func), region_it(region_it), block_it(block_it), op_it(op_it)
{
  region_it_end=region_it->getParentOp()->getRegions().end();
  curPos = std::distance(block_it->getOperations().begin(), op_it);
  operCnt = func->values.size();
};

FunctionMutatorIterator::FunctionMutatorIterator(std::shared_ptr<FunctionMutator> func, mlir::Operation &operation) : func(func), curPos(0)
{
  bool found = false;
  region_it_end=operation.getRegions().end();
  for (region_it = operation.getRegions().begin(); region_it != region_it_end; ++region_it)
  {
    for (block_it = region_it->getBlocks().begin(); block_it != region_it->getBlocks().end(); ++block_it)
    {
      if (!block_it->empty())
      {
        op_it = block_it->getOperations().begin();
        curPos = 0;
        found = true;
      }
      if(found){
        break;
      }
    }
    if(found){
      break;
    }
  }
  assert(found && "should at least find a location");
  operCnt = func->values.size();
}

void FunctionMutatorIterator::nextOperation()
{
  if (op_it->getNumResults() != 0)
  {
    llvm::SmallVector<mlir::Value> val;
    for (size_t i=0;i<op_it->getNumResults();++i)
    {
      val.push_back(op_it->getResult(i));
    }
    func->values.push_back(std::move(val));
  }
  ++op_it;
  ++curPos;
  if (isOpEnd())
  {
    func->values.resize(operCnt);
    nextBlock();
  }
}

void FunctionMutatorIterator::nextBlock()
{
  ++block_it;
  while (!isBlockEnd())
  {
    if (!block_it->empty())
    {
      op_it = block_it->getOperations().begin();
      curPos = 0;
      return;
    }
    ++block_it;
  }
  if (isBlockEnd())
  {
    nextRegion();
  }
}

void FunctionMutatorIterator::nextRegion()
{
  assert(!isRegionEnd());
  ++region_it;
  while (!isRegionEnd())
  {
    if (!region_it->empty())
    {
      block_it = region_it->getBlocks().begin();
      while (!isBlockEnd() && block_it->empty())
      {
        ++block_it;
      }
      if (!isBlockEnd())
      {
        op_it = block_it->getOperations().begin();
        curPos = 0;
        return;
      }
    }
    ++region_it;
  }
}

bool FunctionMutator::canMutate(mlir::FuncOp &func)
{
  for (auto op_it = func.getBody().getBlocks().front().begin();
       op_it != func.getBody().getBlocks().front().end(); ++op_it)
  {
    if (canMutate(*op_it))
    {
      return true;
    }
  }
  return false;
}

void FunctionMutator::init(std::shared_ptr<FunctionMutator> mutator)
{
  mutations.push_back(std::make_unique<ReorderArgumentMutation>(mutator));
  mutations.push_back(std::make_unique<ReplaceValueMutation>(mutator));
  mutations.push_back(std::make_unique<RandomMoveMutation>(mutator));
  funcIt = FunctionMutatorIterator(mutator, *curFunc.getOperation());
  moveToNextMutant();
}

FunctionMutator::FunctionMutator(mlir::FuncOp curFunc,
                                 mlir::BlockAndValueMapping &bavMap,llvm::DenseMap<mlir::Operation *, mlir::Operation *>& opMap)
    : curFunc(curFunc), bavMap(bavMap),opMap(opMap),
      valueFuncs({&FunctionMutator::getRandomDominatedValue,
                  &FunctionMutator::getRandomExtraValue})
{
  values.push_back(llvm::SmallVector<mlir::Value>());
  for (auto argit = curFunc.args_begin(); argit != curFunc.args_end();
       ++argit)
  {
    values.back().push_back(*argit);
  }

  //opit = curFunc.getBody().getBlocks().front().begin();
  //moveToNextMutant();
}

mlir::Value FunctionMutator::getRandomDominatedValue(mlir::Type ty)
{
  for (auto &res : values)
  {
    mlir::Value val = util::findRanomInArray<mlir::Value, mlir::Type>(
        res, ty,
        [](mlir::Value val, mlir::Type ty)
        { return val.getType() == ty; },
        mlir::Value());
    if (val != mlir::Value())
    {
      return bavMap.lookup(val);
    }
  }
  return mlir::Value();
}

mlir::Value FunctionMutator::getRandomExtraValue(mlir::Type ty)
{
  return util::findRanomInArray<mlir::Value, mlir::Type>(
      extraValues, ty,
      [](mlir::Value val, mlir::Type ty)
      { return val.getType() == ty; },
      mlir::Value());
}

void FunctionMutator::mutate()
{
  /*for (size_t i = 0; i < mutations.size(); ++i)
  {
    if (mutations[i]->shouldMutate())
    {
      mutations[i]->reset();
      mutations[i]->mutate();
    }
  }*/
  llvm::errs()<<"===========\n";
  funcIt.getOperation().print(llvm::errs());
  llvm::errs()<<"\n===========\n";
  moveToNextMutant();
}

void FunctionMutator::resetCopy(std::shared_ptr<mlir::ModuleOp> tmpCopy)
{
  assert(!funcIt.isEnd() && "func it shouldn't at end");
  this->tmpCopy = tmpCopy;
  //mlir::Value res = opit->getResult(0);
  //assert(bavMap.contains(res) && "copy should be same!\n");
  //mlir::Value tmpres = bavMap.lookup(res);
  //opitInTmp = tmpres.getDefiningOp()->getIterator();
  mlir::Operation& oper=*funcIt.getRegion().getParentOp();
  assert(opMap.find(&oper)!=opMap.end() && "copy should be the same");
  mlir::Operation& operInTmp=*opMap.find(&oper)->second;
  size_t dist=funcIt.getRegion().getRegionNumber();
  auto region_it_in_tmp=operInTmp.getRegions().begin();
  for(size_t i=0;i<dist;++i,++region_it_in_tmp);

  assert(opMap.find(&funcIt.getOperation())!=opMap.end() && "copy should be the same");
  auto op_it_in_tmp=opMap.find(&funcIt.getOperation())->second->getIterator();
  assert(bavMap.contains(&funcIt.getBlock()) && "copy should be the same");
  auto block_it_in_tmp=bavMap.lookup(&funcIt.getBlock())->getIterator();
  
  funcItInTmp=FunctionMutatorIterator(funcIt.getFunctionMutator(),region_it_in_tmp,block_it_in_tmp,op_it_in_tmp);
  extraValues.clear();
}

mlir::Value FunctionMutator::addFunctionArgument(mlir::Type ty)
{
  mlir::Operation *op = opitInTmp->getParentOp();
  assert(mlir::isa<mlir::FuncOp>(*op) && "operation should be a FuncOp");
  mlir::FuncOp funcInTmp = mlir::dyn_cast<mlir::FuncOp>(*op);
  mlir::Value val = util::addFunctionParameter(funcInTmp, ty);
  extraValues.push_back(val);
  return val;
}

mlir::Value FunctionMutator::getRandomValue(mlir::Type ty)
{
  mlir::Value result;
  size_t sz = valueFuncs.size();
  for (size_t i = 0, pos = Random::getRandomInt() % sz; i < sz; ++i, ++pos)
  {
    if (pos == sz)
    {
      pos = 0;
    }
    result = (this->*valueFuncs[pos])(ty);
    if (result != mlir::Value())
    {
      break;
    }
  }
  return result;
}

mlir::Value FunctionMutator::getOrInsertRandomValue(mlir::Type ty,
                                                    bool forceAdd)
{
  mlir::Value result;
  if (!forceAdd)
  {
    result = getRandomValue(ty);
    if (result != mlir::Value())
    {
      return result;
    }
  }
  return addFunctionArgument(ty);
}

void FunctionMutator::moveToNextOperaion()
{
  if (canVisitInside(funcIt.getOperation()))
  {
    funcItStack.push_back(funcIt);
    funcIt = FunctionMutatorIterator(funcIt.getFunctionMutator(), funcIt.getOperation());
  }
  else
  {
    funcIt.next();
  }
  if (funcIt.isEnd())
  {
    // need to check dominfo in funcIt;
    if (funcItStack.empty())
    {
      funcIt = FunctionMutatorIterator(funcIt.getFunctionMutator(), *curFunc.getOperation());
    }
    else
    {
      funcIt = funcItStack.back();
      funcItStack.pop_back();
      funcIt.next();
    }
  }
}

void FunctionMutator::moveToNextMutant()
{
  moveToNextOperaion();
  while (!canMutate(funcIt.getOperation()))
  {
    moveToNextOperaion();
  }
}

void Mutator::test() {}

static void addOneMappingRecord(
    mlir::Operation *from, mlir::Operation *to,
    llvm::DenseMap<mlir::Operation *, mlir::Operation *> &opMap)
{
  opMap[from] = to;
  assert(from->getNumRegions() == to->getNumRegions() &&
         "both inst should have same number of regions");
  for (auto from_rit = from->getRegions().begin(),
            to_rit = to->getRegions().begin();
       from_rit != from->getRegions().end(); ++from_rit, ++to_rit)
  {
    assert(from_rit->getBlocks().size() == to_rit->getBlocks().size() &&
           "both region should have same number of blocks");
    for (auto from_bit = from_rit->getBlocks().begin(),
              to_bit = to_rit->getBlocks().begin();
         from_bit != from_rit->getBlocks().end(); ++from_bit, ++to_bit)
    {
      assert(from_bit->getOperations().size() ==
                 to_bit->getOperations().size() &&
             "both block should have same number of operations");
      for (auto from_op_it = from_bit->getOperations().begin(),
                to_op_it = to_bit->getOperations().begin();
           from_op_it != from_bit->getOperations().end();
           ++from_op_it, ++to_op_it)
      {
        addOneMappingRecord(&*from_op_it, &*to_op_it, opMap);
      }
    }
  }
}

void Mutator::recopy()
{
  bavMap.clear();
  opMap.clear();
  mlir::Operation *copy = module.getOperation()->clone(bavMap);
  assert(mlir::isa<mlir::ModuleOp>(*copy) && "should be a moudle");
  // tmpCopy = mlir::dyn_cast<mlir::ModuleOp>(*copy);
  tmpCopy =
      std::make_shared<mlir::ModuleOp>(mlir::dyn_cast<mlir::ModuleOp>(*copy));
  assert(tmpCopy != nullptr && "copy should not be null");
  setOpMap();
  for (auto &mutators : functionMutators)
  {
    mutators->resetCopy(tmpCopy);
  }
}

void Mutator::setOpMap()
{
  addOneMappingRecord(module.getOperation(), tmpCopy->getOperation(), opMap);
}

void Mutator::mutateOnce()
{
  recopy();
  functionMutators[curPos]->mutate();
  curFunction = functionMutators[curPos]->getFunctionName();
  ++curPos;
  if (curPos == functionMutators.size())
  {
    curPos = 0;
  }
}

void Mutator::saveModule(const std::string &outputFileName)
{
  std::error_code ec;
  llvm::raw_fd_ostream fout(outputFileName, ec);
  fout << "//Current seed: " << util::Random::getSeed() << "\n";
  fout << "//============tmp copy============\n";
  tmpCopy->print(fout);
  fout.close();
  llvm::errs() << "file wrote to " << outputFileName << "\n";
}

#undef endl