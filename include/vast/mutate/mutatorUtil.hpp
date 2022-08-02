#pragma once
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include <functional>
#include <numeric>
#include <random>
#include <vector>

namespace util {
class Random {
  static std::random_device rd;
  static unsigned seed;
  static std::mt19937 mt;
  static std::uniform_int_distribution<int> intDist;
  static std::uniform_real_distribution<double> dblDist;

public:
  static int getRandomInt() { return intDist(mt); }
  static bool getRandomBool() { return intDist(mt) & 1; }
  static unsigned getSeed() { return seed; }
  static void setSeed(unsigned seed_) {
    seed = seed_;
    mt = std::mt19937(seed);
  }
  static double getRandomDouble() { return dblDist(mt); }
  static float getRandomFloat() { return (float)dblDist(mt); }
};

mlir::BlockArgument addFunctionParameter(mlir::FuncOp func, mlir::Type ty);

template <typename T>
std::vector<mlir::Attribute>
getAttrArrayFromIntArray(mlir::MLIRContext *context, llvm::ArrayRef<T> array) {
  mlir::IntegerType intTy = mlir::IntegerType::get(context, sizeof(T) * 8);
  std::vector<mlir::Attribute> result;
  for (size_t i = 0; i < array.size(); ++i) {
    result.push_back(mlir::IntegerAttr::get(intTy, array[i]));
  }
  return result;
}

template <typename ArrTy, typename T>
ArrTy findRanomInArray(llvm::ArrayRef<ArrTy> arr, T val,
                       std::function<bool(ArrTy, T)> p, ArrTy failed) {
  if (!arr.empty()) {
    size_t idx = Random::getRandomInt() % arr.size();
    for (size_t i = 0; i < arr.size(); ++i, ++idx) {
      if (idx == arr.size()) {
        idx = 0;
      }
      if (p(arr[idx], val)) {
        return arr[idx];
      }
    }
  }
  return failed;
}

template <typename ArrTy> bool isIdenticalArray(llvm::ArrayRef<ArrTy> arr) {
  for (size_t i = 1; i < arr.size(); ++i) {
    if (arr[i] != arr[i - 1]) {
      return false;
    }
  }
  return true;
}

template <typename ArrTy>
llvm::SmallVector<ArrTy> getShuffledArray(llvm::ArrayRef<ArrTy> arr) {
  assert(arr.size() > 1 && "cannot shuffle array with size less than 1");
  llvm::SmallVector<ArrTy> result(arr.begin(), arr.end());
  if (isIdenticalArray<ArrTy>(arr)) {
    return result;
  }
  do {
    std::random_shuffle(result.begin(), result.end());
  } while (result == arr);
  return result;
}

class OperationStateGenerator {
  const static std::string varName;
  static int cnt;

public:
  static mlir::Location getNewLocation(mlir::MLIRContext *context);
  static mlir::OperationState getNewOperationState(mlir::MLIRContext *context,
                                                   std::string name = "");
};
}; // namespace util