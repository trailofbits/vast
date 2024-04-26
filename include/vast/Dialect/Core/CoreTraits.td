// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE_CORETRAITS
#define VAST_DIALECT_CORE_CORETRAITS

include "mlir/IR/DialectBase.td"

def SoftTerminatorTrait : NativeOpTrait< "SoftTerminatorTrait" >, StructuralOpTrait {
    let cppNamespace = "::vast::core";
}

def ReturnLikeTrait : NativeOpTrait< "ReturnLikeTrait" >, StructuralOpTrait {
    let cppNamespace = "::vast::core";
}

def ConstantLikeAttrTrait : NativeAttrTrait<"ConstantLikeAttrTrait"> {
  let cppNamespace = "::vast::core";
}

#endif //VAST_DIALECT_CORE_CORETRAITS
