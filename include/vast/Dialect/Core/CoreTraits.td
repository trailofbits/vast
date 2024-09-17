// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE_CORETRAITS
#define VAST_DIALECT_CORE_CORETRAITS

include "mlir/IR/DialectBase.td"

class Core_NativeOpTrait<
    string name, list<Trait> traits = [],
    code extraOpDeclaration = [{}],
    code extraOpDefinition = [{}]
> : NativeOpTrait< name, traits, extraOpDeclaration, extraOpDefinition > {
    let cppNamespace = "::vast::core";
}

class Core_ParamNativeOpTrait<
    string name, string params,
    list<Trait> traits = []
> : ParamNativeOpTrait< name, params, traits > {
    let cppNamespace = "::vast::core";
}

class Core_NativeAttrTrait<
    string name,
    code extraAttrDeclaration = [{}],
    code extraAttrDefinition = [{}]
> : NativeAttrTrait<name, extraAttrDeclaration, extraAttrDefinition> {
    let cppNamespace = "::vast::core";
}

def Core_SoftTerminatorTrait : Core_NativeOpTrait< "SoftTerminatorTrait" >, StructuralOpTrait;
def Core_ReturnLikeTrait : Core_NativeOpTrait< "ReturnLikeTrait" >, StructuralOpTrait;

def Core_ConstantLikeAttrTrait : Core_NativeAttrTrait<"ConstantLikeAttrTrait">;

//
// Trait to generalize scope lookups, can be removed once we have single scope operation
//
def Core_ScopeLikeTrait : Core_NativeOpTrait< "ScopeLikeTrait" >;

#endif //VAST_DIALECT_CORE_CORETRAITS
