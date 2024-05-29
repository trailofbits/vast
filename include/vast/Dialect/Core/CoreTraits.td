// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE_CORETRAITS
#define VAST_DIALECT_CORE_CORETRAITS

include "mlir/IR/DialectBase.td"

class CoreNativeOpTrait<
    string name, list<Trait> traits = [],
    code extraOpDeclaration = [{}],
    code extraOpDefinition = [{}]
> : NativeOpTrait< name, traits, extraOpDeclaration, extraOpDefinition > {
    let cppNamespace = "::vast::core";
}

class CoreParamNativeOpTrait<
    string name, string params,
    list<Trait> traits = []
> : ParamNativeOpTrait< name, params, traits > {
    let cppNamespace = "::vast::core";
}

class CoreNativeAttrTrait<
    string name,
    code extraAttrDeclaration = [{}],
    code extraAttrDefinition = [{}]
> : NativeAttrTrait<name, extraAttrDeclaration, extraAttrDefinition> {
    let cppNamespace = "::vast::core";
}

def SoftTerminatorTrait : CoreNativeOpTrait< "SoftTerminatorTrait" >, StructuralOpTrait;
def ReturnLikeTrait : CoreNativeOpTrait< "ReturnLikeTrait" >, StructuralOpTrait;

def ConstantLikeAttrTrait : CoreNativeAttrTrait<"ConstantLikeAttrTrait">;

#endif //VAST_DIALECT_CORE_CORETRAITS
