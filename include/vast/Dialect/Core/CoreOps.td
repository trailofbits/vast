// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE_OPS
#define VAST_DIALECT_CORE_OPS

include "mlir/IR/OpBase.td"

include "mlir/IR/RegionKindInterface.td"

include "vast/Dialect/Core/CoreTraits.td"

def Core_ScopeOp : Core_Op< "scope", [NoTerminator, DeclareOpInterfaceMethods<RegionKindInterface>] >
{
  let summary = "VAST scope declaration";
  let description = [{
    Scope operation servers to represent explicitly high-level code scope.
    Other control flow operations represent scopes implicitly.  It is a
    single-region operation.
  }];

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{ $body attr-dict }];
}

def Core_ImplicitReturnOp
  : Core_Op< "implicit.return", [SoftTerminatorTrait, ReturnLikeTrait] >
  , Arguments<(ins Variadic<AnyType>:$result)>
{
  let summary = "VAST implicit return";
  let description = [{
    Op representing return that wasn't explicitely written in the source (e.g. in void fun(){}).
  }];

  let assemblyFormat = "$result `:` type($result) attr-dict";
}

#endif //VAST_DIALECT_CORE_OPS
