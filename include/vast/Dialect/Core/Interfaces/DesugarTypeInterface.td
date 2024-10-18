// Copyright (c) 2024, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_DESUGAR_TYPE_INTERFACE
#define VAST_INTERFACES_DESUGAR_TYPE_INTERFACE

include "mlir/IR/OpBase.td"
include "vast/Dialect/Core/Interfaces/Common.td"

def Core_DesugarTypeInterface : Core_TypeInterface< "DesugarTypeInterface" > {
  let description = [{ Interface to mimic Clang AST desugaring. }];
  let methods = [
      InterfaceMethod<"", "mlir::Type", "getDesugaredType", (ins), [{}],
          /*defaultImplementation=*/ [{
              return $_type;
          }]
      >,
      InterfaceMethod<"", "mlir::Type", "getSingleLevelDesugaredType", (ins), [{}],
          /*defaultImplementation=*/[{
              return $_type;
          }]
      >
  ];
}

#endif //VAST_INTERFACES_DESUGAR_TYPE_INTERFACE
