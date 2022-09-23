#ifndef VAST_TYPED_ATTR_INTERFACE
#define VAST_TYPED_ATTR_INTERFACE

// This is temporary port from BuiltinAttributeInterfaces.td

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TypedAttrInterface
//===----------------------------------------------------------------------===//

def TypedAttrInterface : AttrInterface<"TypedAttr"> {
  let cppNamespace = "::mlir";

  let description = [{
    This interface is used for attributes that have a type. The type of an
    attribute is understood to represent the type of the data contained in the
    attribute and is often used as the type of a value with this data.
  }];

  let methods = [InterfaceMethod<
    "Get the attribute's type",
    "::mlir::Type", "getType"
  >];
}

#endif // VAST_TYPED_ATTR_INTERFACE
