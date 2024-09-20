// Copyright (c) 2023, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE_COREATTRTYPEBASE_TD
#define VAST_DIALECT_CORE_COREATTRTYPEBASE_TD

include "mlir/IR/AttrTypeBase.td"

class Core_SymbolRefParameter<string desc = "", string value = "">
    : AttrOrTypeParameter< "::llvm::StringRef", desc >
{
  let allocator = [{ $_dst = $_allocator.copyInto($_self); }];
  let printer = [{ $_printer.printSymbolName($_self); }];
  let parser  = [{
    [&] () -> ::mlir::FailureOr<std::string> {
        ::mlir::StringAttr name;
        if (::mlir::failed($_parser.parseSymbolName(name))) {
            return ::mlir::failure();
        }
        return name.getValue().str();
    } ()
  }];
  let cppStorageType = "std::string";
  let defaultValue = value;
}

#endif // VAST_DIALECT_CORE_COREATTRTYPEBASE_TD
