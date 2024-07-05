// Copyright (c) 2024-present, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_ANALYSES_RECORD_DECL_INTERFACE
#define VAST_INTERFACES_ANALYSES_RECORD_DECL_INTERFACE

include "mlir/IR/OpBase.td"
include "vast/Interfaces/Common.td"

def ASTRecordDeclInterface : VastOpInterface< "ASTRecordDeclInterface" > {
    let description = [{
       Interface for clang's RecordDecl AST node.
    }];

    let methods = [
        InterfaceMethod< "<description>", "int",
                        "fields", (ins), "", [{
            return 42;
        }] >
    ];
}

#endif // VAST_INTERFACES_ANALYSES_RECORD_DECL_INTERFACE
