// Copyright (c) 2024, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_DECL_STORAGE_INTERFACE
#define VAST_INTERFACES_DECL_STORAGE_INTERFACE

include "mlir/IR/OpBase.td"
include "vast/Dialect/Core/StorageInfo.td"
include "vast/Dialect/Core/Interfaces/Common.td"

def Core_DeclStorageInterface : Core_OpInterface< "DeclStorageInterface" > {
    let description = [{ Interface for querying info about declaration storage. }];

    let methods = [
        InterfaceMethod<"",
            "bool", "hasLocalStorage", (ins), [{}],
            /*defaultImplementation=*/[{
                switch ($_op.getStorageClass()) {
                    case ::vast::core::StorageClass::sc_none:
                        return !isFileVarDecl() && $_op.getThreadStorageClass() == TSClass::tsc_none;
                    case ::vast::core::StorageClass::sc_register: return isLocalVarDecl();
                    case ::vast::core::StorageClass::sc_auto: return true;
                    case ::vast::core::StorageClass::sc_extern:
                    case ::vast::core::StorageClass::sc_static:
                    case ::vast::core::StorageClass::sc_private_extern: return false;
                }

                VAST_UNIMPLEMENTED_MSG("unknown starage class");
            }]
        >,
        InterfaceMethod<"",
            "bool", "isLocalVarDecl", (ins), [{}],
            /*defaultImplementation=*/[{ return this->isInFunctionOrMethodContext(); }]
        >,
        InterfaceMethod<"",
            "bool", "isStaticLocal", (ins), [{}],
            /*defaultImplementation=*/[{
                if (isFileVarDecl())
                    return false;
                auto sc = $_op.getStorageClass();
                if (sc == ::vast::core::StorageClass::sc_static)
                    return true;
                auto tsc = $_op.getThreadStorageClass();
                return sc == ::vast::core::StorageClass::sc_none && tsc == TSClass::tsc_cxx_thread;
            }]
        >,
        InterfaceMethod<"",
            "bool", "hasExternalStorage", (ins), [{}],
            /*defaultImplementation=*/[{
                auto sc = $_op.getStorageClass();
                return sc == ::vast::core::StorageClass::sc_extern
                    || sc == ::vast::core::StorageClass::sc_private_extern;
            }]
        >,
        InterfaceMethod<"",
            "bool", "hasGlobalStorage", (ins), [{}],
            /*defaultImplementation=*/[{ return !hasLocalStorage(); }]
        >,
        InterfaceMethod<"",
            "::vast::core::StorageDuration", "getStorageDuration", (ins), [{}],
            /*defaultImplementation=*/[{
                if (hasLocalStorage())
                    return StorageDuration::sd_automatic;
                if ($_op.getThreadStorageClass() != TSClass::tsc_none)
                    return StorageDuration::sd_thread;
                return StorageDuration::sd_static;
            }]
        >,
        InterfaceMethod<"",
            "::vast::core::DeclContextKind", "getDeclContextKind", (ins), [{}],
            /*defaultImplementation=*/[{
                auto st = core::get_effective_symbol_table_for< core::var_symbol >(this->getOperation())->get_defining_operation();
                if (mlir::isa< mlir::FunctionOpInterface >(st))
                    return DeclContextKind::dc_function;
                if (st->template hasTrait< core::ScopeLikeTrait >())
                    return DeclContextKind::dc_function;
                if (mlir::isa< core::ModuleOp >(st))
                    return DeclContextKind::dc_translation_unit;
                if (mlir::isa< hl::StructDeclOp >(st))
                    return DeclContextKind::dc_record;
                if (mlir::isa< hl::EnumDeclOp >(st))
                    return DeclContextKind::dc_enum;
                VAST_UNIMPLEMENTED_MSG("unknown declaration context");
            }]
        >,
        InterfaceMethod<"",
            "bool", "isStaticDataMember", (ins), [{}],
            /*defaultImplementation=*/[{ return isInRecordContext(); }]
        >,
        InterfaceMethod<"",
            "bool", "isInFileContext", (ins), [{}],
            /*defaultImplementation=*/[{
                auto kind = getDeclContextKind();
                return kind == DeclContextKind::dc_translation_unit
                    || kind == DeclContextKind::dc_namespace;
            }]
        >,
        InterfaceMethod<"",
            "bool", "isInFunctionOrMethodContext", (ins), [{}],
            /*defaultImplementation=*/[{
                auto kind = getDeclContextKind();
                return kind == DeclContextKind::dc_function
                    || kind == DeclContextKind::dc_method
                    || kind == DeclContextKind::dc_capture;
            }]
        >,
        InterfaceMethod<"",
            "bool", "isInRecordContext", (ins), [{}],
            /*defaultImplementation=*/[{
                return getDeclContextKind() == DeclContextKind::dc_record;
            }]
        >,
        InterfaceMethod<"",
            "bool", "isFileVarDecl", (ins), [{}],
            /*defaultImplementation=*/[{
              return isInFileContext() || isStaticDataMember();
            }]
        >
    ];
}

#endif // VAST_INTERFACES_DECL_STORAGE_INTERFACE
