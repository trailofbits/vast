// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

namespace vast::hl
{
    bool isFileContext(DeclContextKind kind) {
        return kind == DeclContextKind::dc_translation_unit
            || kind == DeclContextKind::dc_namespace;
    }

    bool VarDeclOp::isInFileContext() { return isFileContext(getDeclContextKind()); }

    bool isFunctionOrMethodContext(DeclContextKind kind) {
        return kind == DeclContextKind::dc_function
            || kind == DeclContextKind::dc_method
            || kind == DeclContextKind::dc_capture;
    }

    bool VarDeclOp::isInFunctionOrMethodContext() {
        return isFunctionOrMethodContext(getDeclContextKind());
    }

    bool isRecordContext(DeclContextKind kind) { return kind == DeclContextKind::dc_record; }

    bool VarDeclOp::isInRecordContext() { return isRecordContext(getDeclContextKind()); }

    DeclContextKind VarDeclOp::getDeclContextKind() {
        auto st = mlir::SymbolTable::getNearestSymbolTable(*this);
        if (mlir::isa< FuncOp >(st))
            return DeclContextKind::dc_function;
        if (mlir::isa< mlir::ModuleOp >(st))
            return DeclContextKind::dc_translation_unit;
        if (mlir::isa< StructDeclOp >(st))
            return DeclContextKind::dc_record;
        if (mlir::isa< EnumDeclOp >(st))
            return DeclContextKind::dc_enum;
        VAST_UNREACHABLE("unknown declaration context");
    }

    bool VarDeclOp::isStaticDataMember() {
        // If it wasn't static, it would be a FieldDecl.
        return isInRecordContext();
    }

    bool VarDeclOp::isFileVarDecl() {
        return isInFileContext() || isStaticDataMember();
    }

    bool VarDeclOp::isLocalVarDecl() { return isInFunctionOrMethodContext(); }

    bool VarDeclOp::hasLocalStorage() {
        switch (getStorageClass().value()) {
            case StorageClass::sc_none:
                return !isFileVarDecl() && getThreadStorageClass() == TSClass::tsc_none;
            case StorageClass::sc_register: return isLocalVarDecl();
            case StorageClass::sc_auto: return true;
            case StorageClass::sc_extern:
            case StorageClass::sc_static:
            case StorageClass::sc_private_extern: return false;
        }

        VAST_UNREACHABLE("unknown starage class");
    }

    bool VarDeclOp::isStaticLocal() {
        if (isFileVarDecl())
            return false;
        auto sc = getStorageClass();
        if (sc == StorageClass::sc_static)
            return true;
        auto tsc = getThreadStorageClass();
        return sc == StorageClass::sc_none && tsc == TSClass::tsc_cxx_thread;
    }

    bool VarDeclOp::hasExternalStorage() {
        auto sc = getStorageClass();
        return sc == StorageClass::sc_extern || sc == StorageClass::sc_private_extern;
    }

    bool VarDeclOp::hasGlobalStorage() { return !hasLocalStorage(); }

    StorageDuration VarDeclOp::getStorageDuration() {
        if (hasLocalStorage())
            return StorageDuration::sd_automatic;
        if (getThreadStorageClass() != TSClass::tsc_none)
            return StorageDuration::sd_thread;
        return StorageDuration::sd_static;
    }
} // namespace vast::hl
