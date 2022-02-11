// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

namespace vast::hl
{
    bool VarDecl::isLocalVarDecl() { return isInFunctionOrMethodContext(); }

    bool VarDecl::hasLocalStorage() {
        switch (getStorageClass()) {
            case StorageClass::sc_none:
                return !isFileVarDecl() && getThreadStorageClass() == TSClass::tsc_none;
            case StorageClass::sc_register: return isLocalVarDecl();
            case StorageClass::sc_auto: return true;
            case StorageClass::sc_extern:
            case StorageClass::sc_static:
            case StorageClass::sc_private_extern: return false;
        }
    }

    bool VarDecl::isStaticLocal() {
        if (isFileVarDecl())
            return false;
        auto sc = getStorageClass();
        if (sc == StorageClass::sc_static)
            return true;
        auto tsc = getThreadStorageClass();
        return sc == StorageClass::sc_none && tsc == TSClass::tsc_cxx_thread;
    }

    bool VarDecl::hasExternalStorage() {
        auto sc = getStorageClass();
        return sc == StorageClass::sc_extern || sc == StorageClass::sc_private_extern;
    }

    bool VarDecl::hasGlobalStorage() { return !hasLocalStorage(); }

    StorageDuration VarDecl::getStorageDuration() {
        if (hasLocalStorage())
            return StorageDuration::sd_automatic;
        if (getThreadStorageClass() != TSClass::tsc_none)
            return StorageDuration::sd_thread;
        return StorageDuration::sd_static;
    }
} // namespace vast::hl
