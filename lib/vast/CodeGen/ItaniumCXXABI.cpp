// Copyright (c) 2023-present, Trail of Bits, Inc.

#include <vast/CodeGen/CXXABI.hpp>

VAST_RELAX_WARNINGS
#include <clang/Basic/TargetInfo.h>
VAST_UNRELAX_WARNINGS

namespace vast::cg
{
    struct vast_itanium_cxx_abi : vast_cxx_abi {};

    vast_cxx_abi *create_vast_itanium_cxx_abi(const acontext_t &actx) {
        switch (actx.getCXXABIKind()) {
            case clang::TargetCXXABI::GenericItanium:
                VAST_UNIMPLEMENTED_IF(actx.getTargetInfo().getTriple().getArch() == llvm::Triple::le32);
                LLVM_FALLTHROUGH;
            case clang::TargetCXXABI::GenericAArch64:
            case clang::TargetCXXABI::AppleARM64:
                // TODO: this isn't quite right, clang uses AppleARM64CXXABI which inherits
                // from ARMCXXABI. We'll have to follow suit.
                return new vast_itanium_cxx_abi();

            default:
                VAST_UNREACHABLE("bad or NYI ABI kind");
        }
    }
} // namespace vast::cg
