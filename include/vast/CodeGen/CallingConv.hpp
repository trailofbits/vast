// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
// TODO: This whole file needs translation to vast::hl

namespace vast::cg {

    /// CallingConv Namespace - This namespace contains an enum with a value for the
    /// well-known calling conventions.

    /// LLVM IR allows to use arbitrary numbers as calling convention identifiers.
    /// TODO: What should we do for this for VAST

    /// A set of enums which specify the assigned numeric values for known llvm
    /// calling conventions. LLVM Calling Convention Represetnation
    enum class calling_conv : std::uint8_t {
        /// C - The default llvm calling convention, compatible with C. This
        /// convention is the only calling convention that supports varargs calls. As
        /// with typical C calling conventions, the callee/caller have to tolerate
        /// certain amounts of prototype mismatch.
        C = 0
    };


} // namespace vast::cg
