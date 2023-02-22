// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Decl.h>
#include <clang/AST/GlobalDecl.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"

namespace vast::cg {

    /// Implements C++ ABI-specific code generation functions.
    struct vast_cxx_abi {

        /// Similar to added_structor_args, but only notes the number of additional
        /// arguments.
        struct added_structor_arg_counts {
            unsigned _prefix = 0;
            unsigned _suffix = 0;

            added_structor_arg_counts() = default;
            added_structor_arg_counts(unsigned p, unsigned s) : _prefix(p), _suffix(s) {}

            static added_structor_arg_counts prefix(unsigned n) { return {n, 0}; }
            static added_structor_arg_counts suffix(unsigned n) { return {0, n}; }
        };

        /// Additional implicit arguments to add to the beginning (Prefix) and end
        /// (suffix) of a constructor / destructor arg list.
        ///
        /// Note that Prefix should actually be inserted *after* the first existing
        /// arg; `this` arguments always come first.
        struct added_structor_args {
            struct arg_t {
                mlir_value value;
                clang::QualType type;
            };

            llvm::SmallVector<arg_t, 1> _prefix;
            llvm::SmallVector<arg_t, 1> _suffix;

            added_structor_args() = default;
            added_structor_args(llvm::SmallVector<arg_t, 1> p, llvm::SmallVector<arg_t, 1> s)
                : _prefix(std::move(p)), _suffix(std::move(s))
            {}

            static added_structor_args prefix(llvm::SmallVector<arg_t, 1> args) {
                return {std::move(args), {}};
            }
            static added_structor_args suffix(llvm::SmallVector<arg_t, 1> args) {
                return {{}, std::move(args)};
            }
        };

        /// Returns true if the given constructor or destructor is one of the kinds
        /// that the ABI says returns 'this' (only applies when called non-virtually
        /// for destructors).
        ///
        /// There currently is no way to indicate if a destructor returns 'this' when
        /// called virtually, and vast generation does not support this case.
        virtual bool has_this_return(clang::GlobalDecl /* decl */) const {
            return false;
        }

        virtual bool has_most_derived_return(clang::GlobalDecl /* decl */) const {
            return false;
        }

        virtual ~vast_cxx_abi() = default;
    };

    /// Creates and Itanium-family ABI
    vast_cxx_abi *create_vast_itanium_cxx_abi(const acontext_t &actx);

} // namespace vast::cg
