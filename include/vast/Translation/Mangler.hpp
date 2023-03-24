// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/AST/Mangle.h>
#include <clang/Basic/TargetInfo.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/Support/Allocator.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"

namespace vast::cg
{
    struct mangled_name_ref {
        using value_type = string_ref;

        value_type name;

        friend std::strong_ordering operator<=>(const mangled_name_ref &a, const mangled_name_ref &b) {
            return a.name == b.name ? std::strong_ordering::equal :
                   a.name < b.name  ? std::strong_ordering::less
                                    : std::strong_ordering::greater;
        }
    };
} // namespace vast::cg

namespace llvm {
    // Compute a hash_code for a mangled_name_ref

    [[nodiscard]] hash_code hash_value(vast::cg::mangled_name_ref mangled);

    // Provide DenseMapInfo for mangled_name_refs
    template<>
    struct DenseMapInfo< vast::cg::mangled_name_ref, void > {
        using info_type  = vast::cg::mangled_name_ref;
        using value_type = info_type::value_type;
        using base = DenseMapInfo< value_type, void >;

        static inline info_type getEmptyKey() {
            return { base::getEmptyKey() };
        }

        static inline info_type getTombstoneKey() {
            return { base::getTombstoneKey() };
        }

        static unsigned getHashValue(info_type val) {
            return base::getHashValue(val.name);
        }

        static bool isEqual(info_type lhs, info_type rhs) {
            return base::isEqual(lhs.name, rhs.name);
        }
    };
} // namespace llvm

namespace vast::cg
{
    struct CodeGenMangler {

        explicit CodeGenMangler(clang::MangleContext *mangle_context)
            : mangle_context(mangle_context)
        {}

        mangled_name_ref get_mangled_name(
            clang::GlobalDecl decl, const clang::TargetInfo &target_info, const std::string &module_name_hash
        );

        std::optional< clang::GlobalDecl >  lookup_representative_decl(mangled_name_ref name) const;

      private:
        std::string mangle(
            clang::GlobalDecl decl, const std::string &module_name_hash
        ) const;

        std::unique_ptr< clang::MangleContext > mangle_context;

        // An ordered map of canonical GlobalDecls to their mangled names.
        llvm::MapVector< clang::GlobalDecl, mangled_name_ref > mangled_decl_names;
        llvm::StringMap< clang::GlobalDecl, llvm::BumpPtrAllocator > manglings;
    };

} // namespace vast::cg
