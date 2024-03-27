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

#include "vast/CodeGen/Common.hpp"
#include "vast/CodeGen/SymbolGenerator.hpp"

namespace vast::cg
{
    using target_info = clang::TargetInfo;

    using mangle_context = clang::MangleContext;

    struct default_symbol_mangler final : symbol_generator {

        explicit default_symbol_mangler(mangle_context *mangle_context, const std::string &module_name_hash = "")
            : mangle_context(mangle_context), module_name_hash(module_name_hash)
        {}

        std::optional< symbol_name > symbol(clang_global decl) override;
        std::optional< symbol_name > symbol(const clang_named_decl *decl);
        std::optional< symbol_name > symbol(const clang_decl_ref_expr *decl) override;

      private:
        std::optional< std::string > mangle(const clang_named_decl *decl);

        std::unique_ptr< mangle_context > mangle_context;
        const std::string module_name_hash = "";

        // An ordered map of canonical GlobalDecls to their mangled names.
        llvm::MapVector< clang_global, symbol_name > mangled_decl_names;
        llvm::StringMap< clang_global, llvm::BumpPtrAllocator > manglings;
    };

} // namespace vast::cg
