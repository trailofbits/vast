// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/CodeGenVisitorList.hpp"

namespace vast::cg {

    struct type_caching_proxy : fallthrough_list_node {

        mlir_type visit(const clang_type *type, scope_context &scope) override {
            return visit_type(type, cache, scope);
        }

        mlir_type visit(clang_qual_type type, scope_context &scope) override {
            return visit_type(type, qual_cache, scope);
        }

        mlir_type visit_type(auto type, auto& cache, scope_context& scope);

        llvm::DenseMap< const clang_type *, mlir_type > cache;
        llvm::DenseMap< clang_qual_type, mlir_type > qual_cache;
    };

    mlir_type type_caching_proxy::visit_type(auto type, auto& cache, scope_context& scope) {
        if (auto value = cache.lookup(type)) {
            return value;
        }

        if (auto result = next->visit(type, scope)) {
            cache.try_emplace(type, result);
            return result;
        } else {
            return {};
        }
    }

} // namespace vast::cg
