// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Region.h>
#include <clang/AST/Attr.h>
VAST_UNRELAX_WARNINGS

#include <gap/coro/generator.hpp>

#include <vast/Util/TypeList.hpp>

namespace vast::cg
{
    template< typename T >
    gap::generator< T * > filter(auto from) {
        for (auto x : from) {
            if (auto s = dyn_cast< T >(x))
                co_yield s;
        }
    }

    std::string get_namespaced_decl_name(const clang::NamedDecl *decl);
} // namespace vast::cg
