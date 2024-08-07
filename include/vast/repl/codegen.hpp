// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/ASTContext.h>
#include <clang/Frontend/ASTUnit.h>
#include <clang/Tooling/Tooling.h>
#include <mlir/IR/Builders.h>
VAST_UNRELAX_WARNINGS

#include "vast/repl/common.hpp"

#include <filesystem>

#include "vast/Dialect/Core/CoreOps.hpp"

namespace vast::repl::codegen {

    std::unique_ptr< clang::ASTUnit > ast_from_source(string_ref source);

    owning_mlir_module_ref emit_module(const std::filesystem::path &source, mcontext_t &ctx);

} // namespace vast::repl::codegen
