// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/ASTContext.h>
#include <clang/Frontend/ASTUnit.h>
#include <clang/Tooling/Tooling.h>
#include <mlir/IR/Builders.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"
#include "vast/Translation/CodeGen.hpp"

#include <filesystem>

namespace vast::repl::codegen {

    std::unique_ptr< clang::ASTUnit > ast_from_source(const std::string &source);

    std::string get_source(std::filesystem::path source);

} // namespace vast::repl::codegen
