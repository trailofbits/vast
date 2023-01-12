// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include <llvm/Support/ToolOutputFile.h>

namespace mlir {
    class MLIRContext;
} // namespace mlir

namespace clang {
    class ASTContext;
    class FunctionDecl;
} // namespace clang

namespace vast {

    struct vast_gen_consumer;

    struct vast_generator;

} // namesapce vast
