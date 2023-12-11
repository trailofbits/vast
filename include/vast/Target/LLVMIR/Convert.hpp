// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"
#include "vast/Util/Common.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/DialectRegistry.h>
VAST_UNRELAX_WARNINGS

#include <memory>
#include <string>

namespace llvm
{
    class LLVMContext;
    class Module;
} // namespace llvm

namespace mlir
{
    class Operation;
}

namespace vast::target::llvmir
{
    // Lower module into `llvm::Module` - it is expected that `mlir_module` is already
    // lowered as much as possible by vast (for example by calling the `prepare_module`
    // function).
    std::unique_ptr< llvm::Module > translate(
        vast_module mlir_module, llvm::LLVMContext &llvm_ctx
    );

    void register_vast_to_llvm_ir(mlir::DialectRegistry &registry);
    void register_vast_to_llvm_ir(mcontext_t &mctx);

} // namespace vast::target::llvmir
