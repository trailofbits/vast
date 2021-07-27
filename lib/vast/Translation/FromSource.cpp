// Copyright (c) 2021-present, Trail of Bits, Inc.

#include <mlir/Translation.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/Support/SourceMgr.h>

#include <memory>

namespace vast
{
    mlir::LogicalResult registerFromSourceParser()
    {
        mlir::TranslateToMLIRRegistration from_source( "from-source",
            [] (llvm::SourceMgr &mgr, mlir::MLIRContext *ctx) -> mlir::OwningModuleRef {
                return {};
            }
        );

        return mlir::success();
    }
} // namespace vast