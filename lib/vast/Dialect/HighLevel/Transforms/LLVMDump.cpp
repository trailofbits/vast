// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>

#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include <llvm/Support/raw_ostream.h>
VAST_UNRELAX_WARNINGS

#include "PassesDetails.hpp"

namespace vast::hl
{
    struct LLVMDump : LLVMDumpBase< LLVMDump >
    {
        void runOnOperation() override;
    };

    void LLVMDump::runOnOperation()
    {
        auto &mctx = this->getContext();
        mlir::ModuleOp op = this->getOperation();

        mlir::registerLLVMDialectTranslation(mctx);
        llvm::LLVMContext lctx;
        auto lmodule = mlir::translateModuleToLLVMIR(op, lctx);
        if (!lmodule)
            return signalPassFailure();

        std::error_code err;
        llvm::raw_fd_stream out("Test.ll", err);
        VAST_CHECK(err, err.message().c_str());
        out << *lmodule;
        out.flush();
    }
}


std::unique_ptr< mlir::Pass > vast::hl::createLLVMDumpPass()
{
    return std::make_unique< LLVMDump >();
}
