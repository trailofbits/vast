// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>

#include <mlir/Target/LLVMIR/Dialect/All.h>

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/JSON.h>
VAST_UNRELAX_WARNINGS

#include <vast/Dialect/HighLevel/HighLevelDialect.hpp>
#include <vast/Dialect/HighLevel/HighLevelOps.hpp>

#include <vast/Util/TypeSwitch.hpp>

#include "PassesDetails.hpp"

namespace vast::hl
{
    struct ExportFnInfo : ExportFnInfoBase< ExportFnInfo >
    {
        void runOnOperation() override
        {
            auto &mctx = this->getContext();
            mlir::ModuleOp mod = this->getOperation();

            std::error_code ec;
            llvm::raw_fd_ostream out("exported_fn_info.json", ec, llvm::sys::fs::OF_Text);
            VAST_ASSERT(!ec);

            llvm::json::Object top;
            for (auto &op : *mod.getBody())
            {
                auto fn = mlir::dyn_cast< mlir::FuncOp >(op);
                if (!fn)
                    continue;
                const auto &dl_analysis = this->getAnalysis< mlir::DataLayoutAnalysis >();
                const auto &dl = dl_analysis.getAtOrAbove(mod);

                llvm::json::Array args;
                for (auto &arg_type : fn.getArgumentTypes())
                    args.push_back(parse_typeinfo(dl, arg_type));

                llvm::json::Array rets;
                for (auto &ret_type : fn.getResultTypes())
                    rets.push_back(parse_typeinfo(dl, ret_type));

                llvm::json::Object current;
                current["rets"] = std::move(rets);
                current["args"] = std::move(args);

                top[fn.getName().str()] = std::move(current);
            }
            out << llvm::formatv("{0:2}",llvm::json::Value(std::move(top)));
        }

        llvm::json::Object parse_typeinfo(
                const mlir::DataLayout &dl,
                mlir::Type hl_type)
        {
            auto ptr_type = [&](auto ptr_type)
            {
                llvm::json::Object out;
                out["type"] = "ptr";
                out["size"] = dl.getTypeSizeInBits(ptr_type);
                out["element_type"] = parse_typeinfo(dl, ptr_type.getElementType());
                return out;
            };

            auto lvalue_type = [&](auto lvalue_ty)
            {
                return parse_typeinfo(dl, lvalue_ty.getElementType());
            };

            auto void_type = [&](auto void_ty)
            {
                llvm::json::Object out;
                out["size"] = 0;
                out["type"] = void_ty.getMnemonic();
                return out;
            };

            auto scalars = [&](auto scalar)
            {
                llvm::json::Object out;
                out["size"] = dl.getTypeSizeInBits(scalar);
                out["type"] = scalar.getMnemonic();
                out["const"] = scalar.isConst();
                out["volatile"] = scalar.isVolatile();
                return out;
            };
            return TypeSwitch< mlir::Type, llvm::json::Object >(hl_type)
                .Case< hl::LValueType >(lvalue_type)
                .Case< hl::PointerType >(ptr_type)
                .Case< hl::VoidType >(void_type)
                .Case(scalar_types{}, scalars)
                .Default([&](auto) { return llvm::json::Object{}; } );
        }
    };

} // namespace vast::hl

std::unique_ptr< mlir::Pass > vast::hl::createExportFnInfoPass()
{
    return std::make_unique< ExportFnInfo >();
}
