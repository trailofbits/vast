// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/ASTContext.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"

#include <type_traits>

namespace vast::dl
{
    // We are currently using `DLTI` dialect to help encoding data layout information,
    // however in the future custom attributes will be probably preferable.
    // Each entry is mapping `hl::Type -> uint32_t` and in the IR it is encoded as
    // attribute of `ModuleOp`.
    // TODO(lukas): Add alignment information & possibly ABI lowering relevant info?
    struct DLEntry
    {
        using bitwidth_t = uint32_t;

        mlir_type type;
        bitwidth_t bw;

        DLEntry(mlir_type type, bitwidth_t bw) : type(type), bw(bw) {}

    private:
        static mlir_type bw_type(mcontext_t &mctx) { return mlir::IntegerType::get(&mctx, 32); }

        mlir::Attribute wrap_bw(mcontext_t &mctx) const
        {
            // TODO(lukas): There is `UI64Attr` in `IR/OpBase.td` not sure how to include it
            //              though.
            return mlir::IntegerAttr::get(bw_type(mctx), llvm::APInt(32, bw));
        }

        static bitwidth_t unwrap_bw(const mlir::Attribute attr)
        {
            return static_cast< bitwidth_t >(attr.cast< mlir::IntegerAttr >().getInt());
        }

    public:
        // Construct `DLEntry` from attribute.
        // TODO(lukas): Sanity checks.
        static DLEntry unwrap(const mlir::DataLayoutEntryInterface &attr)
        {
            return DLEntry(attr.getKey().dyn_cast< mlir_type >(), unwrap_bw(attr.getValue()));
        }

        // Wrap information in this object as `mlir::Attribute`, which is not attached yet
        // to anything.
        mlir::DataLayoutEntryInterface wrap(mcontext_t &mctx) const
        {
            auto as_attr = wrap_bw(mctx);
            return mlir::DataLayoutEntryAttr::get(type, as_attr);
        }
    };

    // For each type remember its data layout information.
    struct DataLayoutBlueprint {
        bool try_emplace(mlir_type mty, const clang::Type *aty, const acontext_t &actx) {
            // For other types this should be good-enough for now
            auto info = actx.getTypeInfo(aty);
            auto bw   = static_cast< uint32_t >(info.Width);
            return std::get< 1 >(entries.try_emplace(mty, dl::DLEntry{ mty, bw }));
        }

        llvm::DenseMap< mlir_type, dl::DLEntry > entries;
    };

    template< typename Stream >
    auto operator<<(Stream &os, const DataLayoutBlueprint &dl) -> decltype(os << "") {
        for (const auto &[ty, sizes] : dl.entries) {
            os << ty << " ";
            const auto &[byte_s, bit_s] = sizes;
            os << llvm::formatv("[ {}, {} ]\n", byte_s, bit_s);
        }
        return os;
    }

} // namespace vast::dl
