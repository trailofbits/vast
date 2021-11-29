// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Dialect/DLTI/DLTI.h>
VAST_UNRELAX_WARNINGS

#include <type_traits>

namespace vast::dl
{
    using MContext = mlir::MLIRContext;

    // We are currently using `DLTI` dialect to help encoding data layout information,
    // however in the future custom attributes will be probably preferable.
    // Each entry is mapping `hl::Type -> uint32_t` and in the IR it is encoded as
    // attribute of `ModuleOp`.
    // TODO(lukas): Add alignment information & possibly ABI lowering relevant info?
    struct DLEntry
    {
        mlir::Type type;
        uint32_t bw;

        DLEntry(mlir::Type t_, uint32_t bw_) : type(t_), bw(bw_) {}

    private:
        static mlir::Type bw_type(MContext &mctx) { return mlir::IntegerType::get(&mctx, 32); }

        mlir::Attribute wrap_bw(MContext &mctx) const
        {
            // TODO(lukas): There is `UI64Attr` in `IR/OpBase.td` not sure how to include it
            //              though.
            return mlir::IntegerAttr::get(bw_type(mctx), llvm::APInt(32, bw, false));
        }

        static uint32_t unwrap_bw(const mlir::Attribute attr)
        {
            return static_cast< uint32_t >(attr.cast< mlir::IntegerAttr >().getInt());
        }

    public:
        // Construct `DLEntry` from attribute.
        // TODO(lukas): Sanity checks.
        static DLEntry unwrap(const mlir::DataLayoutEntryInterface &attr)
        {
            return DLEntry(attr.getKey().dyn_cast< mlir::Type >(), unwrap_bw(attr.getValue()));
        }

        // Wrap information in this object as `mlir::Attribute`, which is not attached yet
        // to anything.
        mlir::DataLayoutEntryInterface wrap(MContext &mctx) const
        {
            auto as_attr = wrap_bw(mctx);
            return mlir::DataLayoutEntryAttr::get(type, as_attr);
        }
    };
} // namespace vast::dl
