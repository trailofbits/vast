// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Translation/CodeGenContext.hpp"
#include "vast/Translation/CodeGenMeta.hpp"

namespace vast::hl {

    template< MetaGeneratorLike MetaGenerator = DefaultMetaGenerator >
    struct CodeGenVisitorBase
    {
        CodeGenVisitorBase(CodeGenContext &ctx, MetaGenerator &meta)
            : ctx(ctx), meta(meta)
        {}

        CodeGenContext &ctx;
        MetaGenerator &meta;
    };

    template< MetaGeneratorLike MetaGenerator = DefaultMetaGenerator >
    struct CodeGenVisitorBaseWithBuilder : CodeGenVisitorBase< MetaGenerator >
    {
        using Base = CodeGenVisitorBase< MetaGenerator >;

        CodeGenVisitorBaseWithBuilder(CodeGenContext &ctx, MetaGenerator &meta)
            : Base(ctx, meta), _builder(ctx.getBodyRegion())
        {}

        mlir::OpBuilder _builder;
    };

} // namespace vast::hl
