// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Translation/CodeGenContext.hpp"
#include "vast/Translation/CodeGenMeta.hpp"

#include "vast/Util/Common.hpp"

namespace vast::hl {

    template< MetaGeneratorLike MetaGenerator >
    struct CodeGenVisitorBase
    {
        CodeGenVisitorBase(CodeGenContext &ctx, MetaGenerator &meta)
            : ctx(ctx), meta(meta)
        {}

        CodeGenContext &ctx;
        MetaGenerator &meta;
    };

    template< MetaGeneratorLike MetaGenerator >
    struct CodeGenVisitorBaseWithBuilder : CodeGenVisitorBase< MetaGenerator >
    {
        using Base = CodeGenVisitorBase< MetaGenerator >;

        CodeGenVisitorBaseWithBuilder(CodeGenContext &ctx, MetaGenerator &meta)
            : Base(ctx, meta), _builder(ctx.getBodyRegion())
        {}

        Builder _builder;
    };

} // namespace vast::hl
