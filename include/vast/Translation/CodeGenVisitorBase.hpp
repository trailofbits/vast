// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Translation/CodeGenContext.hpp"
#include "vast/Translation/CodeGenMeta.hpp"
#include "vast/Translation/Mangler.hpp"

#include "vast/Util/Common.hpp"

namespace vast::cg {

    template<typename CGContext, MetaGeneratorLike MetaGenerator >
    struct CodeGenVisitorBase
    {
        CodeGenVisitorBase(CGContext &ctx, MetaGenerator &meta) : ctx(ctx), meta(meta)
        {}

        CGContext &ctx;
        MetaGenerator &meta;
    };

    template<typename CGContext, MetaGeneratorLike MetaGenerator >
    struct CodeGenVisitorBaseWithBuilder : CodeGenVisitorBase< CGContext, MetaGenerator >
    {
        using Base = CodeGenVisitorBase< CGContext, MetaGenerator >;

        CodeGenVisitorBaseWithBuilder(CGContext &ctx, MetaGenerator &meta)
            : Base(ctx, meta), _builder(ctx.getBodyRegion())
        {}

        void set_insertion_point_to_start(mlir::Region *region) {
            _builder.setInsertionPointToStart(&region->front());
        }

        void set_insertion_point_to_end(mlir::Region *region) {
            _builder.setInsertionPointToEnd(&region->back());
        }

        void set_insertion_point_to_start(mlir::Block *block) {
            _builder.setInsertionPointToStart(block);
        }

        void set_insertion_point_to_end(mlir::Block *block) {
            _builder.setInsertionPointToEnd(block);
        }

        bool has_insertion_block() {
            return _builder.getInsertionBlock();
        }

        void clear_insertion_point() {
            _builder.clearInsertionPoint();
        }

        insertion_guard make_insertion_guard() {
            return { _builder };
        }

        Builder _builder;
    };

} // namespace vast::cg
