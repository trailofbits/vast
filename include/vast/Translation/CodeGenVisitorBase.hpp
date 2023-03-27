// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Translation/CodeGenContext.hpp"
#include "vast/Translation/CodeGenMeta.hpp"
#include "vast/Translation/Mangler.hpp"

#include "vast/Util/Common.hpp"

namespace vast::cg {

    template< MetaGeneratorLike MetaGenerator >
    struct CodeGenVisitorBase
    {
        CodeGenVisitorBase(CodeGenContext &ctx, MetaGenerator &meta) : ctx(ctx), meta(meta)
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
