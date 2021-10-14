// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "vast/Dialect/HighLevel/HighLevel.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/ErrorHandling.h"

namespace vast::hl
{
    void terminate_body(Builder &bld, Location loc)
    {
        bld.create< ScopeEndOp >(loc);
    }

    void ensure_terminator(Region *region, Builder &bld, Location loc)
    {
        if (region->empty())
            bld.createBlock(region);

        auto &block = region->back();
        if (!block.empty() && block.back().hasTrait< mlir::OpTrait::IsTerminator >())
            return;
        bld.setInsertionPoint(&block, block.end());
        bld.create< ScopeEndOp >(loc);
    }

    namespace detail
    {
        void build_region(Builder &bld, State &st, BuilderCallback callback)
        {
            auto reg = st.addRegion();
            if (callback) {
                bld.createBlock(reg);
                callback(bld, st.location);
            }
        }
    } // namespace detail

    void IfOp::build(Builder &bld, State &st, BuilderCallback condBuilder, BuilderCallback thenBuilder, BuilderCallback elseBuilder)
    {
        assert(condBuilder && "the builder callback for 'condition' block must be present");
        assert(thenBuilder && "the builder callback for 'then' block must be present");

        Builder::InsertionGuard guard(bld);

        detail::build_region(bld, st, condBuilder);
        detail::build_region(bld, st, thenBuilder);
        detail::build_region(bld, st, elseBuilder);
    }

    void WhileOp::build(Builder &bld, State &st, TypeRange result, Value cond, BuilderCallback bodyBuilder)
    {
        assert(bodyBuilder && "the builder callback for 'body' must be present");

        st.addOperands(cond);

        Builder::InsertionGuard guard(bld);
        auto bodyRegion = st.addRegion();
        bld.createBlock(bodyRegion);
        bodyBuilder(bld, st.location);
    }

    void WhileOp::build(Builder &bld, State &st, Value cond, BuilderCallback bodyBuilder)
    {
        build(bld, st, TypeRange(), cond, bodyBuilder);
    }

    void ForOp::build(Builder &bld, State &st, BuilderCallback init, BuilderCallback cond, BuilderCallback incr, BuilderCallback body)
    {
        assert(body && "the builder callback for 'body' must be present");
        Builder::InsertionGuard guard(bld);

        detail::build_region(bld, st, init);
        detail::build_region(bld, st, cond);
        detail::build_region(bld, st, incr);
        detail::build_region(bld, st, body);
    }
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "vast/Dialect/HighLevel/HighLevel.cpp.inc"
