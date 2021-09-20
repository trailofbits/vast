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
        if (!block.empty() && block.back().isKnownTerminator())
            return;
        bld.setInsertionPoint(&block, block.end());
        bld.create< ScopeEndOp >(loc);
    }

    void IfOp::build(Builder &bld, State &st, Value cond, bool withElseRegion)
    {
        build(bld, st, /*resultTypes=*/llvm::None, cond, withElseRegion);
    }

    void IfOp::build(Builder &bld, State &st, TypeRange result, Value cond, bool withElseRegion)
    {
        auto add_terminator = [&] (Builder &nested, Location loc) {
            if (result.empty())
                ensure_terminator(nested.getInsertionBlock()->getParent(), nested, loc);
        };

        auto else_terminator = withElseRegion ? add_terminator : BuilderCallback();
        build(bld, st, result, cond, add_terminator, else_terminator);
    }

    void IfOp::build(Builder &bld, State &st, TypeRange result, Value cond, BuilderCallback thenBuilder, BuilderCallback elseBuilder)
    {
        assert(thenBuilder && "the builder callback for 'then' must be present");

        st.addOperands(cond);
        st.addTypes(result);

        Builder::InsertionGuard guard(bld);
        auto thenRegion = st.addRegion();
        bld.createBlock(thenRegion);
        thenBuilder(bld, st.location);

        auto elseRegion = st.addRegion();
        if (!elseBuilder)
            return;

        bld.createBlock(elseRegion);
        elseBuilder(bld, st.location);
    }

    void IfOp::build(Builder &bld, State &st, Value cond, BuilderCallback thenBuilder, BuilderCallback elseBuilder)
    {
        build(bld, st, TypeRange(), cond, thenBuilder, elseBuilder);
    }
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "vast/Dialect/HighLevel/HighLevel.cpp.inc"
