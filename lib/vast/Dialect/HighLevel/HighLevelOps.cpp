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

    static ParseResult parseConstantOp(Parser &parser, State &st) {
        Attribute val;
        Type type;
        if (parser.parseAttribute(val, "value", st.attributes) || parser.parseOptionalAttrDict(st.attributes))
            return LogicalResult::failure();
        if (parser.parseOptionalColon() || !parser.parseOptionalType(type).hasValue())
            type = val.getType();
        return parser.addTypeToList(val.getType(), st.types);
    }

    static void printConstantOp(Printer &printer, ConstantOp op)
    {
        printer << op.getOperationName() << " ";
        printer.printAttributeWithoutType(op.valueAttr());
        printer.printOptionalAttrDict(op->getAttrs(), {"value"});
        printer << " : " << op.getType();
    }

    FoldResult ConstantOp::fold(llvm::ArrayRef<Attribute> operands)
    {
        assert(operands.empty() && "const has no operands");
        return value();
    }

    void IfOp::build(Builder &bld, State &st, BuilderCallback condBuilder, BuilderCallback thenBuilder, BuilderCallback elseBuilder)
    {
        assert(condBuilder && "the builder callback for 'condition' block must be present");
        assert(thenBuilder && "the builder callback for 'then' block must be present");

        Builder::InsertionGuard guard(bld);

        detail::build_region(bld, st, condBuilder);
        detail::build_region(bld, st, thenBuilder);
        detail::build_region(bld, st, elseBuilder);
    }

    void WhileOp::build(Builder &bld, State &st, BuilderCallback cond, BuilderCallback body)
    {
        assert(cond && "the builder callback for 'condition' block must be present");
        assert(body && "the builder callback for 'body' must be present");

        Builder::InsertionGuard guard(bld);

        detail::build_region(bld, st, cond);
        detail::build_region(bld, st, body);
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

    void DoOp::build(Builder &bld, State &st, BuilderCallback body, BuilderCallback cond)
    {
        assert(body && "the builder callback for 'body' must be present");
        Builder::InsertionGuard guard(bld);

        detail::build_region(bld, st, body);
        detail::build_region(bld, st, cond);
    }

    void SwitchOp::build(Builder &bld, State &st, BuilderCallback init, BuilderCallback cond, BuilderCallback body)
    {
        assert(cond && "the builder callback for 'condition' block must be present");
        Builder::InsertionGuard guard(bld);

        detail::build_region(bld, st, init);
        detail::build_region(bld, st, cond);
        detail::build_region(bld, st, body);
    }

    void CaseOp::build(Builder &bld, State &st, BuilderCallback lhs, BuilderCallback body)
    {
        assert(lhs && "the builder callback for 'case condition' block must be present");
        Builder::InsertionGuard guard(bld);

        detail::build_region(bld, st, lhs);
        detail::build_region(bld, st, body);
    }

    void DefaultOp::build(Builder &bld, State &st, BuilderCallback body)
    {
        assert(body && "the builder callback for 'body' block must be present");
        Builder::InsertionGuard guard(bld);

        detail::build_region(bld, st, body);
    }
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "vast/Dialect/HighLevel/HighLevel.cpp.inc"
