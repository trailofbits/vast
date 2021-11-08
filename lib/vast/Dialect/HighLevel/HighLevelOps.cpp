// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/Support/ErrorHandling.h>

#include <optional>

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

    void ConstantOp::build(Builder &bld, State &st, bool value)
    {
        auto type = BoolType::get(bld.getContext());
        auto attr = bld.getBoolAttr(value);
        return build(bld, st, type, attr);
    }

    void ConstantOp::build(Builder &bld, State &st, IntegerType type, llvm::APInt value)
    {
        auto ity = mlir::IntegerType::get(bld.getContext(), value.getBitWidth());
        auto attr = bld.getIntegerAttr(ity, value);
        return build(bld, st, type, attr);
    }

    void VarOp::build(Builder &bld, State &st, Type type, llvm::StringRef name)
    {
        st.addAttribute( mlir::SymbolTable::getSymbolAttrName(), bld.getStringAttr(name) );
        st.addAttribute( "type", mlir::TypeAttr::get(type) );
    }

    void VarOp::build(Builder &bld, State &st, Type type, llvm::StringRef name, Value initializer)
    {
        build(bld, st, type, name);
        st.addOperands(initializer);
    }

    static void printVarOp(Printer &printer, VarOp op)
    {
        printer << op.getOperationName() << " ";
        printer.printSymbolName(op.sym_name());
        if (op.initializer()) {
            printer << " = " << op.initializer();
        }
        printer << " : " << op.type();
        printer.printOptionalAttrDict(op->getAttrs(), {"sym_name", "type"});
    }

    static ParseResult parseVarOp(Parser &parser, State &st)
    {
        mlir::StringAttr name;
        if (parser.parseSymbolName(name, mlir::SymbolTable::getSymbolAttrName(), st.attributes))
            return mlir::failure();

        Parser::OperandType operand;
        auto initializer = parser.parseOptionalEqual();
        if (succeeded(initializer)) {
            if (parser.parseOperand(operand))
                return mlir::failure();
        }

        Type type;
        if (parser.parseColonType(type) || parser.parseOptionalAttrDict(st.attributes))
            return mlir::failure();
        st.addAttribute( "type", mlir::TypeAttr::get(type) );

        if (succeeded(initializer)) {
            parser.resolveOperand(operand, type, st.operands);
        }

        return mlir::success();
    }

    static ParseResult parseConstantOp(Parser &parser, State &st)
    {
        mlir::Attribute attr;

        // pass default type so that attribute parser does not try
        // to parse high evel type
        auto i64 = parser.getBuilder().getIntegerType(64);
        if (parser.parseAttribute(attr, i64))
            return mlir::failure();

        Type type;
        if (parser.parseColonType(type) || parser.parseOptionalAttrDict(st.attributes))
            return mlir::failure();

        st.addTypes(type);
        st.addAttribute("value", attr);
        return mlir::success();
    }

    static void printConstantOp(Printer &printer, ConstantOp op)
    {
        printer << op.getOperationName() << " ";
        printer.printAttributeWithoutType(op.valueAttr());
        printer << " : ";
        printer.printType(op.getType());
        printer.printOptionalAttrDict(op->getAttrs(), {"value"});
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
