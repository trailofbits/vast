// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

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

    using NameBuilder = llvm::function_ref<void(Value, llvm::StringRef)>;

    using FoldResult = mlir::OpFoldResult;

    FoldResult ConstantOp::fold(mlir::ArrayRef<Attribute> operands) {
        assert(operands.empty() && "constant has no operands");
        return getValue();
    }

    static ParseResult parseConstantOp(Parser &parser, State &st)
    {
        auto loc = parser.getCurrentLocation();
        Attribute attr;
        if (parser.parseAttribute(attr)) {
            parser.emitError(loc, "unknown constant value");
        }

        st.addAttribute("value", attr);

        if (parser.parseOptionalAttrDict(st.attributes)) {
           return mlir::failure();
        }
        return mlir::success();
    }

    static void printConstantOp(Printer &printer, ConstantOp op)
    {
        printer << op.getOperationName() << " ";
        printer.printAttribute(op.valueAttr());
        printer.printOptionalAttrDict(op->getAttrs(), {"value"});
    }

    void build_var_decl(Builder &bld, State &st, Type type, llvm::StringRef name, BuilderCallback initBuilder)
    {
        st.addAttribute( mlir::SymbolTable::getSymbolAttrName(), bld.getStringAttr(name) );
        st.addAttribute( "type", mlir::TypeAttr::get(type) );

        Builder::InsertionGuard guard(bld);
        detail::build_region(bld, st, initBuilder);
    }

    void VarOp::build(Builder &bld, State &st, Type type, llvm::StringRef name, BuilderCallback initBuilder)
    {
        build_var_decl(bld, st, type, name, initBuilder);
    }

    void GlobalOp::build(Builder &bld, State &st, Type type, llvm::StringRef name, BuilderCallback initBuilder)
    {
        build_var_decl(bld, st, type, name, initBuilder);
    }

    mlir::CallInterfaceCallable CallOp::getCallableForCallee()
    {
        return (*this)->getAttrOfType< mlir::SymbolRefAttr >("callee");
    }

    mlir::Operation::operand_range CallOp::getArgOperands() { return operands(); }

    mlir::CallInterfaceCallable IndirectCallOp::getCallableForCallee()
    {
        return (*this)->getOperand(0);
    }

    mlir::Operation::operand_range IndirectCallOp::getArgOperands() { return operands(); }

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
