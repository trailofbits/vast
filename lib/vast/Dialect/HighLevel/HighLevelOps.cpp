// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/Support/ErrorHandling.h>

#include <optional>
#include <variant>

namespace vast::hl
{
    namespace detail
    {
        Region* build_region(Builder &bld, State &st, BuilderCallback callback)
        {
            auto reg = st.addRegion();
            if (callback.has_value()) {
                bld.createBlock(reg);
                callback.value()(bld, st.location);
            }
            return reg;
        }
    } // namespace detail

    using FoldResult = mlir::OpFoldResult;


    FoldResult ConstantOp::fold(mlir::ArrayRef<Attribute> operands) {
        VAST_CHECK(operands.empty(), "constant has no operands");
        return getValue();
    }


    void build_expr_trait(Builder &bld, State &st, Type rty, BuilderCallback expr) {
        VAST_ASSERT(expr && "the builder callback for 'expr' block must be present");
        Builder::InsertionGuard guard(bld);
        detail::build_region(bld, st, expr);
        st.addTypes(rty);
    }

    void SizeOfExprOp::build(Builder &bld, State &st, Type rty, BuilderCallback expr) {
        build_expr_trait(bld, st, rty, expr);
    }

    void AlignOfExprOp::build(Builder &bld, State &st, Type rty, BuilderCallback expr) {
        build_expr_trait(bld, st, rty, expr);
    }

    void VarDeclOp::build(Builder &bld, State &st, Type type, llvm::StringRef name, BuilderCallback init, BuilderCallback alloc) {
        st.addAttribute("name", bld.getStringAttr(name));
        Builder::InsertionGuard guard(bld);

        detail::build_region(bld, st, init);
        detail::build_region(bld, st, alloc);

        st.addTypes(type);
    }

    void EnumDeclOp::build(Builder &bld, State &st, llvm::StringRef name, Type type, BuilderCallback constants) {
        st.addAttribute("name", bld.getStringAttr(name));
        st.addAttribute("type", mlir::TypeAttr::get(type));
        Builder::InsertionGuard guard(bld);
        detail::build_region(bld, st, constants);
    }

    namespace detail {
        void build_record_like_decl(Builder &bld, State &st, llvm::StringRef name, BuilderCallback fields) {
            st.addAttribute("name", bld.getStringAttr(name));

            Builder::InsertionGuard guard(bld);
            detail::build_region(bld, st, fields);
        }
    } // anamespace detail

    void StructDeclOp::build(Builder &bld, State &st, llvm::StringRef name, BuilderCallback fields) {
        detail::build_record_like_decl(bld, st, name, fields);
    }

    void UnionDeclOp::build(Builder &bld, State &st, llvm::StringRef name, BuilderCallback fields) {
        detail::build_record_like_decl(bld, st, name, fields);
    }

    mlir::CallInterfaceCallable CallOp::getCallableForCallee()
    {
        return (*this)->getAttrOfType< mlir::SymbolRefAttr >("callee");
    }

    mlir::CallInterfaceCallable IndirectCallOp::getCallableForCallee()
    {
        return (*this)->getOperand(0);
    }

    void IfOp::build(Builder &bld, State &st, BuilderCallback condBuilder, BuilderCallback thenBuilder, BuilderCallback elseBuilder)
    {
        VAST_ASSERT(condBuilder && "the builder callback for 'condition' block must be present");
        VAST_ASSERT(thenBuilder && "the builder callback for 'then' block must be present");

        Builder::InsertionGuard guard(bld);

        detail::build_region(bld, st, condBuilder);
        detail::build_region(bld, st, thenBuilder);
        detail::build_region(bld, st, elseBuilder);
    }

    void WhileOp::build(Builder &bld, State &st, BuilderCallback cond, BuilderCallback body)
    {
        VAST_ASSERT(cond && "the builder callback for 'condition' block must be present");
        VAST_ASSERT(body && "the builder callback for 'body' must be present");

        Builder::InsertionGuard guard(bld);

        detail::build_region(bld, st, cond);
        detail::build_region(bld, st, body);
    }

    void ForOp::build(Builder &bld, State &st, BuilderCallback cond, BuilderCallback incr, BuilderCallback body)
    {
        VAST_ASSERT(body && "the builder callback for 'body' must be present");
        Builder::InsertionGuard guard(bld);

        detail::build_region(bld, st, cond);
        detail::build_region(bld, st, incr);
        detail::build_region(bld, st, body);
    }

    void DoOp::build(Builder &bld, State &st, BuilderCallback body, BuilderCallback cond)
    {
        VAST_ASSERT(body && "the builder callback for 'body' must be present");
        Builder::InsertionGuard guard(bld);

        detail::build_region(bld, st, body);
        detail::build_region(bld, st, cond);
    }

    void SwitchOp::build(Builder &bld, State &st, BuilderCallback cond, BuilderCallback body)
    {
        VAST_ASSERT(cond && "the builder callback for 'condition' block must be present");
        Builder::InsertionGuard guard(bld);

        detail::build_region(bld, st, cond);
        detail::build_region(bld, st, body);
    }

    void CaseOp::build(Builder &bld, State &st, BuilderCallback lhs, BuilderCallback body)
    {
        VAST_ASSERT(lhs && "the builder callback for 'case condition' block must be present");
        Builder::InsertionGuard guard(bld);

        detail::build_region(bld, st, lhs);
        detail::build_region(bld, st, body);
    }

    void DefaultOp::build(Builder &bld, State &st, BuilderCallback body)
    {
        VAST_ASSERT(body && "the builder callback for 'body' block must be present");
        Builder::InsertionGuard guard(bld);

        detail::build_region(bld, st, body);
    }

    void LabelStmt::build(Builder &bld, State &st, Value label, BuilderCallback substmt)
    {
        st.addOperands(label);

        VAST_ASSERT(substmt && "the builder callback for 'substmt' block must be present");
        Builder::InsertionGuard guard(bld);

        detail::build_region(bld, st, substmt);
    }

    void ExprOp::build(Builder &bld, State &st, Type rty, std::unique_ptr< Region > &&region) {
        Builder::InsertionGuard guard(bld);
        st.addRegion(std::move(region));
        st.addTypes(rty);
    }
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "vast/Dialect/HighLevel/HighLevel.cpp.inc"
