// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/FunctionImplementation.h>

#include <llvm/Support/ErrorHandling.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelUtils.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/Dialect/Core/CoreAttributes.hpp"
#include "vast/Dialect/Core/CoreDialect.hpp"
#include "vast/Dialect/Core/CoreTypes.hpp"
#include "vast/Dialect/Core/Linkage.hpp"
#include "vast/Dialect/Core/Func.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/Dialect.hpp"
#include "vast/Util/Region.hpp"
#include "vast/Util/TypeUtils.hpp"
#include "vast/Util/Enum.hpp"

#include <optional>
#include <variant>

namespace vast::hl
{
    using FoldResult = mlir::OpFoldResult;

    //===----------------------------------------------------------------------===//
    // FuncOp
    //===----------------------------------------------------------------------===//

    static llvm::StringRef getLinkageAttrNameString() { return "linkage"; }

    logical_result FuncOp::verify() {
        return core::verifyFuncOp(*this);
    }

    ParseResult parseFunctionSignatureAndBody(
        Parser &parser, Attribute &funcion_type, mlir::NamedAttrList &attr_dict, Region &body
    ) {
        return core::parseFunctionSignatureAndBody(parser, funcion_type, attr_dict, body);
    }

    void printFunctionSignatureAndBody(
        Printer &printer, FuncOp op, Attribute function_type,
        mlir::DictionaryAttr dict_attr, Region &body
    ) {
        return core::printFunctionSignatureAndBody(printer, op, function_type, dict_attr, body);
    }

    FoldResult ConstantOp::fold(FoldAdaptor adaptor) {
        VAST_CHECK(adaptor.getOperands().empty(), "constant has no operands");
        return adaptor.getValue();
    }

    bool ConstantOp::isBuildableWith(mlir_attr value, mlir_type type) {
        auto typed = mlir::dyn_cast< mlir::TypedAttr >(value);

        if (!typed || typed.getType() != type) {
            return false;
        }

        return value.hasTrait< core::ConstantLikeAttrTrait >();
    }

    void build_expr_trait(Builder &bld, State &st, Type rty, BuilderCallback expr) {
        VAST_ASSERT(expr && "the builder callback for 'expr' region must be present");
        InsertionGuard guard(bld);
        build_region(bld, st, expr);
        st.addTypes(rty);
    }

    void SizeOfExprOp::build(Builder &bld, State &st, Type rty, BuilderCallback expr) {
        build_expr_trait(bld, st, rty, expr);
    }

    void AlignOfExprOp::build(Builder &bld, State &st, Type rty, BuilderCallback expr) {
        build_expr_trait(bld, st, rty, expr);
    }

    void StmtExprOp::build(Builder &bld, State &st, Type rty, std::unique_ptr< Region > &&region) {
        InsertionGuard guard(bld);
        st.addRegion(std::move(region));
        st.addTypes(rty);
    }

    void VarDeclOp::build(Builder &bld, State &st, Type type, llvm::StringRef name, BuilderCallback init, BuilderCallback alloc) {
        st.addAttribute("name", bld.getStringAttr(name));
        InsertionGuard guard(bld);

        build_region(bld, st, init);
        build_region(bld, st, alloc);

        st.addTypes(type);
    }

    void EnumDeclOp::build(Builder &bld, State &st, llvm::StringRef name, Type type, BuilderCallback constants) {
        st.addAttribute("name", bld.getStringAttr(name));
        st.addAttribute("type", mlir::TypeAttr::get(type));
        InsertionGuard guard(bld);
        build_region(bld, st, constants);
    }

    void EnumDeclOp::build(Builder &bld, State &st, llvm::StringRef name) {
        st.addAttribute("name", bld.getStringAttr(name));
        build_empty_region(bld, st);
    }

    namespace detail {
        void build_record_like_decl(Builder &bld, State &st, llvm::StringRef name, BuilderCallback fields) {
            st.addAttribute("name", bld.getStringAttr(name));

            InsertionGuard guard(bld);
            build_region(bld, st, fields);
        }

        void build_cxx_record_like_decl(Builder &bld, State &st, llvm::StringRef name, BuilderCallback bases, BuilderCallback fields) {
            st.addAttribute("name", bld.getStringAttr(name));

            InsertionGuard guard(bld);
            build_region(bld, st, bases);
            build_region(bld, st, fields);
        }
    } // namespace detail

    void StructDeclOp::build(Builder &bld, State &st, llvm::StringRef name, BuilderCallback fields) {
        detail::build_record_like_decl(bld, st, name, fields);
    }

    void UnionDeclOp::build(Builder &bld, State &st, llvm::StringRef name, BuilderCallback fields) {
        detail::build_record_like_decl(bld, st, name, fields);
    }

    void CxxStructDeclOp::build(Builder &bld, State &st, llvm::StringRef name, BuilderCallback bases, BuilderCallback fields) {
        detail::build_cxx_record_like_decl(bld, st, name, bases, fields);
    }

    void ClassDeclOp::build(Builder &bld, State &st, llvm::StringRef name, BuilderCallback bases, BuilderCallback fields) {
        detail::build_cxx_record_like_decl(bld, st, name, bases, fields);
    }

    mlir::CallInterfaceCallable CallOp::getCallableForCallee() {
        return (*this)->getAttrOfType< mlir::SymbolRefAttr >("callee");
    }

    void CallOp::setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
        setOperand(0, callee.get< mlir_value >());
    }

    mlir::CallInterfaceCallable IndirectCallOp::getCallableForCallee() {
        return (*this)->getOperand(0);
    }

    void IndirectCallOp::setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
        setOperand(0, callee.get< mlir_value >());
    }


    void build_logic_op(Builder &bld, State &st, Type type, BuilderCallback lhs, BuilderCallback rhs)
    {
        VAST_ASSERT(lhs && "the builder callback for 'lhs' region must be present");
        VAST_ASSERT(rhs && "the builder callback for 'rhs' region must be present");

        Builder::InsertionGuard guard(bld);

        build_region(bld, st, lhs);
        build_region(bld, st, rhs);
        st.addTypes(type);
    }

    void BinLAndOp::build(Builder &bld, State &st, Type type, BuilderCallback lhs, BuilderCallback rhs)
    {
        build_logic_op(bld, st, type, lhs, rhs);
    }

    void BinLOrOp::build(Builder &bld, State &st, Type type, BuilderCallback lhs, BuilderCallback rhs)
    {
        build_logic_op(bld, st, type, lhs, rhs);
    }

    mlir::ParseResult IfOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
        std::unique_ptr< mlir::Region > condRegion = std::make_unique< mlir::Region >();
        std::unique_ptr< mlir::Region > thenRegion = std::make_unique< mlir::Region >();
        std::unique_ptr< mlir::Region > elseRegion = std::make_unique< mlir::Region >();

        if (parser.parseRegion(*condRegion))
            return mlir::failure();

        if (parser.parseKeyword("then"))
            return mlir::failure();

        if (parser.parseRegion(*thenRegion))
            return mlir::failure();

        if (thenRegion->empty())
            thenRegion->emplaceBlock();

        if (mlir::succeeded(parser.parseOptionalKeyword("else"))) {

            if (parser.parseRegion(*elseRegion))
                return mlir::failure();

            if (elseRegion->empty())
                elseRegion->emplaceBlock();
        }

        if (parser.parseOptionalAttrDict(result.attributes))
          return mlir::failure();

        result.addRegion(std::move(condRegion));
        result.addRegion(std::move(thenRegion));
        result.addRegion(std::move(elseRegion));
        return mlir::success();
    }

    void IfOp::print(mlir::OpAsmPrinter &odsPrinter) {
        odsPrinter << ' ';
        odsPrinter.printRegion(getCondRegion());
        odsPrinter << ' ' << "then" << ' ';
        odsPrinter.printRegion(getThenRegion());

        if (!getElseRegion().empty()) {
            odsPrinter << ' ' << "else" << ' ';
            odsPrinter.printRegion(getElseRegion());
        }

        llvm::SmallVector< llvm::StringRef, 2 > elidedAttrs;
        odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
    }

    void IfOp::build(Builder &bld, State &st, BuilderCallback condBuilder, BuilderCallback thenBuilder, BuilderCallback elseBuilder)
    {
        VAST_ASSERT(condBuilder && "the builder callback for 'condition' region must be present");
        VAST_ASSERT(thenBuilder && "the builder callback for 'then' region must be present");

        InsertionGuard guard(bld);

        build_region(bld, st, condBuilder);
        build_region(bld, st, thenBuilder);
        build_region(bld, st, elseBuilder);
    }

    void CondOp::build(Builder &bld, State &st, Type type, BuilderCallback condBuilder, BuilderCallback thenBuilder, BuilderCallback elseBuilder)
    {
        VAST_ASSERT(condBuilder && "the builder callback for 'condition' region must be present");
        VAST_ASSERT(thenBuilder && "the builder callback for 'true' region must be present");
        VAST_ASSERT(elseBuilder && "the builder callback for 'false' region must be present");

        InsertionGuard guard(bld);

        build_region(bld, st, condBuilder);
        build_region(bld, st, thenBuilder);
        build_region(bld, st, elseBuilder);
        st.addTypes(type);
    }

    bool CondOp::typesMatch(mlir::Type lhs, mlir::Type rhs)
    {
        namespace tt = mlir::TypeTrait;

        if (auto e = mlir::dyn_cast< hl::ElaboratedType >(lhs))
            return typesMatch(e.getElementType(), rhs);
        if (auto e = mlir::dyn_cast< hl::ElaboratedType >(rhs))
            return typesMatch(lhs, e.getElementType());

        return lhs == rhs
            || all_with_trait< tt::IntegralTypeTrait >(lhs, rhs)
            || any_with_trait< tt::TypedefTrait >(lhs, rhs)
            || any_with_trait< tt::TypeOfTrait >(lhs, rhs)
            || all_with_trait< tt::PointerTypeTrait >(lhs, rhs);
    }

    logical_result CondOp::verifyRegions()
    {
        auto then_type = get_maybe_yielded_type(getThenRegion());
        auto else_type = get_maybe_yielded_type(getElseRegion());

        bool compatible = typesMatch(then_type, else_type);
        if (!compatible)
        {
            VAST_REPORT("Failed to verify that return types {0}, {1} in CondOp regions match. See location {2}",
                then_type, else_type, getLoc());
        }
        return mlir::success(compatible);
    }

    void WhileOp::build(Builder &bld, State &st, BuilderCallback cond, BuilderCallback body)
    {
        VAST_ASSERT(cond && "the builder callback for 'condition' region must be present");
        VAST_ASSERT(body && "the builder callback for 'body' region must be present");

        InsertionGuard guard(bld);

        build_region(bld, st, cond);
        build_region(bld, st, body);
    }

    void ForOp::build(Builder &bld, State &st, BuilderCallback cond, BuilderCallback incr, BuilderCallback body)
    {
        VAST_ASSERT(body && "the builder callback for 'body' region must be present");
        InsertionGuard guard(bld);

        build_region(bld, st, cond);
        build_region(bld, st, incr);
        build_region(bld, st, body);
    }

    void DoOp::build(Builder &bld, State &st, BuilderCallback body, BuilderCallback cond)
    {
        VAST_ASSERT(body && "the builder callback for 'body' region must be present");
        InsertionGuard guard(bld);

        build_region(bld, st, body);
        build_region(bld, st, cond);
    }

    void SwitchOp::build(Builder &bld, State &st, BuilderCallback cond, BuilderCallback body)
    {
        VAST_ASSERT(cond && "the builder callback for 'condition' region must be present");
        InsertionGuard guard(bld);

        build_region(bld, st, cond);
        build_region(bld, st, body);
    }

    void CaseOp::build(Builder &bld, State &st, BuilderCallback lhs, BuilderCallback body)
    {
        VAST_ASSERT(lhs && "the builder callback for 'case condition' region must be present");
        InsertionGuard guard(bld);

        build_region(bld, st, lhs);
        build_region(bld, st, body);
    }

    void DefaultOp::build(Builder &bld, State &st, BuilderCallback body)
    {
        VAST_ASSERT(body && "the builder callback for 'body' region must be present");
        InsertionGuard guard(bld);

        build_region(bld, st, body);
    }

    void LabelStmt::build(Builder &bld, State &st, Value label, BuilderCallback substmt)
    {
        st.addOperands(label);

        VAST_ASSERT(substmt && "the builder callback for 'substmt' region must be present");
        InsertionGuard guard(bld);

        build_region(bld, st, substmt);
    }

    void ExprOp::build(Builder &bld, State &st, Type rty, std::unique_ptr< Region > &&region) {
        InsertionGuard guard(bld);
        st.addRegion(std::move(region));
        st.addTypes(rty);
    }

    void TypeOfExprOp::build(Builder &bld, State &st, llvm::StringRef name, Type type, BuilderCallback expr) {
        InsertionGuard guard(bld);
        st.addAttribute("name", bld.getStringAttr(name));
        st.addAttribute("type", mlir::TypeAttr::get(type));
        build_region(bld, st, expr);
    }

    mlir_type TypeDeclOp::getDefinedType() {
        return hl::RecordType::get(getContext(), getName());
    }

    mlir_type TypeDefOp::getDefinedType() {
        return hl::TypedefType::get(getContext(), getName());
    }

    FuncOp getCallee(CallOp call)
    {
        auto coi = mlir::cast<mlir::CallOpInterface>(call.getOperation());
        return mlir::dyn_cast_or_null<FuncOp>(coi.resolveCallable());
    }

    void AsmOp::build(
            Builder &bld,
            State &st,
            mlir::StringAttr asm_template,
            bool is_volatile,
            bool has_goto,
            llvm::ArrayRef< mlir::Value > outs,
            llvm::ArrayRef< mlir::Value > ins,
            mlir::ArrayAttr out_names,
            mlir::ArrayAttr in_names,
            mlir::ArrayAttr out_constraints,
            mlir::ArrayAttr in_constraints,
            mlir::ArrayAttr clobbers,
            llvm::ArrayRef< mlir::Value > labels)
    {
        st.addAttribute(getAsmTemplateAttrName(st.name), asm_template);
        st.addOperands(outs);
        st.addOperands(ins);
        st.addOperands(labels);

        st.addAttribute(getOperandSegmentSizesAttrName(st.name),
                        bld.getDenseI32ArrayAttr({static_cast<int32_t>(outs.size()),
                                                  static_cast<int32_t>(ins.size()),
                                                  static_cast<int32_t>(labels.size())
                                                 })
                        );

        if (is_volatile)
            st.addAttribute(getIsVolatileAttrName(st.name), bld.getUnitAttr());
        if (has_goto)
            st.addAttribute(getHasGotoAttrName(st.name), bld.getUnitAttr());

        if (outs.size() > 0 && out_names)
            st.addAttribute(getOutputNamesAttrName(st.name), out_names);
        if (ins.size() > 0 && in_names)
            st.addAttribute(getInputNamesAttrName(st.name), in_names);

        if (outs.size() > 0 && out_constraints)
            st.addAttribute(getOutputConstraintsAttrName(st.name), out_constraints);
        if (ins.size() > 0 && in_constraints)
            st.addAttribute(getInputConstraintsAttrName(st.name), in_constraints);

        if (clobbers && clobbers.size())
            st.addAttribute(getClobbersAttrName(st.name), clobbers);
    }

    GRAPH_REGION_OP(FuncOp);
    GRAPH_REGION_OP(StmtExprOp);

    GRAPH_REGION_OP(IfOp);
    GRAPH_REGION_OP(WhileOp);
    GRAPH_REGION_OP(ForOp);
    GRAPH_REGION_OP(DoOp);
    GRAPH_REGION_OP(SwitchOp);
    GRAPH_REGION_OP(CaseOp);
    GRAPH_REGION_OP(DefaultOp);
    GRAPH_REGION_OP(LabelStmt);
    GRAPH_REGION_OP(BreakOp);
    GRAPH_REGION_OP(CondOp);
    GRAPH_REGION_OP(ContinueOp);


}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

using namespace vast::hl;

#define GET_OP_CLASSES
#include "vast/Dialect/HighLevel/HighLevel.cpp.inc"
