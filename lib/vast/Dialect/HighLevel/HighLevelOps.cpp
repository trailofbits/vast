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
#include <mlir/IR/FunctionImplementation.h>

#include <llvm/Support/ErrorHandling.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/Region.hpp"
#include "vast/Util/TypeUtils.hpp"

#include <optional>
#include <variant>

namespace vast::hl
{
    using FoldResult = mlir::OpFoldResult;

    //===----------------------------------------------------------------------===//
    // FuncOp
    //===----------------------------------------------------------------------===//

    // This function is adapted from CIR:
    //
    // Verifies linkage types, similar to LLVM:
    // - functions don't have 'common' linkage
    // - external functions have 'external' or 'extern_weak' linkage
    logical_result FuncOp::verify() {
        auto linkage = getLinkage();
        constexpr auto common = GlobalLinkageKind::CommonLinkage;
        if (linkage == common) {
            return emitOpError() << "functions cannot have '"
                << stringifyGlobalLinkageKind(common)
                << "' linkage";
        }

        // isExternal(FunctionOpInterface) only checks for empty bodyonly checks for empty body...
        // We need to be able to handle functions with internal linkage without body.
        if (linkage != GlobalLinkageKind::InternalLinkage && isExternal()) {
            constexpr auto external = GlobalLinkageKind::ExternalLinkage;
            constexpr auto weak_external = GlobalLinkageKind::ExternalWeakLinkage;
            if (linkage != external && linkage != weak_external) {
                return emitOpError() << "external functions must have '"
                    << stringifyGlobalLinkageKind(external)
                    << "' or '"
                    << stringifyGlobalLinkageKind(weak_external)
                    << "' linkage";
            }
            return mlir::success();
        }
        return mlir::success();
    }

    ParseResult parseFunctionSignatureAndBody(
        Parser &parser, Attribute &funcion_type, mlir::NamedAttrList &attr_dict, Region &body
    ) {
        llvm::SmallVector< Parser::Argument, 8 > arguments;
        llvm::SmallVector< mlir::DictionaryAttr, 1 > result_attrs;
        llvm::SmallVector< Type, 8 > arg_types;
        llvm::SmallVector< Type, 4 > result_types;

        auto &builder = parser.getBuilder();

        bool is_variadic = false;
        if (mlir::failed(mlir::function_interface_impl::parseFunctionSignature(
            parser, /*allowVariadic=*/false, arguments, is_variadic, result_types, result_attrs
        ))) {
            return mlir::failure();
        }


        for (auto &arg : arguments) {
            arg_types.push_back(arg.type);
        }

        // create parsed function type
        funcion_type = mlir::TypeAttr::get(
            builder.getFunctionType(arg_types, result_types)
        );

        // If additional attributes are present, parse them.
        if (parser.parseOptionalAttrDictWithKeyword(attr_dict)) {
            return mlir::failure();
        }

        // TODO: Add the attributes to the function arguments.
        // VAST_ASSERT(result_attrs.size() == result_types.size());
        // return mlir::function_interface_impl::addArgAndResultAttrs(
        //     builder, state, arguments, result_attrs
        // );

        auto loc = parser.getCurrentLocation();
        auto parse_result = parser.parseOptionalRegion(
            body, arguments, /* enableNameShadowing */false
        );

        if (parse_result.has_value()) {
            if (failed(*parse_result))
                return mlir::failure();
            // Function body was parsed, make sure its not empty.
            if (body.empty())
                return parser.emitError(loc, "expected non-empty function body");
        }

        return mlir::success();
    }

    void printFunctionSignatureAndBody(
        Printer &printer, FuncOp op, Attribute /* funcion_type */, mlir::DictionaryAttr, Region &body
    ) {
        auto fty = op.getFunctionType();
        mlir::function_interface_impl::printFunctionSignature(
            printer, op, fty.getInputs(), /* variadic */false, fty.getResults()
        );

        mlir::function_interface_impl::printFunctionAttributes(
            printer, op, {"linkage", op.getFunctionTypeAttrName() }
        );

        if (!body.empty()) {
            printer.getStream() << " ";
            printer.printRegion( body,
                /* printEntryBlockArgs */false,
                /* printBlockTerminators */true
            );
        }
    }

    FoldResult ConstantOp::fold(FoldAdaptor adaptor) {
        VAST_CHECK(adaptor.getOperands().empty(), "constant has no operands");
        return adaptor.getValue();
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
        // TODO(void-call): Remove if
        if(rty)
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

    mlir::CallInterfaceCallable CallOp::getCallableForCallee()
    {
        return (*this)->getAttrOfType< mlir::SymbolRefAttr >("callee");
    }

    mlir::CallInterfaceCallable IndirectCallOp::getCallableForCallee()
    {
        return (*this)->getOperand(0);
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

    void BinComma::build(Builder &bld, State &st, Type type, Operation *lhsop, Operation *rhsop)
    {
        int32_t lhs_count = lhsop->getNumResults();
        int32_t rhs_count = rhsop->getNumResults();
        if (lhs_count > 0)
            st.addOperands(lhsop->getResult(0));
        if (rhs_count > 0)
            st.addOperands(rhsop->getResult(0));

        st.addAttribute(getOperandSegmentSizesAttrName(st.name),
                        bld.getDenseI32ArrayAttr({ lhs_count, rhs_count })
        );

        st.addTypes(type);
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

        if (!lhs)
            return !rhs || rhs.isa< hl::VoidType, mlir::NoneType >();
        if (!rhs)
            return !lhs || lhs.isa< hl::VoidType, mlir::NoneType >();

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

    FuncOp getCallee(CallOp call)
    {
        auto coi = mlir::cast<mlir::CallOpInterface>(call.getOperation());
        return mlir::dyn_cast_or_null<FuncOp>(coi.resolveCallable());
    }
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "vast/Dialect/HighLevel/HighLevel.cpp.inc"
