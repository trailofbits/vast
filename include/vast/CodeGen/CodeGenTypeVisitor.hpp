// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/TypeVisitor.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"
#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/CodeGenVisitorLens.hpp"

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"

namespace vast::cg {

    hl::IntegerKind  get_integer_kind (const clang::BuiltinType *ty);
    hl::FloatingKind get_floating_kind(const clang::BuiltinType *ty);

    hl::SizeParam get_size_attr(const clang::ArrayType *ty, mcontext_t &ctx);

    template< typename Derived >
    struct CodeGenTypeVisitor
        : clang::TypeVisitor< CodeGenTypeVisitor< Derived >, Type >
        , CodeGenVisitorLens< CodeGenTypeVisitor< Derived >, Derived >
        , CodeGenBuilder< CodeGenTypeVisitor< Derived >, Derived >
    {
        using base_type = clang::TypeVisitor< CodeGenTypeVisitor< Derived >, Type >;

        using base_type::Visit;

        using LensType = CodeGenVisitorLens< CodeGenTypeVisitor< Derived >, Derived >;

        using LensType::derived;
        using LensType::context;
        using LensType::mcontext;
        using LensType::acontext;

        using LensType::meta_location;

        using LensType::visit;

        using Builder = CodeGenBuilder< CodeGenTypeVisitor< Derived >, Derived >;

        using Builder::builder;
        using Builder::make_type_yield_builder;

        using qualifiers   = clang::Qualifiers;

        template< typename high_level_type >
        auto type_builder() {
            return this->template make_type< high_level_type >().bind(&mcontext());
        }

        auto with_ucv_qualifiers(auto &&state, bool is_unsigned, qualifiers q) {
            return std::forward< decltype(state) >(state).bind_if( is_unsigned || q.hasConst() || q.hasVolatile(),
                hl::UCVQualifiersAttr::get(&mcontext(), is_unsigned, q.hasConst(), q.hasVolatile())
            );
        }

        auto with_cv_qualifiers(auto &&state, qualifiers q) {
            return std::forward< decltype(state) >(state).bind_if( q.hasConst() || q.hasVolatile(),
                hl::CVQualifiersAttr::get(&mcontext(), q.hasConst(), q.hasVolatile())
            );
        }

        auto with_cvr_qualifiers(auto &&state, qualifiers q) {
            return std::forward< decltype(state) >(state).bind_if( q.hasConst() || q.hasVolatile() || q.hasRestrict(),
                hl::CVRQualifiersAttr::get(&mcontext(), q.hasConst(), q.hasVolatile(), q.hasRestrict())
            );
        }

        auto with_qualifiers(auto &&state, const clang::BuiltinType *ty, qualifiers quals) -> mlir_type {
            return with_cv_qualifiers(std::forward< decltype(state) >(state), quals).freeze();
        }

        template< typename high_level_type >
        auto with_qualifiers(const clang::BuiltinType *ty, qualifiers quals) -> mlir_type;

        template< typename type > requires std::same_as< type, hl::VoidType >
        auto with_qualifiers(const clang::BuiltinType *ty, qualifiers quals) -> mlir_type {
            return with_cv_qualifiers(type_builder< hl::VoidType >(), quals).freeze();
        }

        template< typename type > requires std::same_as< type, hl::BoolType >
        auto with_qualifiers(const clang::BuiltinType *ty, qualifiers quals) -> mlir_type {
            return with_qualifiers(type_builder< hl::BoolType >(), ty, quals);
        }

        template< hl::high_level_integer_type type >
        auto with_qualifiers(const clang::BuiltinType *ty, qualifiers quals) -> mlir_type {
            VAST_ASSERT(ty->isIntegerType());
            return with_ucv_qualifiers(type_builder< type >(), ty->isUnsignedIntegerType(), quals).freeze();
        }

        template< hl::high_level_floating_type type >
        auto with_qualifiers(const clang::BuiltinType *ty, qualifiers quals) -> mlir_type {
            return with_qualifiers(type_builder< type >(), ty, quals);
        }

        auto with_qualifiers(const clang::PointerType *ty, qualifiers quals) -> mlir_type {
            auto pointee = visit(ty->getPointeeType());
            return with_cvr_qualifiers(type_builder< hl::PointerType >().bind(pointee), quals).freeze();
        }

        auto with_qualifiers(const clang::ArrayType *ty, qualifiers quals) -> mlir_type {
            auto element_type = visit(ty->getElementType());
            return with_cvr_qualifiers(type_builder< hl::ArrayType >()
                .bind(get_size_attr(ty, mcontext()))
                .bind(element_type), quals)
                .freeze();
        }

        auto make_name_attr(string_ref name) {
            return mlir::StringAttr::get(&mcontext(), name);
        }

        auto with_qualifiers(const clang::RecordType *ty, qualifiers quals) -> mlir_type {
            auto name = make_name_attr( context().decl_name(ty->getDecl()) );
            return with_cv_qualifiers( type_builder< hl::RecordType >().bind(name), quals ).freeze();
        }

        auto with_qualifiers(const clang::EnumType *ty, qualifiers quals) -> mlir_type {
            auto name = make_name_attr( context().decl_name(ty->getDecl()) );
            return with_cv_qualifiers( type_builder< hl::RecordType >().bind(name), quals ).freeze();
        }

        auto with_qualifiers(const clang::TypedefType *ty, qualifiers quals) -> mlir_type {
            auto name = make_name_attr( ty->getDecl()->getName() );
            return with_cvr_qualifiers( type_builder< hl::TypedefType >().bind(name), quals ).freeze();
        }

        std::string type_of_expr_name(clang::Expr *underlying) {
            std::string name;
            llvm::raw_string_ostream output(name);
            underlying->printPretty(output, nullptr, acontext().getPrintingPolicy());
            return name;
        }

        auto with_qualifiers(const clang::TypeOfExprType *ty, qualifiers quals) -> mlir_type {
            clang::Expr *underlying_expr = ty->getUnderlyingExpr();
            auto name = derived().type_of_expr_name(underlying_expr);

            this->template make_operation< hl::TypeOfExprOp >()
                .bind(meta_location(underlying_expr))
                .bind(name)
                .bind(visit(underlying_expr->getType()))
                .bind(make_type_yield_builder(underlying_expr))
                .freeze();

            return with_cvr_qualifiers(type_builder< hl::TypeOfExprType >().bind(name), quals)
                .freeze();
        }

        auto with_qualifiers(const clang::TypeOfType *ty, qualifiers quals) -> mlir_type {
            auto type = visit(ty->getUnmodifiedType());
            this->template create< hl::TypeOfTypeOp >(meta_location(ty), type);
            return with_cvr_qualifiers(type_builder< hl::TypeOfTypeType >().bind(type), quals)
                .freeze();
        }

        auto with_qualifiers(const clang::ElaboratedType *ty, qualifiers quals) -> mlir_type {
            auto element_type = visit(ty->getNamedType());
            return with_cvr_qualifiers(type_builder< hl::ElaboratedType >().bind(element_type), quals).freeze();
        }

        auto with_qualifiers(const clang::BlockPointerType *ty, qualifiers quals) -> mlir_type {
            auto pointee = visit(ty->getPointeeType());
            return with_cvr_qualifiers(type_builder< hl::PointerType >().bind(pointee), quals).freeze();
        }

        auto Visit(clang::QualType ty) -> mlir_type {
            auto underlying = ty.getTypePtr();
            auto quals      = ty.getLocalQualifiers();
            if (auto t = llvm::dyn_cast< clang::BuiltinType >(underlying)) {
                return VisitBuiltinType(t, quals);
            }

            if (auto t = llvm::dyn_cast< clang::PointerType >(underlying)) {
                return VisitPointerType(t, quals);
            }

            if (auto t = llvm::dyn_cast< clang::ArrayType >(underlying)) {
                return VisitArrayType(t, quals);
            }

            if (auto t = llvm::dyn_cast< clang::ElaboratedType >(underlying)) {
                return VisitElaboratedType(t, quals);
            }

            if (auto t = llvm::dyn_cast< clang::RecordType >(underlying)) {
                return VisitRecordType(t, quals);
            }

            if (auto t = llvm::dyn_cast< clang::EnumType >(underlying)) {
                return VisitEnumType(t, quals);
            }

            if (auto t = llvm::dyn_cast< clang::TypedefType >(underlying)) {
                return VisitTypedefType(t, quals);
            }

            if (auto t = llvm::dyn_cast< clang::ParenType >(underlying)) {
                return VisitParenType(t, quals);
            }

            if (auto t = llvm::dyn_cast< clang::FunctionProtoType >(underlying)) {
                return VisitFunctionProtoType(t, quals);
            }

            if (auto t = llvm::dyn_cast< clang::FunctionNoProtoType >(underlying)) {
                return VisitFunctionNoProtoType(t, quals);
            }

            if (auto t = llvm::dyn_cast< clang::DecayedType >(underlying)) {
                return VisitDecayedType(t, quals);
            }

            if (auto t = llvm::dyn_cast< clang::BlockPointerType >(underlying)) {
                return VisitBlockPointerType(t, quals);
            }

            if (auto t = llvm::dyn_cast< clang::AttributedType >(underlying)) {
                return VisitAttributedType(t, quals);
            }

            if (auto t = llvm::dyn_cast< clang::AdjustedType >(underlying)) {
                return VisitAdjustedType(t, quals);
            }

            if (auto t = llvm::dyn_cast< clang::LValueReferenceType >(underlying)) {
                return VisitLValueReferenceType(t, quals);
            }

            if (auto t = llvm::dyn_cast< clang::RValueReferenceType >(underlying)) {
                return VisitRValueReferenceType(t, quals);
            }

            if (auto t = llvm::dyn_cast< clang::TypeOfExprType >(underlying)) {
                return VisitTypeOfExprType(t, quals);
            }

            if (auto t = llvm::dyn_cast< clang::TypeOfType >(underlying)) {
                return VisitTypeOfType(t, quals);
            }

            return {};
        }

        auto VisitElaboratedType(const clang::ElaboratedType *ty, qualifiers quals) -> mlir_type {
            return with_qualifiers(ty, quals);
        }

        auto VisitElaboratedType(const clang::ElaboratedType *ty) -> mlir_type {
            return VisitElaboratedType(ty, ty->desugar().getQualifiers());
        }

        auto VisitLValueType(clang::QualType ty) -> mlir_type {
            auto element_type = visit(ty);
            if (mlir::isa< hl::LValueType >(element_type)) {
                return element_type;
            }
            return hl::LValueType::get(&mcontext(), element_type);
        }

        auto VisitFunctionType(const clang::FunctionType *ty) -> mlir_type {
            llvm::SmallVector< Type > args;

            if (auto prototype = clang::dyn_cast< clang::FunctionProtoType >(ty)) {
                for (auto param : prototype->getParamTypes()) {
                    args.push_back(VisitLValueType(param));
                }
            }

            auto *mctx = &mcontext();
            if (ty->getReturnType()->isVoidType()) {
                return mlir::FunctionType::get(mctx, args, {});
            } else {
                return mlir::FunctionType::get(mctx, args, visit(ty->getReturnType()));
            }
        }

        auto VisitBuiltinType(const clang::BuiltinType *ty) -> mlir_type {
            return VisitBuiltinType(ty, ty->desugar().getQualifiers());
        }

        auto VisitBuiltinType(const clang::BuiltinType *ty, qualifiers quals) -> mlir_type {
            if (ty->isVoidType()) {
                return with_qualifiers< hl::VoidType >(ty, quals);
            }

            if (ty->isBooleanType()) {
                return with_qualifiers< hl::BoolType >(ty, quals);
            }

            if (ty->isIntegerType()) {
                switch (get_integer_kind(ty)) {
                    case hl::IntegerKind::Char:     return with_qualifiers< hl::CharType >(ty, quals);
                    case hl::IntegerKind::Short:    return with_qualifiers< hl::ShortType >(ty, quals);
                    case hl::IntegerKind::Int:      return with_qualifiers< hl::IntType >(ty, quals);
                    case hl::IntegerKind::Long:     return with_qualifiers< hl::LongType >(ty, quals);
                    case hl::IntegerKind::LongLong: return with_qualifiers< hl::LongLongType >(ty, quals);
                    case hl::IntegerKind::Int128:   return with_qualifiers< hl::Int128Type >(ty, quals);
                }
            }

            if (ty->isFloatingType()) {
                switch (get_floating_kind(ty)) {
                    case hl::FloatingKind::Half:       return with_qualifiers< hl::HalfType >(ty, quals);
                    case hl::FloatingKind::BFloat16:   return with_qualifiers< hl::BFloat16Type >(ty, quals);
                    case hl::FloatingKind::Float:      return with_qualifiers< hl::FloatType >(ty, quals);
                    case hl::FloatingKind::Double:     return with_qualifiers< hl::DoubleType >(ty, quals);
                    case hl::FloatingKind::LongDouble: return with_qualifiers< hl::LongDoubleType >(ty, quals);
                    case hl::FloatingKind::Float128:   return with_qualifiers< hl::Float128Type >(ty, quals);
                }
            }

            return Type{};
        }

        auto VisitPointerType(const clang::PointerType *ty, qualifiers quals) -> mlir_type {
            return with_qualifiers(ty, quals);
        }

        auto VisitPointerType(const clang::PointerType *ty) -> mlir_type {
            return VisitPointerType(ty, ty->desugar().getQualifiers());
        }

        auto VisitArrayType(const clang::ArrayType *ty, qualifiers quals) -> mlir_type {
            return with_qualifiers(ty, quals);
        }

        auto VisitArrayType(const clang::ArrayType *ty) -> mlir_type {
            return VisitArrayType(ty, qualifiers());
        }

        auto VisitRecordType(const clang::RecordType *ty, qualifiers quals) -> mlir_type {
            return with_qualifiers(ty, quals);
        }

        auto VisitRecordType(const clang::RecordType *ty) -> mlir_type {
            return VisitRecordType(ty, ty->desugar().getQualifiers());
        }

        auto VisitEnumType(const clang::EnumType *ty, qualifiers quals) -> mlir_type {
            return with_qualifiers(ty, quals);
        }

        auto VisitEnumType(const clang::EnumType *ty) -> mlir_type {
            return VisitEnumType(ty, ty->desugar().getQualifiers());
        }

        auto VisitTypedefType(const clang::TypedefType *ty, qualifiers quals) -> mlir_type {
            return with_qualifiers(ty, quals);
        }

        auto VisitTypedefType(const clang::TypedefType *ty) -> mlir_type {
            return VisitTypedefType(ty, ty->desugar().getQualifiers());
        }

        auto VisitParenType(const clang::ParenType *ty, qualifiers /* quals */) -> mlir_type {
            return hl::ParenType::get(&mcontext(), visit(ty->getInnerType()));
        }

        auto VisitParenType(const clang::ParenType *ty) -> mlir_type {
            return VisitParenType(ty, ty->desugar().getQualifiers());
        }

        auto VisitFunctionNoProtoType(const clang::FunctionNoProtoType *ty, qualifiers /* quals */) -> mlir_type {
            return VisitFunctionType(ty);
        }

        auto VisitFunctionNoProtoType(const clang::FunctionNoProtoType *ty) -> mlir_type {
            return VisitFunctionNoProtoType(ty, ty->desugar().getQualifiers());
        }

        auto VisitFunctionProtoType(const clang::FunctionProtoType *ty, qualifiers /* quals */) -> mlir_type {
            return VisitFunctionType(ty);
        }

        auto VisitFunctionProtoType(const clang::FunctionProtoType *ty) -> mlir_type {
            return VisitFunctionProtoType(ty, ty->desugar().getQualifiers());
        }

        auto VisitDecayedType(const clang::DecayedType *ty, qualifiers /* quals */) -> mlir_type {
            return hl::DecayedType::get(&mcontext(), visit(ty->getDecayedType()));
        }

        auto VisitDecayedType(const clang::DecayedType *ty) -> mlir_type {
            return VisitDecayedType(ty, ty->desugar().getQualifiers());
        }

        auto VisitBlockPointerType(const clang::BlockPointerType *ty, qualifiers quals) -> mlir_type {
            return with_qualifiers(ty, quals);
        }

        auto VisitBlockPointerType(const clang::BlockPointerType *ty) -> mlir_type {
            return with_qualifiers(ty, ty->desugar().getQualifiers());
        }

        auto VisitAttributedType(const clang::AttributedType *ty, qualifiers /* quals */) -> mlir_type {
            // FIXME add qualifiers?
            return hl::AttributedType::get(&mcontext(), visit(ty->getModifiedType()));
        }

        auto VisitAttributedType(const clang::AttributedType *ty) -> mlir_type {
            return VisitAttributedType(ty,  ty->desugar().getQualifiers());
        }

        auto VisitAdjustedType(const clang::AdjustedType *ty, qualifiers /* quals */) -> mlir_type {
            // FIXME add qualifiers?
            auto orig = visit(ty->getOriginalType());
            auto adju = visit(ty->getAdjustedType());
            return hl::AdjustedType::get(&mcontext(), orig, adju);
        }

        auto VisitAdjustedType(const clang::AdjustedType *ty) -> mlir_type {
            return VisitAdjustedType(ty,  ty->desugar().getQualifiers());
        }

        template< typename ValueType, typename ClangType >
        auto create_reference_type(const ClangType *ty, qualifiers /* quals */) -> mlir_type {
            auto pointee = visit(ty->getPointeeTypeAsWritten());
            auto ref = hl::ReferenceType::get(&mcontext(), pointee);
            return ValueType::get(&mcontext(), ref);
        }

        auto VisitLValueReferenceType(const clang::LValueReferenceType *ty, qualifiers quals) -> mlir_type {
            return create_reference_type< hl::LValueType >(ty, quals);
        }

        auto VisitLValueReferenceType(const clang::LValueReferenceType *ty) -> mlir_type {
            return VisitLValueReferenceType(ty, ty->desugar().getQualifiers());
        }

        auto VisitRValueReferenceType(const clang::RValueReferenceType *ty, qualifiers quals) -> mlir_type {
            return create_reference_type< hl::RValueType >(ty, quals);
        }

        auto VisitRValueReferenceType(const clang::RValueReferenceType *ty) -> mlir_type {
            return VisitRValueReferenceType(ty, ty->desugar().getQualifiers());
        }

        auto VisitTypeOfExprType(const clang::TypeOfExprType *ty, qualifiers quals = {}) -> mlir_type {
            return with_qualifiers(ty, quals);
        }

        auto VisitTypeOfType(const clang::TypeOfType *ty, qualifiers quals = {}) -> mlir_type {
            return with_qualifiers(ty, quals);
        }
    };

    template< typename Derived >
    struct CodeGenTypeVisitorWithDataLayout
        : CodeGenTypeVisitor< Derived >
        , CodeGenVisitorLens< CodeGenTypeVisitorWithDataLayout< Derived >, Derived >
    {
        using Base = CodeGenTypeVisitor< Derived >;

        using LensType = CodeGenVisitorLens< CodeGenTypeVisitorWithDataLayout< Derived >, Derived >;

        using LensType::context;
        using LensType::acontext;

        bool is_forward_declared(const clang_type *ty) const {
            if (auto tag = ty->getAsTagDecl()) {
                return !tag->getDefinition();
            }
            return false;
        }

        auto StoreDataLayout(const clang_type *orig, mlir_type out) -> mlir_type {
            if (!orig->isFunctionType() && !is_forward_declared(orig)) {
                context().data_layout().try_emplace(out, orig, acontext());
            }

            return out;
        }

        auto Visit(const clang_type *ty) -> mlir_type {
            if (auto gen = Base::Visit(ty)) {
                return StoreDataLayout(ty, gen);
            }
            return {};
        }

        auto Visit(clang::QualType ty) -> mlir_type {
            if (auto gen = Base::Visit(ty)) {
                auto [underlying, quals] = ty.split();
                return StoreDataLayout(underlying, gen);
            }
            return {};
        }
    };

} // namespace vast::cg
