// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/TypeVisitor.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"
#include "vast/Translation/CodeGenMeta.hpp"
#include "vast/Translation/CodeGenBuilder.hpp"
#include "vast/Translation/CodeGenVisitorBase.hpp"
#include "vast/Translation/CodeGenVisitorLens.hpp"

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"

namespace vast::hl {

    IntegerKind  get_integer_kind (const clang::BuiltinType *ty);
    FloatingKind get_floating_kind(const clang::BuiltinType *ty);

    SizeParam get_size_attr(const clang::ArrayType *ty, MContext &ctx);

    template< typename Derived >
    struct CodeGenTypeVisitorMixin
        : clang::TypeVisitor< CodeGenTypeVisitorMixin< Derived >, Type >
        , CodeGenVisitorLens< CodeGenTypeVisitorMixin< Derived >, Derived >
        , CodeGenBuilderMixin< CodeGenTypeVisitorMixin< Derived >, Derived >
    {
        using base_type = clang::TypeVisitor< CodeGenTypeVisitorMixin< Derived >, Type >;

        using base_type::Visit;

        using LensType = CodeGenVisitorLens< CodeGenTypeVisitorMixin< Derived >, Derived >;

        using LensType::derived;
        using LensType::context;
        using LensType::mcontext;
        using LensType::visit;

        using qualifiers   = clang::Qualifiers;

        template< typename high_level_type >
        auto type_builder() {
            return this->template make_type< high_level_type >().bind(&mcontext());
        }

        auto with_ucv_qualifiers(auto &&state, bool is_unsigned, qualifiers q) {
            return std::forward< decltype(state) >(state).bind_if( is_unsigned || q.hasConst() || q.hasVolatile(),
                UCVQualifiersAttr::get(&mcontext(), is_unsigned, q.hasConst(), q.hasVolatile())
            );
        }

        auto with_cv_qualifiers(auto &&state, qualifiers q) {
            return std::forward< decltype(state) >(state).bind_if( q.hasConst() || q.hasVolatile(),
                CVQualifiersAttr::get(&mcontext(), q.hasConst(), q.hasVolatile())
            );
        }

        auto with_cvr_qualifiers(auto &&state, qualifiers q) {
            return std::forward< decltype(state) >(state).bind_if( q.hasConst() || q.hasVolatile() || q.hasRestrict(),
                CVRQualifiersAttr::get(&mcontext(), q.hasConst(), q.hasVolatile(), q.hasRestrict())
            );
        }

        auto with_qualifiers(auto &&state, const clang::BuiltinType *ty, qualifiers quals) -> mlir_type {
            return with_cv_qualifiers(std::forward< decltype(state) >(state), quals).freeze();
        }

        template< typename high_level_type >
        auto with_qualifiers(const clang::BuiltinType *ty, qualifiers quals) -> mlir_type;

        template<>
        auto with_qualifiers< VoidType >(const clang::BuiltinType *ty, qualifiers quals) -> mlir_type {
            return with_cv_qualifiers(type_builder< VoidType >(), quals).freeze();
        }

        template<>
        auto with_qualifiers< BoolType >(const clang::BuiltinType *ty, qualifiers quals) -> mlir_type {
            return with_qualifiers(type_builder< BoolType >(), ty, quals);
        }

        template< high_level_integer_type type >
        auto with_qualifiers(const clang::BuiltinType *ty, qualifiers quals) -> mlir_type {
            VAST_ASSERT(ty->isIntegerType());
            return with_ucv_qualifiers(type_builder< type >(), ty->isUnsignedIntegerType(), quals).freeze();
        }

        template< high_level_floating_type type >
        auto with_qualifiers(const clang::BuiltinType *ty, qualifiers quals) -> mlir_type {
            return with_qualifiers(type_builder< type >(), ty, quals);
        }

        auto with_qualifiers(const clang::PointerType *ty, qualifiers quals) -> mlir_type {
            auto pointee = visit(ty->getPointeeType());
            return with_cvr_qualifiers(type_builder< PointerType >().bind(pointee), quals).freeze();
        }

        auto with_qualifiers(const clang::ArrayType *ty, qualifiers quals) -> mlir_type {
            auto element_type = visit(ty->getElementType());
            return with_cvr_qualifiers(type_builder< ArrayType >()
                .bind(get_size_attr(ty, mcontext()))
                .bind(element_type), quals)
                .freeze();
        }

        auto make_name_attr(string_ref name) {
            return mlir::StringAttr::get(&mcontext(), name);
        }

        auto with_qualifiers(const clang::RecordType *ty, qualifiers quals) -> mlir_type {
            auto name = make_name_attr( context().decl_name(ty->getDecl()) );
            return with_cv_qualifiers( type_builder< RecordType >().bind(name), quals ).freeze();
        }

        auto with_qualifiers(const clang::EnumType *ty, qualifiers quals) -> mlir_type {
            auto name = make_name_attr( context().decl_name(ty->getDecl()) );
            return with_cv_qualifiers( type_builder< RecordType >().bind(name), quals ).freeze();
        }

        auto with_qualifiers(const clang::TypedefType *ty, qualifiers quals) -> mlir_type {
            auto name = make_name_attr( ty->getDecl()->getName() );
            return with_cvr_qualifiers( type_builder< TypedefType >().bind(name), quals ).freeze();
        }

        auto with_qualifiers(const clang::ElaboratedType *ty, qualifiers quals) -> mlir_type {
            auto element_type = visit(ty->getNamedType());
            return with_cvr_qualifiers(type_builder< ElaboratedType >().bind(element_type), quals).freeze();
        }

        auto Visit(clang::QualType ty) -> mlir_type {
            auto underlying = ty.getTypePtr();
            auto quals      = ty.getQualifiers();
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

            VAST_UNREACHABLE("unsupported qualified type");
            return Type{};
        }

        auto VisitElaboratedType(const clang::ElaboratedType *ty, qualifiers quals) -> mlir_type {
            return with_qualifiers(ty, quals);
        }

        auto VisitElaboratedType(const clang::ElaboratedType *ty) -> mlir_type {
            return VisitElaboratedType(ty, ty->desugar().getQualifiers());
        }

        auto VisitLValueType(clang::QualType ty) -> mlir_type {
            return LValueType::get(&mcontext(), visit(ty));
        }

        auto VisitFunctionType(const clang::FunctionType *ty) -> mlir_type {
            llvm::SmallVector< Type > args;

            if (auto prototype = clang::dyn_cast< clang::FunctionProtoType >(ty)) {
                for (auto param : prototype->getParamTypes()) {
                    args.push_back(VisitLValueType(param));
                }
            }

            auto rty   = visit(ty->getReturnType());
            auto *mctx = &mcontext();
            return mlir::FunctionType::get(mctx, args, rty);
        }

        auto VisitBuiltinType(const clang::BuiltinType *ty) -> mlir_type {
            return VisitBuiltinType(ty, ty->desugar().getQualifiers());
        }

        auto VisitBuiltinType(const clang::BuiltinType *ty, qualifiers quals) -> mlir_type {
            if (ty->isVoidType()) {
                return with_qualifiers< VoidType >(ty, quals);
            }

            if (ty->isBooleanType()) {
                return with_qualifiers< BoolType >(ty, quals);
            }

            if (ty->isIntegerType()) {
                switch (get_integer_kind(ty)) {
                    case IntegerKind::Char:     return with_qualifiers< CharType >(ty, quals);
                    case IntegerKind::Short:    return with_qualifiers< ShortType >(ty, quals);
                    case IntegerKind::Int:      return with_qualifiers< IntType >(ty, quals);
                    case IntegerKind::Long:     return with_qualifiers< LongType >(ty, quals);
                    case IntegerKind::LongLong: return with_qualifiers< LongLongType >(ty, quals);
                    case IntegerKind::Int128:   return with_qualifiers< Int128Type >(ty, quals);
                }
            }

            if (ty->isFloatingType()) {
                switch (get_floating_kind(ty)) {
                    case FloatingKind::Half:       return with_qualifiers< HalfType >(ty, quals);
                    case FloatingKind::BFloat16:   return with_qualifiers< BFloat16Type >(ty, quals);
                    case FloatingKind::Float:      return with_qualifiers< FloatType >(ty, quals);
                    case FloatingKind::Double:     return with_qualifiers< DoubleType >(ty, quals);
                    case FloatingKind::LongDouble: return with_qualifiers< LongDoubleType >(ty, quals);
                    case FloatingKind::Float128:   return with_qualifiers< Float128Type >(ty, quals);
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
            return ParenType::get(&mcontext(), visit(ty->getInnerType()));
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
            return DecayedType::get(&mcontext(), visit(ty->getDecayedType()));
        }

        auto VisitDecayedType(const clang::DecayedType *ty) -> mlir_type {
            return VisitDecayedType(ty, ty->desugar().getQualifiers());
        }
    };

    template< typename Derived >
    struct CodeGenTypeVisitorWithDataLayoutMixin
        : CodeGenTypeVisitorMixin< Derived >
        , CodeGenVisitorLens< CodeGenTypeVisitorWithDataLayoutMixin< Derived >, Derived >
    {
        using Base = CodeGenTypeVisitorMixin< Derived >;

        using LensType = CodeGenVisitorLens< CodeGenTypeVisitorWithDataLayoutMixin< Derived >, Derived >;

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
            return StoreDataLayout(ty, Base::Visit(ty));
        }

        auto Visit(clang::QualType ty) -> mlir_type {
            auto [underlying, quals] = ty.split();
            return StoreDataLayout(underlying, Base::Visit(ty));
        }
    };

} // namespace vast::hl
