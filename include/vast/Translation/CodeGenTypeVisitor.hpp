// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/TypeVisitor.h>
VAST_UNRELAX_WARNINGS

#include "vast/Translation/CodeGenMeta.hpp"
#include "vast/Translation/CodeGenVisitorBase.hpp"
#include "vast/Translation/CodeGenVisitorLens.hpp"

namespace vast::hl {

    IntegerKind  get_integer_kind (const clang::BuiltinType *ty);
    FloatingKind get_floating_kind(const clang::BuiltinType *ty);

    SizeParam get_size_attr(const clang::ArrayType *ty, MContext &ctx);

    template< typename Derived >
    struct CodeGenTypeVisitorMixin
        : clang::TypeVisitor< CodeGenTypeVisitorMixin< Derived >, Type >
        , CodeGenVisitorLens< CodeGenTypeVisitorMixin< Derived >, Derived >
    {
        using base_type = clang::TypeVisitor< CodeGenTypeVisitorMixin< Derived >, Type >;

        using base_type::Visit;

        using LensType = CodeGenVisitorLens< CodeGenTypeVisitorMixin< Derived >, Derived >;

        using LensType::derived;
        using LensType::context;
        using LensType::mcontext;
        using LensType::visit;

        Type Visit(clang::QualType ty) {
            auto [underlying, quals] = ty.split();
            auto res = visit(underlying);

            // TODO(Heno): add qualifiers
            // if (ty.isConstQualified()) {

            // }

            // if (ty.isRestrictQualified()) {

            // }

            // if (ty.isVolatileQualified()) {

            // }

            return res;
        }

        Type VisitElaboratedType(const clang::ElaboratedType *ty) {
            return visit(ty->getNamedType());
        }

        Type VisitLValueType(clang::QualType ty) {
            return LValueType::get(&mcontext(), visit(ty));
        }

        Type VisitFunctionType(const clang::FunctionType *ty) {
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

        Type VisitBuiltinType(const clang::BuiltinType *ty) {
            auto *mctx = &mcontext();

            if (ty->isVoidType()) {
                return VoidType::get(mctx);
            }

            if (ty->isBooleanType()) {
                return BoolType::get(mctx);
            }

            if (ty->isIntegerType()) {
                auto u = ty->isUnsignedIntegerType();

                switch (get_integer_kind(ty)) {
                    case IntegerKind::Char:     return CharType::get(mctx, u);
                    case IntegerKind::Short:    return ShortType::get(mctx, u);
                    case IntegerKind::Int:      return IntType::get(mctx, u);
                    case IntegerKind::Long:     return LongType::get(mctx, u);
                    case IntegerKind::LongLong: return LongLongType::get(mctx, u);
                    case IntegerKind::Int128:   return Int128Type::get(mctx, u);
                }
            }

            if (ty->isFloatingType()) {
                switch (get_floating_kind(ty)) {
                    case FloatingKind::Half:       return HalfType::get(mctx);
                    case FloatingKind::BFloat16:   return BFloat16Type::get(mctx);
                    case FloatingKind::Float:      return FloatType::get(mctx);
                    case FloatingKind::Double:     return DoubleType::get(mctx);
                    case FloatingKind::LongDouble: return LongDoubleType::get(mctx);
                    case FloatingKind::Float128:   return Float128Type::get(mctx);
                }
            }

            return Type{};
        }

        Type VisitPointerType(const clang::PointerType *ty) {
            return PointerType::get(&mcontext(), visit(ty->getPointeeType()));
        }

        Type VisitArrayType(const clang::ArrayType *ty) {
            auto &ctx = mcontext();
            auto element_type = visit(ty->getElementType());
            return ArrayType::get(&ctx, get_size_attr(ty, ctx), element_type);
        }

        Type VisitTagType(const clang::TagDecl *decl) {
            auto name = context().decl_name(decl);
            return NamedType::get(&mcontext(), name);
        }

        Type VisitRecordType(const clang::RecordType *ty) {
            return VisitTagType(ty->getDecl());
        }

        Type VisitEnumType(const clang::EnumType *ty) {
            return VisitTagType(ty->getDecl());
        }

        Type VisitTypedefType(const clang::TypedefType *ty) {
            auto name = ty->getDecl()->getName();
            return NamedType::get(&mcontext(), name);
        }

        Type VisitFunctionNoProtoType(const clang::FunctionNoProtoType *type) {
            return VisitFunctionType(type);
        }

        Type VisitFunctionProtoType(const clang::FunctionProtoType *type) {
            return VisitFunctionType(type);
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

        bool is_forward_declared(const clang::Type *ty) const {
            if (auto tag = ty->getAsTagDecl()) {
                return !tag->getDefinition();
            }
            return false;
        }

        Type StoreDataLayout(const clang::Type *orig, Type out) {
            if (!orig->isFunctionType() && !is_forward_declared(orig)) {
                context().data_layout().try_emplace(out, orig, acontext());
            }

            return out;
        }

        Type Visit(const clang::Type *ty) {
            return StoreDataLayout(ty, Base::Visit(ty));
        }

        Type Visit(clang::QualType ty) {
            auto [underlying, quals] = ty.split();
            return StoreDataLayout(underlying, Base::Visit(ty));
        }
    };

} // namespace vast::hl
