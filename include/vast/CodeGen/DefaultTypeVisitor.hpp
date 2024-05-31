// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"
#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"

namespace vast::cg {

    using clang_qualifiers = clang::Qualifiers;

    struct default_type_visitor : type_visitor_base< default_type_visitor >
    {
        using base = type_visitor_base< default_type_visitor >;
        using base::base;

        using type_visitor_base< default_type_visitor >::Visit;

        template< typename type >
        auto compose_type() {
            return bld.compose_start< type >([&] (auto &&...args) {
                return type::get(&self.mcontext(), std::forward< decltype(args) >(args)...);
            });
        }

        mlir_type visit(const clang_type *type) { return Visit(type); }
        mlir_type visit(clang_qual_type type) { return Visit(type); }

        mlir_type Visit(clang_qual_type ty);

        mlir_type VisitElaboratedType(const clang::ElaboratedType *ty);
        mlir_type VisitElaboratedType(const clang::ElaboratedType *ty, clang_qualifiers quals);

        mlir_type VisitBuiltinType(const clang::BuiltinType *ty);
        mlir_type VisitBuiltinType(const clang::BuiltinType *ty, clang_qualifiers quals);

        mlir_type VisitPointerType(const clang::PointerType *ty);
        mlir_type VisitPointerType(const clang::PointerType *ty, clang_qualifiers quals);

        mlir_type VisitArrayType(const clang::ArrayType *ty);
        mlir_type VisitArrayType(const clang::ArrayType *ty, clang_qualifiers quals);

        mlir_type VisitVectorType(const clang::VectorType *ty);
        mlir_type VisitVectorType(const clang::VectorType *ty, clang_qualifiers quals);

        mlir_type VisitRecordType(const clang::RecordType *ty);
        mlir_type VisitRecordType(const clang::RecordType *ty, clang_qualifiers quals);

        mlir_type VisitEnumType(const clang::EnumType *ty);
        mlir_type VisitEnumType(const clang::EnumType *ty, clang_qualifiers quals);

        mlir_type VisitTypedefType(const clang::TypedefType *ty);
        mlir_type VisitTypedefType(const clang::TypedefType *ty, clang_qualifiers quals);

        mlir_type VisitParenType(const clang::ParenType *ty);
        mlir_type VisitParenType(const clang::ParenType *ty, clang_qualifiers quals);

        mlir_type VisitFunctionType(const clang::FunctionType *ty);

        mlir_type VisitFunctionProtoType(const clang::FunctionProtoType *ty);
        mlir_type VisitFunctionProtoType(const clang::FunctionProtoType *ty, clang_qualifiers quals);

        mlir_type VisitFunctionNoProtoType(const clang::FunctionNoProtoType *ty);
        mlir_type VisitFunctionNoProtoType(const clang::FunctionNoProtoType *ty, clang_qualifiers quals);

        mlir_type VisitDecayedType(const clang::DecayedType *ty);
        mlir_type VisitDecayedType(const clang::DecayedType *ty, clang_qualifiers quals);

        mlir_type VisitBlockPointerType(const clang::BlockPointerType *ty);
        mlir_type VisitBlockPointerType(const clang::BlockPointerType *ty, clang_qualifiers quals);

        mlir_type VisitAttributedType(const clang::AttributedType *ty);
        mlir_type VisitAttributedType(const clang::AttributedType *ty, clang_qualifiers quals);

        mlir_type VisitAdjustedType(const clang::AdjustedType *ty);
        mlir_type VisitAdjustedType(const clang::AdjustedType *ty, clang_qualifiers quals);

        mlir_type VisitLValueReferenceType(const clang::LValueReferenceType *ty);
        mlir_type VisitLValueReferenceType(const clang::LValueReferenceType *ty, clang_qualifiers quals);

        mlir_type VisitRValueReferenceType(const clang::RValueReferenceType *ty);
        mlir_type VisitRValueReferenceType(const clang::RValueReferenceType *ty, clang_qualifiers quals);

        mlir_type VisitTypeOfExprType(const clang::TypeOfExprType *ty);
        mlir_type VisitTypeOfExprType(const clang::TypeOfExprType *ty, clang_qualifiers quals);

        mlir_type VisitTypeOfType(const clang::TypeOfType *ty);
        mlir_type VisitTypeOfType(const clang::TypeOfType *ty, clang_qualifiers quals);

        mlir_type VisitComplexType(const clang::ComplexType *ty);
        mlir_type VisitComplexType(const clang::ComplexType *ty, clang_qualifiers quals);

      private:
        auto with_ucv_qualifiers(auto &&state, bool is_unsigned, clang_qualifiers q) {
            return std::forward< decltype(state) >(state)
                .bind_if( is_unsigned || q.hasConst() || q.hasVolatile(),
                    hl::UCVQualifiersAttr::get(&self.mcontext(), is_unsigned, q.hasConst(), q.hasVolatile())
                );
        }

        auto with_cv_qualifiers(auto &&state, clang_qualifiers q) {
            return std::forward< decltype(state) >(state)
                .bind_if( q.hasConst() || q.hasVolatile(),
                    hl::CVQualifiersAttr::get(&self.mcontext(), q.hasConst(), q.hasVolatile())
                );
        }

        auto with_cvr_qualifiers(auto &&state, clang_qualifiers q) {
            return std::forward< decltype(state) >(state)
                .bind_if( q.hasConst() || q.hasVolatile() || q.hasRestrict(),
                    hl::CVRQualifiersAttr::get(&self.mcontext(), q.hasConst(), q.hasVolatile(), q.hasRestrict())
                );
        }

        mlir_type with_qualifiers(auto &&state, const clang::BuiltinType *ty, clang_qualifiers quals) {
            return with_cv_qualifiers(std::forward< decltype(state) >(state), quals).freeze();
        }

        template< typename high_level_type >
        mlir_type with_qualifiers(const clang::BuiltinType *ty, clang_qualifiers quals);

        template< typename type > requires std::same_as< type, hl::VoidType >
        mlir_type with_qualifiers(const clang::BuiltinType *ty, clang_qualifiers quals)  {
            return with_cv_qualifiers(compose_type< hl::VoidType >(), quals).freeze();
        }

        template< typename type > requires std::same_as< type, hl::BoolType >
        mlir_type with_qualifiers(const clang::BuiltinType *ty, clang_qualifiers quals) {
            return with_qualifiers(compose_type< hl::BoolType >(), ty, quals);
        }

        template< hl::high_level_integer_type type >
        mlir_type with_qualifiers(const clang::BuiltinType *ty, clang_qualifiers quals) {
            VAST_ASSERT(ty->isIntegerType());
            return with_ucv_qualifiers(compose_type< type >(), ty->isUnsignedIntegerType(), quals).freeze();
        }

        template< hl::high_level_floating_type type >
        mlir_type with_qualifiers(const clang::BuiltinType *ty, clang_qualifiers quals) {
            return with_qualifiers(compose_type< type >(), ty, quals);
        }

        template< typename value_type, typename clang_type >
        auto create_reference_type(const clang_type *ty, clang_qualifiers /* quals */) -> mlir_type {
            // FIXME add qualifiers?
            auto pointee = self.visit(ty->getPointeeTypeAsWritten());
            auto ref = hl::ReferenceType::get(&self.mcontext(), pointee);
            return value_type::get(&self.mcontext(), ref);
        }

        template< typename vast_type, typename record_type >
        mlir_type mk_compound_type(const record_type *ty, clang_qualifiers quals) {
            if (auto symbol = self.symbol(ty->getDecl())) {
                auto name = mlir::StringAttr::get(&self.mcontext(), symbol.value());
                return with_cv_qualifiers(compose_type< vast_type >().bind(name), quals).freeze();
            }

            return {};
        }

    };

} // namespace vast::cg
