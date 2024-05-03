// Copyright (c) 2024, Trail of Bits, Inc.

#include "vast/CodeGen/DefaultTypeVisitor.hpp"

#include "vast/CodeGen/Util.hpp"

namespace vast::cg {

    using BuiltinType = clang::BuiltinType;

    hl::IntegerKind get_integer_kind(const clang::BuiltinType *ty)
    {
        switch (ty->getKind()) {
            case BuiltinType::Char_U:
            case BuiltinType::UChar:
            case BuiltinType::Char_S:
            case BuiltinType::SChar:
                return hl::IntegerKind::Char;
            case BuiltinType::Short:
            case BuiltinType::UShort:
                return hl::IntegerKind::Short;
            case BuiltinType::Int:
            case BuiltinType::UInt:
                return hl::IntegerKind::Int;
            case BuiltinType::Long:
            case BuiltinType::ULong:
                return hl::IntegerKind::Long;
            case BuiltinType::LongLong:
            case BuiltinType::ULongLong:
                return hl::IntegerKind::LongLong;
            case BuiltinType::Int128:
            case BuiltinType::UInt128:
                return hl::IntegerKind::Int128;
            default:
                VAST_UNIMPLEMENTED_MSG("unknown integer kind");
        }
    }

    hl::FloatingKind get_floating_kind(const clang::BuiltinType *ty)
    {
        switch (ty->getKind()) {
            case BuiltinType::Half:
            case BuiltinType::Float16:
                return hl::FloatingKind::Half;
            case BuiltinType::BFloat16:
                return hl::FloatingKind::BFloat16;
            case BuiltinType::Float:
                return hl::FloatingKind::Float;
            case BuiltinType::Double:
                return hl::FloatingKind::Double;
            case BuiltinType::LongDouble:
                return hl::FloatingKind::LongDouble;
            case BuiltinType::Float128:
                return hl::FloatingKind::Float128;
            default:
                VAST_UNIMPLEMENTED_MSG("unknown floating kind");
        }
    }

    namespace detail {
        hl::SizeParam get_size_attr(const clang::ConstantArrayType *arr, mcontext_t &ctx) {
            // Represents the canonical version of C arrays with a specified
            // constant size.

            // For example, the canonical type for 'int A[4 + 4*100]' is a
            // ConstantArrayType where the element type is 'int' and the size is
            // 404.
            return hl::SizeParam(arr->getSize().getLimitedValue());
        }

        hl::SizeParam get_size_attr(const clang::DependentSizedArrayType *arr, mcontext_t &ctx) {
            // Represents an array type in C++ whose size is a value-dependent
            // expression.

            // For example:

            // template<typename T, int Size> class array { T data[Size]; };

            // For these types, we won't actually know what the array bound is
            // until template instantiation occurs, at which point this will become
            // either a ConstantArrayType or a VariableArrayType.
            return {};
        }

        hl::SizeParam get_size_attr(const clang::IncompleteArrayType *arr, mcontext_t &ctx) {
            // Represents a C array with an unspecified size.

            // For example 'int A[]' has an IncompleteArrayType where the
            // element type is 'int' and the size is unspecified.
            return {};
        }

        hl::SizeParam get_size_attr(const clang::VariableArrayType *arr, mcontext_t &ctx) {
            // Represents a C array with a specified size that is not an
            // integer-constant-expression.

            // For example, 'int s[x+foo()]'. Since the size expression is an
            // arbitrary expression, we store it as such.

            // Note: VariableArrayType's aren't uniqued (since the expressions
            // aren't) and should not be: two lexically equivalent variable array
            // types could mean different things, for example, these variables do
            // not have the same type dynamically:

            // void foo(int x) { int Y[x]; ++x; int Z[x]; }
            return {};
        }
    } // namespace detail

    hl::SizeParam get_size_attr(const clang::ArrayType *ty, mcontext_t &ctx) {
        return llvm::TypeSwitch< const clang::ArrayType *, hl::SizeParam >(ty)
            .Case< clang::ConstantArrayType
                 , clang::DependentSizedArrayType
                 , clang::IncompleteArrayType
                 , clang::VariableArrayType
            >( [&] (const auto *array_type) {
                return detail::get_size_attr(array_type, ctx);
            });
    }

    mlir_type default_type_visitor::Visit(clang_qual_type ty) {
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

    mlir_type default_type_visitor::VisitElaboratedType(const clang::ElaboratedType *ty) {
        return VisitElaboratedType(ty, ty->desugar().getLocalQualifiers());
    }

    mlir_type default_type_visitor::VisitElaboratedType(const clang::ElaboratedType *ty, clang_qualifiers quals) {
        auto element_type  = visit(ty->getNamedType());
        return with_cvr_qualifiers(
            compose_type< hl::ElaboratedType >().bind(element_type), quals
        ).freeze();
    }

    mlir_type default_type_visitor::VisitBuiltinType(const clang::BuiltinType *ty) {
        return VisitBuiltinType(ty, ty->desugar().getLocalQualifiers());
    }

    mlir_type default_type_visitor::VisitBuiltinType(const clang::BuiltinType *ty, clang_qualifiers quals) {
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

    mlir_type default_type_visitor::VisitPointerType(const clang::PointerType *ty) {
        return VisitPointerType(ty, ty->desugar().getLocalQualifiers());
    }

    mlir_type default_type_visitor::VisitPointerType(const clang::PointerType *ty, clang_qualifiers quals) {
        auto pointee = visit(ty->getPointeeType());
        return with_cvr_qualifiers(compose_type< hl::PointerType >().bind(pointee), quals).freeze();
    }

    mlir_type default_type_visitor::VisitArrayType(const clang::ArrayType *ty) {
        return VisitArrayType(ty, clang_qualifiers());
    }

    mlir_type default_type_visitor::VisitArrayType(const clang::ArrayType *ty, clang_qualifiers quals) {
        return with_cvr_qualifiers(
            compose_type< hl::ArrayType >()
                // bind also uninitialized size
                .bind_always(get_size_attr(ty, self.mcontext()))
                .bind(self.visit(ty->getElementType())),
            quals
        ).freeze();
    }

    mlir_type default_type_visitor::VisitRecordType(const clang::RecordType *ty) {
        return VisitRecordType(ty, ty->desugar().getLocalQualifiers());
    }

    mlir_type default_type_visitor::VisitRecordType(const clang::RecordType *ty, clang_qualifiers quals) {
        return mk_record_type(ty, quals);
    }

    mlir_type default_type_visitor::VisitEnumType(const clang::EnumType *ty) {
        return VisitEnumType(ty, ty->desugar().getLocalQualifiers());
    }

    mlir_type default_type_visitor::VisitEnumType(const clang::EnumType *ty, clang_qualifiers quals) {
        return mk_record_type(ty, quals);
    }

    mlir_type default_type_visitor::VisitTypedefType(const clang::TypedefType *ty) {
        return VisitTypedefType(ty, ty->desugar().getLocalQualifiers());
    }

    mlir_type default_type_visitor::VisitTypedefType(const clang::TypedefType *ty, clang_qualifiers quals) {
        auto decl = ty->getDecl();
        if (auto symbol = self.symbol(decl)) {
            auto name = mlir::StringAttr::get(&self.mcontext(), symbol.value());

            // TODO deal with va_list in preprocessing pass
            if (symbol->contains("va_list")) {
                self.visit(decl->getASTContext().getBuiltinVaListDecl());
            }

            return with_cvr_qualifiers(compose_type< hl::TypedefType >().bind(name), quals).freeze();
        }

        return {};
    }

    mlir_type default_type_visitor::VisitParenType(const clang::ParenType *ty) {
        return VisitParenType(ty, ty->desugar().getLocalQualifiers());
    }

    mlir_type default_type_visitor::VisitParenType(const clang::ParenType *ty, clang_qualifiers quals) {
        return hl::ParenType::get(&self.mcontext(), self.visit(ty->getInnerType()));
    }

    mlir_type default_type_visitor::VisitFunctionType(const clang::FunctionType *ty) {
        return self.visit(ty, false /* variadic */);
    }

    mlir_type default_type_visitor::VisitFunctionProtoType(const clang::FunctionProtoType *ty) {
        return VisitFunctionProtoType(ty, ty->desugar().getLocalQualifiers());
    }

    mlir_type default_type_visitor::VisitFunctionProtoType(const clang::FunctionProtoType *ty, clang_qualifiers quals) {
        return self.visit(ty, ty->isVariadic());
    }

    mlir_type default_type_visitor::VisitFunctionNoProtoType(const clang::FunctionNoProtoType *ty) {
        return VisitFunctionNoProtoType(ty, ty->desugar().getLocalQualifiers());
    }

    mlir_type default_type_visitor::VisitFunctionNoProtoType(const clang::FunctionNoProtoType *ty, clang_qualifiers quals) {
        return self.visit(ty, false /* variadic */);
    }

    mlir_type default_type_visitor::VisitDecayedType(const clang::DecayedType *ty) {
        return VisitDecayedType(ty, ty->desugar().getLocalQualifiers());
    }

    mlir_type default_type_visitor::VisitDecayedType(const clang::DecayedType *ty, clang_qualifiers quals) {
        return hl::DecayedType::get(&self.mcontext(), self.visit(ty->getDecayedType()));
    }

    mlir_type default_type_visitor::VisitBlockPointerType(const clang::BlockPointerType *ty) {
        return VisitBlockPointerType(ty, ty->desugar().getLocalQualifiers());
    }

    mlir_type default_type_visitor::VisitBlockPointerType(const clang::BlockPointerType *ty, clang_qualifiers quals) {
        auto pointee = visit(ty->getPointeeType());
        return with_cvr_qualifiers(compose_type< hl::PointerType >().bind(pointee), quals).freeze();
    }

    mlir_type default_type_visitor::VisitAttributedType(const clang::AttributedType *ty) {
        return VisitAttributedType(ty, ty->desugar().getLocalQualifiers());
    }

    mlir_type default_type_visitor::VisitAttributedType(const clang::AttributedType *ty, clang_qualifiers quals) {
        // FIXME add qualifiers?
        return hl::AttributedType::get(&self.mcontext(), self.visit(ty->getModifiedType()));
    }

    mlir_type default_type_visitor::VisitAdjustedType(const clang::AdjustedType *ty) {
        return VisitAdjustedType(ty, ty->desugar().getLocalQualifiers());
    }

    mlir_type default_type_visitor::VisitAdjustedType(const clang::AdjustedType *ty, clang_qualifiers quals) {
        // FIXME add qualifiers?
        auto orig = self.visit(ty->getOriginalType());
        auto adju = self.visit(ty->getAdjustedType());
        return hl::AdjustedType::get(&self.mcontext(), orig, adju);
    }

    mlir_type default_type_visitor::VisitLValueReferenceType(const clang::LValueReferenceType *ty) {
        return VisitLValueReferenceType(ty, ty->desugar().getLocalQualifiers());
    }

    mlir_type default_type_visitor::VisitLValueReferenceType(const clang::LValueReferenceType *ty, clang_qualifiers quals) {
        return create_reference_type< hl::LValueType >(ty, quals);
    }

    mlir_type default_type_visitor::VisitRValueReferenceType(const clang::RValueReferenceType *ty) {
        return VisitRValueReferenceType(ty, ty->desugar().getLocalQualifiers());
    }

    mlir_type default_type_visitor::VisitRValueReferenceType(const clang::RValueReferenceType *ty, clang_qualifiers quals) {
        return create_reference_type< hl::RValueType >(ty, quals);
    }

    mlir_type default_type_visitor::VisitTypeOfExprType(const clang::TypeOfExprType *ty) {
        return VisitTypeOfExprType(ty, ty->desugar().getLocalQualifiers());
    }

    mlir_type default_type_visitor::VisitTypeOfExprType(const clang::TypeOfExprType *ty, clang_qualifiers quals) {
        return {};
    }

    mlir_type default_type_visitor::VisitTypeOfType(const clang::TypeOfType *ty) {
        return VisitTypeOfType(ty, ty->desugar().getLocalQualifiers());
    }

    mlir_type default_type_visitor::VisitTypeOfType(const clang::TypeOfType *ty, clang_qualifiers quals) {
        return {};
    }

} // namespace vast::hl
