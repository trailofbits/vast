// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Translation/CodeGenTypeVisitor.hpp"

namespace vast::hl {

    using BuiltinType = clang::BuiltinType;

    IntegerKind get_integer_kind(const clang::BuiltinType *ty)
    {
        switch (ty->getKind()) {
            case BuiltinType::Char_U:
            case BuiltinType::UChar:
            case BuiltinType::Char_S:
            case BuiltinType::SChar:
                return IntegerKind::Char;
            case BuiltinType::Short:
            case BuiltinType::UShort:
                return IntegerKind::Short;
            case BuiltinType::Int:
            case BuiltinType::UInt:
                return IntegerKind::Int;
            case BuiltinType::Long:
            case BuiltinType::ULong:
                return IntegerKind::Long;
            case BuiltinType::LongLong:
            case BuiltinType::ULongLong:
                return IntegerKind::LongLong;
            case BuiltinType::Int128:
            case BuiltinType::UInt128:
                return IntegerKind::Int128;
            default:
                VAST_UNREACHABLE("unknown integer kind");
        }
    }

    FloatingKind get_floating_kind(const clang::BuiltinType *ty)
    {
        switch (ty->getKind()) {
            case BuiltinType::Half:
            case BuiltinType::Float16:
                return FloatingKind::Half;
            case BuiltinType::BFloat16:
                return FloatingKind::BFloat16;
            case BuiltinType::Float:
                return FloatingKind::Float;
            case BuiltinType::Double:
                return FloatingKind::Double;
            case BuiltinType::LongDouble:
                return FloatingKind::LongDouble;
            case BuiltinType::Float128:
                return FloatingKind::Float128;
            default:
                VAST_UNREACHABLE("unknown floating kind");
        }
    }

    namespace detail {
        SizeParam get_size_attr(const clang::ConstantArrayType *arr, MContext &ctx) {
            // Represents the canonical version of C arrays with a specified
            // constant size.

            // For example, the canonical type for 'int A[4 + 4*100]' is a
            // ConstantArrayType where the element type is 'int' and the size is
            // 404.
            return SizeParam(arr->getSize().getLimitedValue());
        }

        SizeParam get_size_attr(const clang::DependentSizedArrayType *arr, MContext &ctx) {
            // Represents an array type in C++ whose size is a value-dependent
            // expression.

            // For example:

            // template<typename T, int Size> class array { T data[Size]; };

            // For these types, we won't actually know what the array bound is
            // until template instantiation occurs, at which point this will become
            // either a ConstantArrayType or a VariableArrayType.
            return {};
        }

        SizeParam get_size_attr(const clang::IncompleteArrayType *arr, MContext &ctx) {
            // Represents a C array with an unspecified size.

            // For example 'int A[]' has an IncompleteArrayType where the
            // element type is 'int' and the size is unspecified.
            return {};
        }

        SizeParam get_size_attr(const clang::VariableArrayType *arr, MContext &ctx) {
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

    SizeParam get_size_attr(const clang::ArrayType *ty, MContext &ctx) {
        return llvm::TypeSwitch< const clang::ArrayType *, SizeParam >(ty)
            .Case< clang::ConstantArrayType
                 , clang::DependentSizedArrayType
                 , clang::IncompleteArrayType
                 , clang::VariableArrayType
            >( [&] (const auto *array_type) {
                return detail::get_size_attr(array_type, ctx);
            });
    }

} // namespace vast::hl
