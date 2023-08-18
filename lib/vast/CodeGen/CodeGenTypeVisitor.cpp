// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenTypeVisitor.hpp"

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
                VAST_UNREACHABLE("unknown integer kind");
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
                VAST_UNREACHABLE("unknown floating kind");
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

} // namespace vast::hl
