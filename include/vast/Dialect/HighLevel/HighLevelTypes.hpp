// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Type.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"
#include "vast/Util/DataLayout.hpp"
#include "vast/Util/TypeList.hpp"
#include "vast/Util/Types.hpp"

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"

#include "vast/Interfaces/TypeQualifiersInterfaces.hpp"
#include "vast/Interfaces/AliasTypeInterface.hpp"
#include "vast/Interfaces/DefaultDataLayoutTypeInterface.hpp"
#include "vast/Interfaces/ElementTypeInterface.hpp"
#include "vast/Interfaces/AST/TypeInterface.hpp"

#include "vast/Dialect/Core/TypeTraits.hpp"
#include "vast/Dialect/Core/CoreTypes.hpp"

namespace vast::hl
{
    using SizeParam = std::optional< std::uint64_t >;

    static auto unknown_size = SizeParam{ std::nullopt };

    mlir_type strip_elaborated(mlir_type);
    mlir_type strip_elaborated(mlir_value);

    mlir_type strip_value_category(mlir_type);
    mlir_type strip_value_category(mlir_value);

    mlir_type strip_complex(mlir_type);
    mlir_type strip_complex(mlir_value);

} // namespace vast::hl

#define GET_TYPEDEF_CLASSES
#include "vast/Dialect/HighLevel/HighLevelTypes.h.inc"

#include <set>

namespace vast::hl
{
    using Context = mlir::MLIRContext;

    using DialectParser = mlir::AsmParser;
    using DialectPrinter = mlir::AsmPrinter;

    using integer_types = util::type_list<
        CharType, ShortType, IntType, LongType, LongLongType, Int128Type
    >;

    template< typename T >
    concept high_level_integer_type = integer_types::contains< T >;

    using floating_types = util::type_list<
        HalfType, BFloat16Type, FloatType, DoubleType, LongDoubleType, Float128Type
    >;

    template< typename T >
    concept high_level_floating_type = floating_types::contains< T >;

    using scalar_types = util::concat<
        util::type_list< BoolType >, integer_types, floating_types
    >;

    template< typename T >
    concept high_level_scalar_type = scalar_types::contains< T >;

    using composite_types = util::type_list< ArrayType >;

    using high_level_types = util::concat<
        scalar_types, composite_types, util::type_list< VoidType >
    >;

    template< typename T >
    concept high_level_type = high_level_types::contains< T >;

    using generic_types = util::type_list< LValueType, PointerType >;

    /* integer types */
    enum class IntegerKind { Char, Short, Int, Long, LongLong, Int128 };

    constexpr inline std::string_view to_string(IntegerKind kind)
    {
        switch (kind) {
            case IntegerKind::Char:     return "char";
            case IntegerKind::Short:    return "short";
            case IntegerKind::Int:      return "int";
            case IntegerKind::Long:     return "long";
            case IntegerKind::LongLong: return "longlong";
            case IntegerKind::Int128:   return "int128";
        }
    }

    /* floating-point types */
    enum class FloatingKind { Half, BFloat16, Float, Double, LongDouble, Float128 };

    inline std::string to_string(FloatingKind kind)
    {
        switch (kind) {
            case FloatingKind::Half:       return "half";
            case FloatingKind::BFloat16:   return "bfloat16";
            case FloatingKind::Float:      return "float";
            case FloatingKind::Double:     return "double";
            case FloatingKind::LongDouble: return "longdouble";
            case FloatingKind::Float128:   return "float128";
        }
    }

    core::FunctionType getFunctionType(mlir_type function_pointer, vast_module mod);

    core::FunctionType getFunctionType(Value callee);
    core::FunctionType getFunctionType(mlir::CallOpInterface call);
    core::FunctionType getFunctionType(mlir::CallInterfaceCallable callee, vast_module mod);

    mlir_type getTypedefType(TypedefType type, vast_module mod);

    // unwraps all typedef aliases to get to real underlying type
    mlir_type getBottomTypedefType(TypedefType def, vast_module mod);

    mlir_type getBottomTypedefType(mlir_type type, vast_module mod);

    // Usually record types are wrapped in `elaborated` or `lvalue` - this helper
    // takes care of traversing them.
    // Returns no value if the type is not a record type.
    // TODO(hl): Invent an interface/trait?
    auto name_of_record(mlir_type t) -> std::optional< std::string >;

    bool isBoolType(mlir_type type);
    bool isIntegerType(mlir_type type);
    bool isFloatingType(mlir_type type);

    bool isSigned(mlir_type type);
    bool isUnsigned(mlir_type type);

    bool isHighLevelType(mlir_type type);

    static inline mlir_type to_std_float_type(mlir_type ty) {
        using fty = mlir::FloatType;
        auto ctx = ty.getContext();
        return llvm::TypeSwitch< mlir_type, mlir_type >(ty)
            .Case< HalfType       >([&] (auto t) { return fty::getF16(ctx);  })
            .Case< BFloat16Type   >([&] (auto t) { return fty::getBF16(ctx); })
            .Case< FloatType      >([&] (auto t) { return fty::getF32(ctx);  })
            .Case< DoubleType     >([&] (auto t) { return fty::getF64(ctx);  })
            .Case< LongDoubleType >([&] (auto t) { return fty::getF80(ctx);  })
            .Case< Float128Type   >([&] (auto t) { return fty::getF128(ctx); })
            .Default([] (auto t) {
                VAST_FATAL("unknown float type: {0}", format_type(t));
                return mlir_type();
            });
    }

} // namespace vast::hl
