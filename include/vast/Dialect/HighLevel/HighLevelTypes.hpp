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

#include "vast/Dialect/Core/TypeTraits.hpp"

namespace vast::hl
{
    template< typename ConcreteTy >
    struct DefaultDL {
        using dl_t = mlir::DataLayout;
        using dl_entries_ref = mlir::DataLayoutEntryListRef;

        static unsigned getTypeSizeInBits(const dl_t &dl, dl_entries_ref entries)
        {
            VAST_CHECK(entries.size() != 0,
                "Query for getTypeSizeInBits for {0} failed: Must have at least one entry!",
                format_type(ConcreteTy{})
            );

            std::optional<uint32_t> out;
            auto handle_entry = [&](auto &dl_entry) {
                if (!out) out = dl_entry.bw;
                VAST_CHECK(*out == dl_entry.bw, "Inconsistent entries");
            };
            apply_on_valid_entries(entries, handle_entry);
            VAST_CHECK(out.has_value(), "getTypeSizeBits entries did not yield result.");
            return *out;
        }

        static unsigned getABIAlignment(const dl_t &dl, dl_entries_ref entries)
        {
            VAST_UNIMPLEMENTED;
        }

        static unsigned getPreferredAlignment(const dl_t &dl, dl_entries_ref entries)
        {
            VAST_UNIMPLEMENTED;
        }

        static void apply_on_valid_entries(dl_entries_ref entries, auto &f)
        {
            for (const auto &entry : entries)
            {
                auto dl_entry = dl::DLEntry::unwrap(entry);
                if (is_valid_entry_type(dl_entry.type))
                    f(dl_entry);
            }
        }

        // TODO(lukas): This may no longer be needed.
        static bool is_valid_entry_type(mlir::Type t)
        {
            return t.isa< ConcreteTy >();
        }
    };

    using SizeParam = llvm::Optional< std::uint64_t >;

    static auto unknown_size = SizeParam{ llvm::NoneType() };

} // namespace vast::hl

#define GET_TYPEDEF_CLASSES
#include "vast/Dialect/HighLevel/HighLevelTypes.h.inc"

#include <set>

namespace vast::hl
{
    using Type = mlir::Type;
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

    mlir::FunctionType getFunctionType(Type function_pointer, Module mod);

    mlir::FunctionType getFunctionType(Value callee);
    mlir::FunctionType getFunctionType(mlir::CallOpInterface call);
    mlir::FunctionType getFunctionType(mlir::CallInterfaceCallable callee, Module mod);

    Type getTypedefType(TypedefType type, Module mod);

    // unwraps all typedef aliases to get to real underlying type
    Type getBottomTypedefType(TypedefType def, Module mod);

    bool isBoolType(mlir::Type type);
    bool isIntegerType(mlir::Type type);
    bool isFloatingType(mlir::Type type);

    bool isSigned(mlir::Type type);
    bool isUnsigned(mlir::Type type);

    bool isHighLevelType(mlir::Type type);

    static inline mlir::Type to_std_float_type(mlir::Type ty) {
        using fty = mlir::FloatType;
        auto ctx = ty.getContext();
        return llvm::TypeSwitch<mlir::Type, mlir::Type>(ty)
            .Case< HalfType       >([&] (auto t) { return fty::getF16(ctx);  })
            .Case< BFloat16Type   >([&] (auto t) { return fty::getBF16(ctx); })
            .Case< FloatType      >([&] (auto t) { return fty::getF32(ctx);  })
            .Case< DoubleType     >([&] (auto t) { return fty::getF64(ctx);  })
            .Case< LongDoubleType >([&] (auto t) { return fty::getF80(ctx);  })
            .Case< Float128Type   >([&] (auto t) { return fty::getF128(ctx); })
            .Default([] (auto t) {
                VAST_UNREACHABLE("unknown float type: {0}", format_type(t));
                return mlir::Type();
            });
    }

} // namespace vast::hl
