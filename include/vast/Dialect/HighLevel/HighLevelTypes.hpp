// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Dialect/DLTI/DLTI.h>

#include <clang/AST/Type.h>

#include <llvm/ADT/Hashing.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/TypeList.hpp"
#include "vast/Util/DataLayout.hpp"

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"

namespace vast::hl
{
    template< typename ConcreteTy >
    struct DefaultDL {
        using dl_t = mlir::DataLayout;
        using dl_entries_ref = mlir::DataLayoutEntryListRef;

        static unsigned getTypeSizeInBits(const dl_t &dl, dl_entries_ref entries)
        {
            CHECK(entries.size() != 0,
                  "Query for getTypeSizeinBits for {0} failed: Must have at least one entry!");

            std::optional<uint32_t> out;
            auto handle_entry = [&](auto &dl_entry) {
                if (!out) out = dl_entry.bw;
                CHECK(*out == dl_entry.bw, "Inconsistent entries");
            };
            apply_on_valid_entries(entries, handle_entry);
            CHECK(out.has_value(), "getTypeSizeBits entries did not yield result.");
            return *out;
        }

        static unsigned getABIAlignment(const dl_t &dl, dl_entries_ref entries)
        {
            UNIMPLEMENTED;
        }

        static unsigned getPreferredAlignment(const dl_t &dl, dl_entries_ref entries)
        {
            UNIMPLEMENTED;
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
} // namespace vast::hl

#define GET_TYPEDEF_CLASSES
#include "vast/Dialect/HighLevel/HighLevelTypes.h.inc"

#include <set>

namespace vast::hl
{
    using Type = mlir::Type;
    using Context = mlir::MLIRContext;

    using DialectParser = mlir::DialectAsmParser;
    using DialectPrinter = mlir::DialectAsmPrinter;

    using integer_types = util::type_list<
        CharType, ShortType, IntType, LongType, LongLongType, Int128Type
    >;

    using floating_types = util::type_list<
        FloatType, DoubleType, LongDoubleType
    >;

    using scalar_types = util::concat<
        util::type_list< BoolType >, integer_types, floating_types
    >;

    using composite_types = util::type_list<
        RecordType, ConstantArrayType, PointerType
    >;

    using high_level_types = util::concat<
        scalar_types, composite_types, util::type_list< VoidType >
    >;

    /* integer types */
    enum class IntegerKind { Char, Short, Int, Long, LongLong, Int128 };

    inline std::string to_string(IntegerKind kind)
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

    mlir::FunctionType getFunctionType(PointerType functionPointer);
    mlir::FunctionType getFunctionType(mlir::Type functionPointer);

    bool isBoolType(mlir::Type type);
    bool isIntegerType(mlir::Type type);
    bool isFloatingType(mlir::Type type);

    bool isSigned(mlir::Type type);
    bool isUnsigned(mlir::Type type);

    bool isHighLevelType(mlir::Type type);

} // namespace vast::hl
