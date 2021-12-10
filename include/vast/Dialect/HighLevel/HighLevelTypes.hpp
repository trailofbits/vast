// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>

#include <clang/AST/Type.h>

#include <llvm/ADT/Hashing.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Parser.hpp"
#include "vast/Util/TypeList.hpp"

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"

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

    /* floating point types */
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

} // namespace vast::hl
