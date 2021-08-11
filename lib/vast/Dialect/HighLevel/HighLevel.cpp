// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/HighLevel.hpp"
#include "mlir/IR/TypeSupport.h"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SMLoc.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <llvm/ADT/TypeSwitch.h>

namespace vast::hl
{
    void HighLevelDialect::initialize()
    {
        addOperations<
            #define GET_OP_LIST
            #include "vast/Dialect/HighLevel/HighLevel.cpp.inc"
        >();
        addTypes<
            #define GET_TYPEDEF_LIST
            #include "vast/Dialect/HighLevel/HighLevelTypes.cpp.inc"
        >();
    }

    using string_ref = llvm::StringRef;
    using dialect_parser = mlir::DialectAsmParser;
    using dialect_printer = mlir::DialectAsmPrinter;

    namespace detail
    {
        type failure(dialect_parser &parser)
        {
            parser.emitError(parser.getNameLoc(), "Failed to parse HighLevel type");
            return {};
        }

        type dispatch(dialect_parser &parser)
        {
            string_ref key;
            if (failed(parser.parseKeyword(&key)))
                return {};

            auto ctx = parser.getBuilder().getContext();

            using type_parser = llvm::function_ref< type() >;

            return llvm::StringSwitch< type_parser >(key)
                .Case("void", [&] { return VoidType::get(ctx); })
                .Default([&] { return failure(parser); })();
        }

        type parse_type(dialect_parser &parser)
        {
            return dispatch(parser);
        }

        string_ref get_type_keyword(type ty)
        {
            return llvm::TypeSwitch< type, string_ref >(ty)
                .Case< VoidType >([&] (type) { return "void"; })
                .Default([] (type) -> string_ref {
                    llvm_unreachable("unexpected 'highlevel' type kind");
                });
        }

        void print_type(type ty, dialect_printer &os)
        {
            os.printType(ty);
        }
    } // namespace detail

    // Parse a type registered to this dialect.
    type HighLevelDialect::parseType(dialect_parser &parser) const
    {
        return detail::parse_type(parser);
    }

    // Print a type registered to this dialect.
    void HighLevelDialect::printType(type ty, dialect_printer &os) const
    {
        return detail::print_type(ty, os);
    }



} // namespace vast::hl