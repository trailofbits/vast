// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/SMLoc.h>


namespace vast::hl
{
    void HighLevelDialect::initialize()
    {
        registerTypes();
        registerAttributes();

        addOperations<
            #define GET_OP_LIST
            #include "vast/Dialect/HighLevel/HighLevel.cpp.inc"
        >();
    }

    using string_ref = llvm::StringRef;
    using DialectParser = mlir::DialectAsmParser;
    using DialectPrinter = mlir::DialectAsmPrinter;

    // Parse a type registered to this dialect.
    Type HighLevelDialect::parseType(DialectParser &parser) const
    {
        return Type(); //detail::parse_type(parser);
    }

    void HighLevelDialect::printType(Type type, DialectPrinter &os) const
    {
        os << to_string(type.cast<HighLevelType>());
    }

} // namespace vast::hl

#include "vast/Dialect/HighLevel/HighLevelDialect.cpp.inc"

// Provide implementations for the enums we use.
#include "vast/Dialect/HighLevel/HighLevelEnums.cpp.inc"
