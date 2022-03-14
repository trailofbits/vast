// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/DialectImplementation.h>
VAST_RELAX_WARNINGS

namespace vast::hl
{
    using Context = mlir::MLIRContext;

} // namespace vast::hl

#define GET_ATTRDEF_CLASSES
#include "vast/Dialect/HighLevel/HighLevelAttributes.cpp.inc"
namespace vast::hl
{
    using DialectParser = mlir::DialectAsmParser;
    using DialectPrinter = mlir::DialectAsmPrinter;

    void HighLevelDialect::registerAttributes()
    {
        addAttributes<
            #define GET_ATTRDEF_LIST
            #include "vast/Dialect/HighLevel/HighLevelAttributes.cpp.inc"
        >();
    }

    // FIXME: commented out, because high-level dialect does not have any attributes yet
    // Attribute HighLevelDialect::parseAttribute(DialectParser &parser, Type type) const
    // {
    //     auto loc = parser.getCurrentLocation();

    //     llvm::StringRef mnemonic;
    //     if (parser.parseKeyword(&mnemonic))
    //         return Attribute();

    //     Attribute result;
    //     if (generatedAttributeParser(getContext(), parser, mnemonic, type, result).hasValue()) {
    //         return result;
    //     }

    //     parser.emitError(loc, "unexpected high-level attribute '" + mnemonic + "'");
    //     return Attribute();
    // }

    // void HighLevelDialect::printAttribute(Attribute attr, DialectPrinter &p) const
    // {
    //     if (failed(generatedAttributePrinter(attr, p)))
    //         VAST_UNREACHABLE("Unexpected attribute");
    // }

} // namespace vast::hl
