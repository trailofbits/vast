// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Util/FieldAdditions.hpp"
#include "vast/Util/Warnings.hpp"
#include <gap/core/overloads.hpp>

VAST_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/DialectImplementation.h>
VAST_RELAX_WARNINGS

#define GET_ATTRDEF_CLASSES
#include "vast/Dialect/HighLevel/HighLevelAttributes.cpp.inc"

namespace vast::hl
{
    mlir_attr OffsetOfNodeAttr::parseAttrWithInteger(mlir::AsmParser &parser) {
        if (mlir::failed(parser.parseColon())) {
            return {};
        }

        unsigned int value;
        if (mlir::failed(parser.parseInteger(value)) || mlir::failed(parser.parseGreater())) {
            return {};
        }
        return get(parser.getContext(), value);
    }

    mlir_attr OffsetOfNodeAttr::parseAttrWithString(mlir::AsmParser &parser) {
        if (mlir::failed(parser.parseColon())) {
            return {};
        }

        std::string value;
        if (mlir::failed(parser.parseString(&value)) || mlir::failed(parser.parseGreater())) {
            return {};
        }
        return get(parser.getContext(), mlir::StringAttr::get(parser.getContext(), value));
    }

    mlir_attr OffsetOfNodeAttr::parse(mlir::AsmParser &parser, mlir_type) {
        if (mlir::failed(parser.parseLess())) {
            return {};
        }
        if (mlir::succeeded(parser.parseOptionalKeyword("index"))) {
            return parseAttrWithInteger(parser);
        }

        if (mlir::succeeded(parser.parseOptionalKeyword("identifier"))) {
            return parseAttrWithString(parser);
        }

        return {};
    }

    void OffsetOfNodeAttr::print(mlir::AsmPrinter &printer) const {
        auto value = getValue();
        printer << "<";
        std::visit(
            gap::overloaded{
                [&](unsigned int index) { printer << "index : " << index; },
                [&](const mlir::StringAttr &str) { printer << "identifier : " << str; } },
            value
        );
        printer << ">";
    }

    void HighLevelDialect::registerAttributes()
    {
        addAttributes<
            #define GET_ATTRDEF_LIST
            #include "vast/Dialect/HighLevel/HighLevelAttributes.cpp.inc"
        >();
    }

} // namespace vast::hl
