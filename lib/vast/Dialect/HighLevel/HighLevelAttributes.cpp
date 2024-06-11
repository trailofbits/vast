// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
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
    mlir_attr OffsetOfNodeAttr::parse(mlir::AsmParser &parser, mlir_type) {
        if (mlir::failed(parser.parseLess())) {
            return {};
        }
        if (mlir::succeeded(parser.parseOptionalKeyword("index"))) {
            if (mlir::failed(parser.parseColon())) {
                return {};
            }

            unsigned int value;
            if (mlir::failed(parser.parseInteger(value)) || mlir::failed(parser.parseGreater()))
            {
                return {};
            }
            return get(parser.getContext(), value);
        }

        if (mlir::succeeded(parser.parseOptionalKeyword("identifier"))) {
            if (mlir::failed(parser.parseColon())) {
                return {};
            }

            std::string value;
            if (mlir::failed(parser.parseString(&value)) || mlir::failed(parser.parseGreater()))
            {
                return {};
            }
            return get(parser.getContext(), value);
        }

        return {};
    }

    void OffsetOfNodeAttr::print(mlir::AsmPrinter &printer) const {
        auto value = getValue();
        printer << "<";
        auto print_identifier = [&](const mlir::StringAttr &str) {
            printer << "identifier : " << str;
        };
        auto print_index = [&](unsigned int index) { printer << "index : " << index; };
        std::visit(gap::overloaded{ print_index, print_identifier }, value);
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
