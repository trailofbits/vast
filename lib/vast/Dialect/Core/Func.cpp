// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Support/LogicalResult.h>

#include <mlir/IR/OperationSupport.h>

#include <mlir/Interfaces/FunctionImplementation.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/Core/Func.hpp"

#include "vast/Dialect/Core/CoreAttributes.hpp"
#include "vast/Dialect/Core/CoreDialect.hpp"
#include "vast/Dialect/Core/Linkage.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/Dialect.hpp"
#include "vast/Util/Region.hpp"

namespace vast::core
{
    //===----------------------------------------------------------------------===//
    // FuncOp
    //===----------------------------------------------------------------------===//

    llvm::StringRef getLinkageAttrNameString() { return "linkage"; }

    ParseResult parseFunctionSignatureAndBody(
        Parser &parser, Attribute &funcion_type, mlir::NamedAttrList &attr_dict, Region &body
    ) {
        using core::GlobalLinkageKind;

        llvm::SmallVector< Parser::Argument, 8 > arguments;
        llvm::SmallVector< mlir::DictionaryAttr, 1 > result_attrs;
        llvm::SmallVector< Type, 8 > arg_types;
        llvm::SmallVector< Type, 4 > result_types;

        // Default to external linkage if no keyword is provided.
        if (!attr_dict.getNamed(getLinkageAttrNameString())) {
            attr_dict.append(
                getLinkageAttrNameString(),
                core::GlobalLinkageKindAttr::get(
                    parser.getContext(),
                    parse_optional_vast_keyword< GlobalLinkageKind >(
                        parser, GlobalLinkageKind::ExternalLinkage
                    )
                )
            );
        }

        bool is_variadic = false;
        if (mlir::failed(mlir::function_interface_impl::parseFunctionSignature(
            parser, /*allowVariadic=*/true, arguments, is_variadic, result_types, result_attrs
        ))) {
            return mlir::failure();
        }

        for (auto &arg : arguments) {
            arg_types.push_back(arg.type);
        }

        // create parsed function type
        funcion_type = mlir::TypeAttr::get(
            core::FunctionType::get(
                arg_types, result_types, is_variadic
            )
        );

        // If additional attributes are present, parse them.
        if (parser.parseOptionalAttrDictWithKeyword(attr_dict)) {
            return mlir::failure();
        }

        // TODO: Add the attributes to the function arguments.
        // VAST_ASSERT(result_attrs.size() == result_types.size());
        // return mlir::function_interface_impl::addArgAndResultAttrs(
        //     builder, state, arguments, result_attrs
        // );

        auto loc = parser.getCurrentLocation();
        auto parse_result = parser.parseOptionalRegion(
            body, arguments, /* enableNameShadowing */false
        );

        if (parse_result.has_value()) {
            if (failed(*parse_result))
                return mlir::failure();
            // Function body was parsed, make sure its not empty.
            if (body.empty())
                return parser.emitError(loc, "expected non-empty function body");
        }

        return mlir::success();
    }

} // namespace vast::core
