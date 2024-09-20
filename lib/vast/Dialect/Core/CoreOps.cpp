// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Dialect/Core/CoreDialect.hpp"
#include "vast/Dialect/Core/CoreOps.hpp"
#include "vast/Dialect/Core/CoreTypes.hpp"
#include "vast/Dialect/Core/CoreAttributes.hpp"

#include "vast/Util/Common.hpp"

#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/OpImplementation.h>

#include <mlir/Interfaces/FunctionImplementation.h>

#include <llvm/Support/ErrorHandling.h>

#include <optional>
#include <variant>

namespace vast::core
{
    //
    // ModuleOp
    //

    void ModuleOp::build(mlir_builder &builder, op_state &state, std::optional< string_ref > name) {
        state.addRegion()->emplaceBlock();

        if (name.has_value()) {
            state.addAttribute(symbol_attr_name(), builder.getStringAttr(name.value()));
        }
    }

    ModuleOp ModuleOp::create(loc_t loc, std::optional< string_ref > name) {
        mlir_builder builder(loc->getContext());
        return builder.create< ModuleOp >(loc, name);
    }

    mlir::DataLayoutSpecInterface ModuleOp::getDataLayoutSpec() {
        // Take the first and only (if present) attribute that implements the
        // interface. This needs a linear search, but is called only once per data
        // layout object construction that is used for repeated queries.
        for (auto attr : getOperation()->getAttrs()) {
            if (auto spec = llvm::dyn_cast< mlir::DataLayoutSpecInterface >(attr.getValue())) {
                return spec;
            }
        }
        return {};
    }

    logical_result ModuleOp::verify() {
        // Check that there is at most one data layout spec attribute.
        string_ref layout_spec_attr_name;
        mlir::DataLayoutSpecInterface layout_spec;
        for (const auto &na : (*this)->getAttrs()) {
            if (auto spec = llvm::dyn_cast< mlir::DataLayoutSpecInterface >(na.getValue())) {
                if (layout_spec) {
                    mlir::InFlightDiagnostic diag = emitOpError()
                        << "expects at most one data layout attribute";
                    diag.attachNote()
                        << "'" << layout_spec_attr_name << "' is a data layout attribute";
                    diag.attachNote()
                        << "'" << na.getName().getValue() << "' is a data layout attribute";
                }
                layout_spec_attr_name = na.getName().strref();
                layout_spec         = spec;
            }
        }

        return mlir::success();
    }

} // namespace vast::core

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "vast/Dialect/Core/Core.cpp.inc"
