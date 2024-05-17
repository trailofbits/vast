// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Dialect/Core/CoreDialect.hpp"
#include "vast/Dialect/Core/CoreOps.hpp"
#include "vast/Dialect/Core/CoreTypes.hpp"
#include "vast/Dialect/Core/CoreAttributes.hpp"

#include "vast/Interfaces/AliasTypeInterface.hpp"

#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/SMLoc.h>
#include <optional>
#include <type_traits>
#include <vector>

namespace vast::core
{
    using mlir::OpAsmDialectInterface;

    struct CoreOpAsmDialectInterface : OpAsmDialectInterface
    {
        using OpAsmDialectInterface::OpAsmDialectInterface;

        AliasResult getAlias(mlir_type type, llvm::raw_ostream &os) const final {
            if (mlir::isa< CoreDialect >(type.getDialect())) {
                if (auto ty = type.dyn_cast< AliasTypeInterface >()) {
                    os << ty.getAlias();
                    return ty.getAliasResultKind();
                }
            }

            return AliasResult::NoAlias;
        }

        AliasResult getAlias(mlir_attr attr, llvm::raw_ostream &os) const final {
            if (auto at = attr.dyn_cast< core::VoidAttr >()) {
                os << "void_value";
                return AliasResult::FinalAlias;
            }

            if (auto at = attr.dyn_cast< core::BooleanAttr >()) {
                os << (at.getValue() ? "true" : "false");
                return AliasResult::FinalAlias;
            }

            return AliasResult::NoAlias;
        }
    };

    void CoreDialect::initialize()
    {
        registerTypes();
        registerAttributes();

        addOperations<
            #define GET_OP_LIST
            #include "vast/Dialect/Core/Core.cpp.inc"
        >();

        addInterfaces< CoreOpAsmDialectInterface >();
    }

    using OpBuilder = mlir::OpBuilder;

    Operation *CoreDialect::materializeConstant(OpBuilder &builder, Attribute value, Type type, Location loc)
    {
        VAST_UNIMPLEMENTED;
    }
} // namespace vast::core

#include "vast/Dialect/Core/CoreDialect.cpp.inc"

// Provide implementations for enum classes.
#include "vast/Dialect/Core/CoreEnums.cpp.inc"
