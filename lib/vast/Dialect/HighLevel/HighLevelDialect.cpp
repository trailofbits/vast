// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/Util/Functions.hpp"
#include "vast/Util/Common.hpp"

#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/DialectInterface.h>

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/SMLoc.h>
#include <optional>
#include <type_traits>
#include <vector>


namespace vast::hl
{
    using mlir::OpAsmDialectInterface;

    struct HighLevelOpAsmDialectInterface : OpAsmDialectInterface
    {
        using OpAsmDialectInterface::OpAsmDialectInterface;

        AliasResult getAlias(mlir_type type, llvm::raw_ostream &os) const final {
            if (auto ty = type.dyn_cast< hl::VoidType >()) {
                os << ty.getAlias();
                return AliasResult::OverridableAlias;
            }

            if (auto ty = type.dyn_cast< hl::BoolType >()) {
                os << ty.getAlias();
                return AliasResult::OverridableAlias;
            }

            return AliasResult::NoAlias;
        }
    };

    void HighLevelDialect::initialize()
    {
        registerTypes();
        registerAttributes();

        addOperations<
            #define GET_OP_LIST
            #include "vast/Dialect/HighLevel/HighLevel.cpp.inc"
        >();

        addInterfaces< HighLevelOpAsmDialectInterface >();
    }

    using DialectParser = mlir::AsmParser;
    using DialectPrinter = mlir::AsmPrinter;

    Operation *HighLevelDialect::materializeConstant(Builder &builder, Attribute value, Type type, Location loc)
    {
        VAST_UNIMPLEMENTED;
    }

} // namespace vast::hl

#include "vast/Dialect/HighLevel/HighLevelDialect.cpp.inc"

// Provide implementations for the enums we use.
#include "vast/Dialect/HighLevel/HighLevelEnums.cpp.inc"
