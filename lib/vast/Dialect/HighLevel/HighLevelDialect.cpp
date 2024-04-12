// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/Interfaces/AliasTypeInterface.hpp"

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
            if (mlir::isa< HighLevelDialect >(type.getDialect())) {
                if (auto ty = type.dyn_cast< AliasTypeInterface >()) {
                    os << ty.getAlias();
                    return ty.getAliasResultKind();
                }
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

    operation HighLevelDialect::materializeConstant(mlir_builder &builder, mlir_attr value, mlir_type type, loc_t loc)
    {
        if (ConstantOp::isBuildableWith(value, type)) {
            auto typed = mlir::cast< mlir::TypedAttr >(value);
            return builder.create< ConstantOp >(loc, type, typed);
        }

        return {};
    }

} // namespace vast::hl

#include "vast/Dialect/HighLevel/HighLevelDialect.cpp.inc"

// Provide implementations for the enums we use.
#include "vast/Dialect/HighLevel/HighLevelEnums.cpp.inc"
