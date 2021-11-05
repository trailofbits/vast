// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

#define  GET_TYPEDEF_CLASSES
#include "vast/Dialect/HighLevel/HighLevelTypes.cpp.inc"

#include <llvm/ADT/TypeSwitch.h>

namespace vast::hl
{
    namespace detail
    {
        using Storage = mlir::TypeStorage;
    } // namespace detail

    bool HighLevelType::isGround()
    {
        return llvm::TypeSwitch< HighLevelType, bool >( *this )
            .Case< VoidType, BoolType, IntegerType, FloatingType >( [] (Type) { return true; } )
            .Default( [] (Type) { llvm_unreachable("unknown high-level type"); return false; } );
    }

    VoidType VoidType::get(Context *ctx) { return Base::get(ctx); }

    BoolType BoolType::get(Context *ctx) { return Base::get(ctx); }

    IntegerType IntegerType::get(Context *ctx) { return Base::get(ctx); }

    FloatingType FloatingType::get(Context *ctx) { return Base::get(ctx); }

} // namespace vast::hl
