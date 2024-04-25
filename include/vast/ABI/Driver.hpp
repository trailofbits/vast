// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

#include "vast/ABI/ABI.hpp"
#include "vast/ABI/Classify.hpp"

namespace vast::abi {
    template< typename FnOp >
    auto make_x86_64(FnOp fn, const mlir::DataLayout &dl) {
        using out        = func_info< FnOp >;
        using classifier = classifier_base< out, mlir::DataLayout >;
        return make< FnOp, classifier >(fn, dl);
    }
} // namespace vast::abi
