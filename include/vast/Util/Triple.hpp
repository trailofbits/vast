/*
 * Copyright (c) 2023 Trail of Bits, Inc.
 */

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/Debug.h>

#include <mlir/IR/Builders.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/Core/CoreDialect.hpp"

#include <string>

namespace vast
{
    void set_triple(auto op, std::string triple)
    {
        mlir::OpBuilder bld(op);
        auto attr = bld.getAttr< mlir::StringAttr >(triple);
        op->setAttr(core::CoreDialect::getTargetTripleAttrName(), attr);
    }
} // namespace vast
