// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Dialect/PDL/IR/PDL.h>
#include <mlir/Dialect/PDLInterp/IR/PDLInterp.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Parser/Parser.h>
VAST_UNRELAX_WARNINGS

#include "vast/Conversion/HLToFunc.h.inc"
