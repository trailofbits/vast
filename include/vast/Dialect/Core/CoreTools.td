// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_CORE_DIALECT_TOOLS
#define VAST_CORE_DIALECT_TOOLS

include "mlir/IR/RegionKindInterface.td"

def HasOneBlock   : CPred<"$_self.hasOneBlock()">;

def ValueRegion : Region< HasOneBlock, "value region" >;

#endif // VAST_CORE_DIALECT_TOOLS
