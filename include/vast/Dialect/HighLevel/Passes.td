// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_PASSES_TD
#define VAST_DIALECT_HIGHLEVEL_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerHighLevelTypes : Pass<"vast-hl-lower-types"> {
  let summary = "Lower high-level types to standard types";
  let description = [{ }];

  let constructor = "vast::hl::createLowerHighLevelTypesPass()";
}

#endif // VAST_DIALECT_HIGHLEVEL_PASSES_TD
