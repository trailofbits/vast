// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

#include "PassesDetails.hpp"

namespace vast::hl
{
    struct LowerHighLevelTypesPass : LowerHighLevelTypesBase< LowerHighLevelTypesPass >
    {
        void runOnOperation() override;
    };

    void LowerHighLevelTypesPass::runOnOperation() {}
}


std::unique_ptr< mlir::Pass > vast::hl::createLowerHighLevelTypesPass()
{
  return std::make_unique< LowerHighLevelTypesPass >();
}
