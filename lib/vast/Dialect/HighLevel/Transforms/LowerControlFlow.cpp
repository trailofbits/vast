// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

#include "PassesDetails.hpp"

namespace vast::hl
{
    struct LowerHighLevelControlFlowPass : LowerHighLevelControlFlowBase< LowerHighLevelControlFlowPass >
    {
        void runOnOperation() override;
    };

    void LowerHighLevelControlFlowPass::runOnOperation() {}
}


std::unique_ptr< mlir::Pass > vast::hl::createLowerHighLevelControlFlowPass()
{
  return std::make_unique< LowerHighLevelControlFlowPass >();
}
