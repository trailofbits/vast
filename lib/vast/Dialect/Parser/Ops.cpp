// Copyright (c) 2024, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

#include "vast/Dialect/Parser/Dialect.hpp"
#include "vast/Dialect/Parser/Ops.hpp"
#include "vast/Dialect/Parser/Types.hpp"

#include "vast/Util/Common.hpp"

using namespace vast::pr;

#define GET_OP_CLASSES
#include "vast/Dialect/Parser/Parser.cpp.inc"

namespace vast::pr {

    using fold_result_t = ::llvm::SmallVectorImpl< ::mlir::OpFoldResult >;

    logical_result NoParse::fold(FoldAdaptor adaptor, fold_result_t &results) {
        auto op  = getOperation();
        auto res = mlir::failure();
        for (size_t i = 0; i < getNumOperands(); ++i) {
            if (auto noparse = mlir::dyn_cast< NoParse >(getOperand(i).getDefiningOp())) {
                op->eraseOperand(i);
                i   = 0;
                res = mlir::success();
            }
        }
        return res;
    }

} // namespace vast::pr
