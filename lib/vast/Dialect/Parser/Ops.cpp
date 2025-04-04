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

    using fold_result = ::mlir::OpFoldResult;
    using fold_results = ::llvm::SmallVectorImpl< fold_result >;

    template< typename op_t >
    logical_result forward_same_operation(
        op_t op, auto adaptor, fold_results &results
    ) {
        if (op.getNumOperands() == 1 && op.getNumResults() == 1) {
            if (auto operand = op.getOperand(0); mlir::isa< op_t >(operand.getDefiningOp())) {
                if (operand.getType() == op->getOpResult(0).getType()) {
                    results.push_back(operand);
                    return mlir::success();
                }
            }
        }

        if (op.getNumOperands() > 0) {
            auto ty               = op.getOperand(0).getType();
            auto is_same_type     = [ty](auto val) { return val.getType() == ty; };
            auto all_of_same_type = [is_same_type](const auto &rng) {
                return llvm::all_of(rng, is_same_type);
            };

            if (all_of_same_type(op.getOperands()) && all_of_same_type(op.getResults())) {
                results.push_back(op.getOperand(0));
                return mlir::success();
            }
        }

        return mlir::failure();
    }

    logical_result Source::fold(FoldAdaptor adaptor, fold_results &results) {
        return forward_same_operation(*this, adaptor, results);
    }

    logical_result Sink::fold(FoldAdaptor adaptor, fold_results &results) {
        return forward_same_operation(*this, adaptor, results);
    }

    logical_result Parse::fold(FoldAdaptor adaptor, fold_results &results) {
        return forward_same_operation(*this, adaptor, results);
    }

    logical_result NoParse::fold(FoldAdaptor adaptor, fold_results &results) {
        return forward_same_operation(*this, adaptor, results);
    }

    logical_result MaybeParse::fold(FoldAdaptor adaptor, fold_results &results) {
        return forward_same_operation(*this, adaptor, results);
    }

    fold_result Cast::fold(FoldAdaptor adaptor) {
        if (auto operand = getOperand(); mlir::isa< Cast >(operand.getDefiningOp())) {
            if (operand.getType() == getType()) {
                return operand;
            }
        }

        return {};
    }

} // namespace vast::pr
