// Copyright (c) 2023-present, Trail of Bits, Inc.
#include "vast/Dialect/HighLevel/Passes.hpp"

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/PatternMatch.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"
#include "vast/Util/DialectConversion.hpp"

#include "vast/Conversion/Common/Types.hpp"
#include "vast/Conversion/Common/Rewriter.hpp"

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/Dialect/Core/CoreOps.hpp"

#include "PassesDetails.hpp"

namespace vast::hl
{

    struct HLCanonize : HLCanonizeBase< HLCanonize > {
        using base = HLCanonizeBase< HLCanonize >;
        using rewriter_t = conv::rewriter_wrapper_t< mlir::IRRewriter >;

        std::vector< operation > to_remove;
        void insert_void_return(hl::FuncOp &op, rewriter_t &rewriter ) {
            auto g = rewriter.guard();
            rewriter->setInsertionPointToEnd(&op.getBody().back());
            auto void_type = rewriter->getType< hl::VoidType >();
            auto void_const = rewriter->create< hl::ConstantOp >(op.getLoc(), void_type);
            rewriter->create< core::ImplicitReturnOp >(op.getLoc(), void_const.getResult());
        }

        void run(Operation *op, rewriter_t &rewriter) {
            if (mlir::isa< hl::SkipStmt >(op)) {
                to_remove.emplace_back(op);
                return;
            }

            if (auto fn = mlir::dyn_cast< hl::FuncOp >(op)) {
                if(!fn.isDeclaration()) {
                    auto &last_block = fn.getBody().back();
                    if (last_block.empty()
                        || !is_one_of_mlir< hl::ReturnOp, core::ImplicitReturnOp >(
                            &last_block.back()
                        ))
                    {
                        insert_void_return(fn, rewriter);
                    }
                }
            }
            for (auto &region : op->getRegions()) {
                run(&region, rewriter);
            }
        }

        void run(Region *region, rewriter_t &rewriter) {
            for (auto &block : region->getBlocks())
                run(&block, rewriter);
        }

        void run(Block *block, rewriter_t &rewriter) {
            for (auto &op : block->getOperations())
                run(&op, rewriter);
        }

        void runOnOperation() override
        {
            auto op = getOperation();
            auto rewriter = mlir::IRRewriter(&getContext());
            auto bld = rewriter_t(rewriter);

            run(op, bld);

            for (auto op : to_remove)
                op->erase();
        }
    };

    std::unique_ptr< mlir::Pass > createHLCanonizePass()
    {
        return std::make_unique< vast::hl::HLCanonize >();
    }
} // namespace vast::hl

