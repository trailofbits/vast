// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/Analysis/Clang/CFG.hpp"

#include <clang/Analysis/Analyses/IntervalPartition.h>
#include <clang/Analysis/Analyses/PostOrderCFGView.h>
#include <llvm/ADT/PriorityQueue.h>

namespace vast::analysis {

    /// A worklist implementation where the enqueued blocks will be dequeued based
    /// on the order defined by 'Comp'.
    template< typename Comp, unsigned QueueSize >
    class DataflowWorklistBase {
        llvm::BitVector EnqueuedBlocks;
        llvm::PriorityQueue< cfg::CFGBlockInterface,
                             llvm::SmallVector< cfg::CFGBlockInterface, QueueSize >,
                             Comp > WorkList;
    public:
        DataflowWorklistBase(cfg::CFGInterface Cfg, Comp C)
            : EnqueuedBlocks(Cfg.getNumBlockIDs()), WorkList(C) {}

        void enqueueBlock(cfg::CFGBlockInterface Block) {
            if (Block && !EnqueuedBlocks[Block.getBlockID()]) {
                EnqueuedBlocks[Block.getBlockID()] = true;
                WorkList.push(Block);
            }
        }

        cfg::CFGBlockInterface dequeue() {
            if (WorkList.empty()) {
                return {};
            }
            cfg::CFGBlockInterface B = WorkList.top();
            WorkList.pop();
            EnqueuedBlocks[B.getBlockID()] = false;
            return B;
        }
    };

    struct ReversePostOrderCompare {
        clang::PostOrderCFGView::BlockOrderCompare Cmp;
        bool operator()(cfg::CFGBlockInterface lhs, cfg::CFGBlockInterface rhs) const {
            VAST_UNIMPLEMENTED;
            // return Cmp(rhs, lhs);
        }
    };

    /// A worklist implementation for forward dataflow analysis. The enqueued
    /// blocks will be dequeued in reverse post order. The worklist cannot contain
    /// the same block multiple times at once.
    struct ForwardDataflowWorklist
        : DataflowWorklistBase< ReversePostOrderCompare, 20 > {

        ForwardDataflowWorklist(cfg::CFGInterface Cfg, clang::PostOrderCFGView *POV)
        : DataflowWorklistBase(Cfg, ReversePostOrderCompare{POV->getComparator()}) {}

        ForwardDataflowWorklist(cfg::CFGInterface Cfg, AnalysisDeclContextInterface Ctx)
            : ForwardDataflowWorklist(Cfg, Ctx. template getAnalysis< clang::PostOrderCFGView >()) {}

        void enqueueSuccessors(cfg::CFGBlockInterface Block) {
            for (auto B : Block.succs()) {
                enqueueBlock(mlir::dyn_cast< cfg::CFGBlockInterface >(*B));
            }
        }
    };
} // namespace vast::analysis
