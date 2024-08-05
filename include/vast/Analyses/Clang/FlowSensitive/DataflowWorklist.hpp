// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/Analyses/Clang/CFG.hpp"

#include <clang/Analysis/Analyses/IntervalPartition.h>
#include <clang/Analysis/Analyses/PostOrderCFGView.h>
#include <llvm/ADT/PriorityQueue.h>

namespace vast::analyses {

    /// A worklist implementation where the enqueued blocks will be dequeued based
    /// on the order defined by 'Comp'.
    template< typename Comp, unsigned QueueSize, typename CFG, typename CFGBlock >
    class DataflowWorklistBase {
        llvm::BitVector EnqueuedBlocks;
        llvm::PriorityQueue< const CFGBlock *,
                             llvm::SmallVector< const CFGBlock *, QueueSize >,
                             Comp > WorkList;
    public:
        DataflowWorklistBase(const CFG &Cfg, Comp C) : EnqueuedBlocks(Cfg.getNumBlockIDs()), WorkList(C) {}

        void enqueueBlock(const CFGBlock *Block) {
            if (Block && !EnqueuedBlocks[Block->getBlockID()]) {
                EnqueuedBlocks[Block->getBlockID()] = true;
                WorkList.push(Block);
            }
        }

        const CFGBlock *dequeue() {
            if (WorkList.empty()) {
                return nullptr;
            }
            const CFGBlock *B = WorkList.top();
            WorkList.pop();
            EnqueuedBlocks[B->getBlockID()] = false;
            return B;
        }
    };

    template< typename CFGBlock >
    struct ReversePostOrderCompare {
        clang::PostOrderCFGView::BlockOrderCompare Cmp;
        bool operator()(const CFGBlock *lhs, const CFGBlock *rhs) const {
            return Cmp(rhs, lhs);
        }
    };

    /// A worklist implementation for forward dataflow analysis. The enqueued
    /// blocks will be dequeued in reverse post order. The worklist cannot contain
    /// the same block multiple times at once.
    template< typename CFG, typename CFGBlock, typename AnalysisDeclContext >
    struct ForwardDataflowWorklist
        : DataflowWorklistBase< ReversePostOrderCompare< CFGBlock >, 20, CFG, CFGBlock > {

        /*
        ForwardDataflowWorklist(const CFG &Cfg, PostOrderCFGView *POV)
        : DataflowWorklistBase(Cfg,
            ReversePostOrderCompare{POV->getComparator()}) {}
        */

        ForwardDataflowWorklist(const CFG &Cfg, AnalysisDeclContext &Ctx)
            : ForwardDataflowWorklist(Cfg, Ctx. template getAnalysis< clang::PostOrderCFGView >()) {}

        void enqueueSuccessors(const CFGBlock *Block) {
            for (auto B : Block->succs()) {
                enqueueBlock(B);
            }
        }
    };
} // namespace vast::analyses
