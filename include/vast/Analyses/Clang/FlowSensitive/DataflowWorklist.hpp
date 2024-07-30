// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/Analyses/Clang/CFG.hpp"

#include <clang/Analysis/Analyses/IntervalPartition.h>
#include <clang/Analysis/Analyses/PostOrderCFGView.h>
#include <llvm/ADT/PriorityQueue.h>

namespace vast::analyses {

    /// A worklist implementation where the enqueued blocks will be dequeued based
    /// on the order defined by 'Comp'.
    template< typename Comp, unsigned QueueSize, typename CFGT, typename CFGBlockT >
    class DataflowWorklistBase {
        llvm::BitVector EnqueuedBlocks;
        llvm::PriorityQueue< const CFGBlockT *,
                             llvm::SmallVector< const CFGBlockT *, QueueSize >,
                             Comp > WorkList;
    public:
        DataflowWorklistBase(const CFGT &Cfg, Comp C) : EnqueuedBlocks(Cfg.getNumBlockIDs()), WorkList(C) {}

        void enqueueBlock(const CFGBlockT *Block) {
            if (Block && !EnqueuedBlocks[Block->getBlockID()]) {
                EnqueuedBlocks[Block->getBlockID()] = true;
                WorkList.push(Block);
            }
        }

        const CFGBlockT *dequeue() {
            if (WorkList.empty()) {
                return nullptr;
            }
            const CFGBlockT *B = WorkList.top();
            WorkList.pop();
            EnqueuedBlocks[B->getBlockID()] = false;
            return B;
        }
    };

    template< typename CFGBlockT >
    struct ReversePostOrderCompare {
        clang::PostOrderCFGView::BlockOrderCompare Cmp;
        bool operator()(const CFGBlockT *lhs, const CFGBlockT *rhs) const {
            return Cmp(rhs, lhs);
        }
    };

    /// A worklist implementation for forward dataflow analysis. The enqueued
    /// blocks will be dequeued in reverse post order. The worklist cannot contain
    /// the same block multiple times at once.
    template< typename CFGT, typename CFGBlockT, typename AnalysisDeclContextT >
    struct ForwardDataflowWorklist
        : DataflowWorklistBase< ReversePostOrderCompare< CFGBlockT >, 20, CFGT, CFGBlockT > {

        /*
        ForwardDataflowWorklist(const CFGT &Cfg, PostOrderCFGView *POV)
        : DataflowWorklistBase(Cfg,
            ReversePostOrderCompare{POV->getComparator()}) {}
        */

        ForwardDataflowWorklist(const CFGT &Cfg, AnalysisDeclContextT &Ctx)
            : ForwardDataflowWorklist(Cfg, Ctx. template getAnalysis< clang::PostOrderCFGView >()) {}

        void enqueueSuccessors(const CFGBlockT *Block) {
            for (auto B : Block->succs()) {
                enqueueBlock(B);
            }
        }
    };
} // namespace vast::analyses
