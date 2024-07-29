// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "clang/Analysis/Analyses/IntervalPartition.h"
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "llvm/ADT/PriorityQueue.h"

#include "vast/Analyses/CFG.hpp"

namespace vast::analyses {

    /// A worklist implementation where the enqueued blocks will be dequeued based
    /// on the order defined by 'Comp'.
    template< typename Comp, unsigned QueueSize, typename CFG_T, typename CFGBlock_T >
    class DataflowWorklistBase {
        llvm::BitVector EnqueuedBlocks;
        llvm::PriorityQueue< const CFGBlock_T *,
                             llvm::SmallVector< const CFGBlock_T *, QueueSize >,
                             Comp > WorkList;
    public:
        DataflowWorklistBase(const CFG_T &Cfg, Comp C) : EnqueuedBlocks(Cfg.getNumBlockIDs()), WorkList(C) {}

        void enqueueBlock(const CFGBlock_T *Block) {
            if (Block && !EnqueuedBlocks[Block->getBlockID()]) {
                EnqueuedBlocks[Block->getBlockID()] = true;
                WorkList.push(Block);
            }
        }

        const CFGBlock_T *dequeue() {
            if (WorkList.empty()) {
                return nullptr;
            }
            const CFGBlock_T *B = WorkList.top();
            WorkList.pop();
            EnqueuedBlocks[B->getBlockID()] = false;
            return B;
        }
    };

    template< typename CFGBlock_T >
    struct ReversePostOrderCompare {
        clang::PostOrderCFGView::BlockOrderCompare Cmp;
        bool operator()(const CFGBlock_T *lhs, const CFGBlock_T *rhs) const {
            return Cmp(rhs, lhs);
        }
    };

    /// A worklist implementation for forward dataflow analysis. The enqueued
    /// blocks will be dequeued in reverse post order. The worklist cannot contain
    /// the same block multiple times at once.
    template< typename CFG_T, typename CFGBlock_T, typename AnalysisDeclContext_T >
    struct ForwardDataflowWorklist
        : DataflowWorklistBase< ReversePostOrderCompare< CFGBlock_T >, 20, CFG_T, CFGBlock_T > {

        /*
        ForwardDataflowWorklist(const CFG_T &Cfg, PostOrderCFGView *POV)
        : DataflowWorklistBase(Cfg,
            ReversePostOrderCompare{POV->getComparator()}) {}
        */

        ForwardDataflowWorklist(const CFG_T &Cfg, AnalysisDeclContext_T &Ctx)
            : ForwardDataflowWorklist(Cfg, Ctx. template getAnalysis< clang::PostOrderCFGView >()) {}

        void enqueueSuccessors(const CFGBlock_T *Block) {
            for (auto B : Block->succs()) {
                enqueueBlock(B);
            }
        }
    };
} // namespace vast::analyses
