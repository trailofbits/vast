// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include <llvm/ADT/TypeSwitch.h>

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

namespace vast::hl
{

    template< typename Derived, typename Result = void, typename ...Args >
    struct RecursiveHighLevelOperationsVisitor
    {
        Derived& self() { return *static_cast< Derived* >(this); }
        const Derived& self() const { return *static_cast< const Derived* >(this); }

        Result visitIgnore(mlir::Operation *op)
        {
            return Result();
        }

        Result visitUnhandled(mlir::Operation *op)
        {
            if (report_unhandled_cases) {
                llvm::errs() << "unhandled: " << op->getName() << "\n";
            }
            return Result();
        }

        Result visit(mlir::Operation *op, Args &&... args)
        {
            return visitUnhandled(op);
        }

        Result visit(mlir::FuncOp fn, Args &&... args)
        {
            for (auto &region : fn) {
                visit(region);
            }
            return Result();
        }

        Result visit(mlir::ModuleOp mod, Args &&... args)
        {
            return visit(mod.getBodyRegion(), std::forward< Args >(args)... );
        }

        Result visit(mlir::Region &region, Args &&... args)
        {
            for (auto &bb : region) {
                visit(bb, std::forward< Args >(args)... );
            }
            return Result();
        }

        Result visit(mlir::Block &block, Args &&... args)
        {
            for (auto &op : block) {
                dispatch(&op, std::forward< Args >(args)... );
            }
            return Result();
        }

        Result dispatch(mlir::Operation *op, Args &&... args)
        {
            return llvm::TypeSwitch< mlir::Operation *, Result >(op)
                .template Case<
                    mlir::ModuleOp,
                    mlir::FuncOp,
                    #define GET_OP_LIST
                    #include "vast/Dialect/HighLevel/HighLevel.cpp.inc"
                >( [&] (auto operation) -> Result {
                    return self().visit(operation, std::forward< Args >(args)... );
                })
                .Default([&] (auto operation) {
                    return visitIgnore(operation);
                });
        }

        void set_report_unhandled_cases(bool set = true)
        {
            report_unhandled_cases = set;
        }

    private:
        bool report_unhandled_cases = false;

    };

} // namespace vast::hl
