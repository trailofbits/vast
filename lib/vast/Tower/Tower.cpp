// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Tower/Tower.hpp"

#include "mlir/Pass/PassManager.h"
#include "vast/Util/Common.hpp"

namespace vast::tower
{
    static auto insert_op_ptr_loc(mlir::Operation *op) -> void {
        auto ctx = op->getContext();
        auto ol  = mlir::OpaqueLoc::get< mlir::Operation  *>(op, ctx);
        op->setLoc(mlir::FusedLoc::get({ op->getLoc() }, ol, ctx));
    }

    static auto get_op_ptr_loc(mlir::Operation *op) -> mlir::Operation * {
        auto fl = mlir::cast< mlir::FusedLoc >(op->getLoc());
        auto ol = mlir::cast< mlir::OpaqueLoc >(fl.getMetadata());
        return mlir::OpaqueLoc::getUnderlyingLocation< mlir::Operation * >(ol);
    }

    static auto remove_op_ptr_loc(mlir::Operation *op) -> void {
        auto fl = mlir::cast< mlir::FusedLoc >(op->getLoc());
        op->setLoc(fl.getLocations().front());
    }

    auto manager_t::apply(handle_t handle, pass_ptr_t pass) -> handle_t {
        handle.mod.walk(insert_op_ptr_loc);

        _modules.emplace_back(mlir::cast< vast_module >(handle.mod->clone()));

        auto id  = _modules.size() - 1;
        auto mod = _modules.back().get();

        mlir::PassManager pm(&_ctx);
        pm.addPass(std::move(pass));

        VAST_CHECK(mlir::succeeded(pm.run(mod)), "Some pass in apply() failed");

        mlir::IRMapping prev;

        mod.walk([&prev](mlir::Operation *op) {
            prev.map(op, get_op_ptr_loc(op));
            remove_op_ptr_loc(op);
        });

        return { id, mod, prev };
    }
} // namespace vast::tower