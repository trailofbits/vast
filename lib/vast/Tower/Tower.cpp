// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Tower/Tower.hpp"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace vast::tower
{
    auto manager_t::apply(handle_t handle, pass_ptr_t pass) -> handle_t {
        _modules.emplace_back(mlir::cast< vast_module >(handle.mod->clone()));

        auto id  = _modules.size() - 1;
        auto mod = _modules.back().get();

        mlir::PassManager pm(&_ctx);
        pm.addPass(std::move(pass));

        auto run_result = pm.run(_modules.back().get());
        VAST_CHECK(mlir::succeeded(run_result), "Some pass in apply() failed");

        return { id, mod };
    }
} // namespace vast::tower