// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/Pass.h>
VAST_UNRELAX_WARNINGS

#include "vast/Tower/Handle.hpp"
#include "vast/Tower/Link.hpp"
#include "vast/Tower/LocationInfo.hpp"
#include "vast/Tower/Storage.hpp"

namespace vast::tw {

    struct default_loc_rewriter_t
    {
        static auto insert(operation op) -> void;
        static auto remove(operation op) -> void;
        static auto prev(operation op) -> operation;
    };

    template< typename loc_rewriter_t >
    struct tower
    {
        using loc_rewriter = loc_rewriter_t;

        module_storage storage;

        using application_result_t = std::tuple< handle_t, link_t >;
        application_result_t apply(handle_t root, const conversion_path_t &path);

     private:
        void make_top();

     public:

        static auto get(mcontext_t &ctx, owning_module_ref mod)
            -> std::tuple< tower, handle_t > {
            tower t(ctx, std::move(mod));
            handle_t h{ .id = 0, .mod = t._modules[0].get() };
            return { std::move(t), h };
        }

        auto apply(handle_t handle, mlir::PassManager &pm) -> handle_t {
            handle.mod.walk(loc_rewriter::insert);

            _modules.emplace_back(mlir::cast< vast_module >(handle.mod->clone()));

            auto id  = _modules.size() - 1;
            auto mod = _modules.back().get();

            if (mlir::failed(pm.run(mod))) {
                VAST_FATAL("some pass in apply() failed");
            }

            handle.mod.walk(loc_rewriter::remove);

            return { id, mod };
        }

        auto apply(handle_t handle, owning_pass_ptr pass) -> handle_t {
            mlir::PassManager pm(_ctx);
            pm.addPass(std::move(pass));
            return apply(handle, pm);
        }

        auto top() -> handle_t { return { _modules.size(), _modules.back().get() }; }

      private:
        using module_storage_t = llvm::SmallVector< owning_module_ref, 2 >;

        mcontext_t *_ctx;
        module_storage_t _modules;

        tower(mcontext_t &ctx, owning_module_ref mod) : _ctx(&ctx) {
            _modules.emplace_back(std::move(mod));
        }
    };

    using default_tower = tower< default_loc_rewriter_t >;

} // namespace vast::tw
