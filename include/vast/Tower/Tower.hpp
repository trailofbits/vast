// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/Pass.h>
VAST_UNRELAX_WARNINGS

namespace vast::tw {

    struct default_loc_rewriter_t
    {
        static auto insert(operation op) -> void;
        static auto remove(operation op) -> void;
        static auto prev(operation op) -> operation;
    };


    // Is allowed to have state?
    struct location_maker {
      private:
        // Encoded as `mlir::FusedLocation(original, mlir::OpaqueLocation(pointer_to_self))`
        using raw_loc_t = mlir::FusedLoc;

        static raw_loc_t raw_loc(operation op) {
            auto raw = mlir::dyn_cast< raw_loc_t >(op->getLoc());
            VAST_ASSERT(raw);
            return raw;
        }

        template< std::size_t idx > requires (idx < 2)
        static loc_t get(raw_loc_t raw) {
            auto locs = raw.getLocations();
            VAST_ASSERT(locs.size() == 2);
            return locs[idx];
        }

        static loc_t prev(raw_loc_t raw) { return get< 0 >(raw); }
        static loc_t self(raw_loc_t raw) { return get< 1 >(raw); }

        static loc_t prev(operation op) { return prev(raw_loc(op)); }
        static loc_t self(operation op) { return self(raw_loc(op)); }

        static auto parse(operation op) {
            return std::make_tuple(prev(op), self(op));
        }

      public:
        // For the given operation return location to be used in this module.
        loc_t next(operation low_op);

        struct location_query {
            // Returns `true` if `low` and `high` are linked
            static bool are_tied(operation high, operation low);
        };
    };

    template< typename loc_rewriter_t >
    struct tower
    {
        using loc_rewriter = loc_rewriter_t;

        struct handle_t
        {
            std::size_t id;
            vast_module mod;
        };

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
