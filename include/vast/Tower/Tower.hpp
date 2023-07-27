// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Util/Common.hpp"

namespace mlir
{
    class Pass;
}

namespace vast::tower
{
    struct handle_t {
        std::size_t id;
        vast_module mod;
    };

    using pass_ptr_t = std::unique_ptr< mlir::Pass >;

    struct manager_t {
        static auto get(mcontext_t &ctx, owning_module_ref mod)
            -> std::tuple< manager_t, handle_t > {
            manager_t m(ctx, std::move(mod));
            handle_t h{ .id = 0, .mod = m._modules[0].get() };
            return { std::move(m), h };
        }

        auto apply(handle_t handle, pass_ptr_t pass) -> handle_t;

      private:
        using module_storage_t = llvm::SmallVector< owning_module_ref, 2 >;

        mcontext_t &_ctx;
        module_storage_t _modules;

        manager_t(mcontext_t &ctx, owning_module_ref mod)
            : _ctx(ctx) {
            _modules.emplace_back(std::move(mod));
        }
    };
} // namespace vast::tower