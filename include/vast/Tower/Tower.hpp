// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
VAST_UNRELAX_WARNINGS

#include "vast/Tower/Handle.hpp"
#include "vast/Tower/Link.hpp"
#include "vast/Tower/LocationInfo.hpp"
#include "vast/Tower/Storage.hpp"

namespace vast::tw {

    struct tower
    {
      private:
        [[maybe_unused]] mcontext_t &mctx;
        module_storage storage;
        handle_t top_handle;

      public:
        tower(mcontext_t &mctx, location_info_t &li, owning_module_ref root) : mctx(mctx) {
            mk_root(li, root->getOperation());
            top_handle = storage.store(root_conversion(), std::move(root));
        }

      private:
        // TODO: Move somewhere else.
        static conversion_path_t root_conversion() { return {}; }

      public:
        handle_t top() const { return top_handle; }

        link_ptr apply(handle_t, location_info_t &, mlir::PassManager &);
    };

} // namespace vast::tw
