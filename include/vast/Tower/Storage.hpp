// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
VAST_UNRELAX_WARNINGS

#include "vast/Tower/Handle.hpp"

#include "vast/Dialect/Core/CoreOps.hpp"

#include <algorithm>
#include <numeric>

namespace vast::tw {

    struct module_storage
    {
        // TODO: API-wise, we probably want to accept any type that is `mlir::OwningOpRef< T >`?
        handle_t store(const conversion_path_t &path, owning_mlir_module_ref mod) {
            auto id      = allocate_id(path);
            auto [it, _] = storage.insert({id, std::move(mod)});
            return { id, it->second.get() };
        }

        void remove(handle_t) { VAST_UNIMPLEMENTED; }

      private:
        conversion_path_fingerprint_t fingerprint(const conversion_path_t &path) const {
            return std::accumulate(path.begin(), path.end(), std::string{});
        }

        handle_id_t allocate_id(const conversion_path_t &path) {
            return allocate_id(fingerprint(path));
        }

        handle_id_t allocate_id(const conversion_path_fingerprint_t &fp) {
            // Later here we want to return the cached module?
            VAST_CHECK(!conversion_tree.count(fp), "For now cannot do caching!");
            auto id = next_id++;
            conversion_tree.emplace(fp, id);
            return id;
        }

        std::size_t next_id = 0;
        llvm::DenseMap< handle_id_t, owning_mlir_module_ref > storage;
        // TODO: This is just a prototyping shortcut, we may want something smarter here.
        std::unordered_map< conversion_path_fingerprint_t, handle_id_t > conversion_tree;
    };
} // namespace vast::tw
