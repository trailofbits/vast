// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Tower/Tower.hpp"

#include <gap/core/ranges.hpp>

namespace vast::tw {

    struct link_builder : mlir::PassInstrumentation
    {
        location_info_t &li;
        module_storage &storage;

        // Start empty and after each callback add to it.
        conversion_passes_t path   = {};
        // TODO: Remove.
        conversion_path_t str_path = {};

        std::vector< handle_t > handles;
        link_vector steps;

        explicit link_builder(location_info_t &li, module_storage &storage, handle_t root)
            : li(li), storage(storage), handles{ root } {}

        void runAfterPass(pass_ptr pass, operation op) override {
            auto current = op;
            while (current && !mlir::isa< mlir_module >(current)) {
                current = op->getParentOp();
            }
            auto mod = mlir::dyn_cast< mlir_module >(current);
            VAST_CHECK(mod, "Pass inside tower was not run on module!");

            // Update locations so each operation now has a unique loc that also
            // encodes backlink.
            path.emplace_back(pass);

            // Location transformation depends on the command line argument
            // of passes. If it is empty, don't perform location transformation.
            if (!pass->getArgument().empty()) {
                str_path.emplace_back(pass->getArgument().str());
                transform_locations(li, str_path, op);
            }

            // Clone the module to make it persistent
            owning_mlir_module_ref persistent = mlir::dyn_cast< mlir_module >(mod->clone());

            auto from = handles.back();
            handles.emplace_back(storage.store(path, std::move(persistent)));
            steps.emplace_back(std::make_unique< conversion_step >(from, handles.back(), li));
        }

        auto take_links() { return std::move(steps); }
    };

    namespace {

        link_vector
        construct_steps(const std::vector< handle_t > &handles, location_info_t &li) {
            VAST_ASSERT(handles.size() >= 2);
            link_vector out;
            for (std::size_t i = 1; i < handles.size(); ++i) {
                out.emplace_back(
                    std::make_unique< conversion_step >(handles[i - 1], handles[i], li)
                );
            }
            return out;
        }

    } // namespace

    link_vector tower::mk_full_path(handle_t root, location_info_t &li, mlir::PassManager &pm) {
        auto bld = std::make_unique< link_builder >(li, storage, top());

        // We need to access some of the data after passes are ran.
        auto raw_bld = bld.get();
        pm.addInstrumentation(std::move(bld));

        // We need to do a clone, because we received a handle - this means that the module
        // is already stored and should not be modified.
        auto clone = root.mod->clone();

        // TODO: What if this fails?
        std::ignore = pm.run(clone);
        return raw_bld->take_links();
    }

    link_ptr tower::apply(handle_t root, location_info_t &li, mlir::PassManager &requested_pm) {
        std::vector< mlir::Pass * > requested_passes;
        for (auto &p : requested_pm.getPasses()) {
            requested_passes.push_back(&p);
        }
        auto [handles, suffix] = storage.get_maximum_prefix_path(requested_passes, root);

        // This path is completely new.
        if (handles.empty()) {
            return std::make_unique< fat_link >(mk_full_path(root, li, requested_pm));
        }

        auto as_steps = construct_steps(handles, li);

        // This path is already present - construct a link.
        if (suffix.empty()) {
            return std::make_unique< fat_link >(std::move(as_steps));
        }

        auto pm = mlir::PassManager(requested_pm.getContext());
        copy_passes(pm, suffix);

        auto new_steps = mk_full_path(handles.back(), li, pm);
        // TODO: Update with newer stdlib
        as_steps.insert(
            as_steps.end(), std::make_move_iterator(new_steps.begin()),
            std::make_move_iterator(new_steps.end())
        );

        return std::make_unique< fat_link >(std::move(as_steps));
    }

} // namespace vast::tw
