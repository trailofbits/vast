// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Tower/Tower.hpp"

namespace vast::tw {

    struct link_builder : mlir::PassInstrumentation
    {
        location_info_t &li;
        module_storage &storage;

        // Start empty and after each callback add to it.
        conversion_path_t path = {};

        std::vector< handle_t > handles;
        link_vector steps;

        explicit link_builder(location_info_t &li, module_storage &storage, handle_t root)
            : li(li), storage(storage), handles{ root } {}

        void runAfterPass(pass_ptr pass, operation op) override {
            auto mod = mlir::dyn_cast< core::module >(op);
            VAST_CHECK(mod, "Pass inside tower was not run on module!");

            // Update locations so each operation now has a unique loc that also
            // encodes backlink.
            path.emplace_back(pass->getArgument().str());
            transform_locations(li, path, mod);

            core::owning_module_ref persistent = mlir::dyn_cast< core::module >(op->clone());

            auto from = handles.back();
            handles.emplace_back(storage.store(path, std::move(persistent)));
            steps.emplace_back(std::make_unique< conversion_step >(from, handles.back(), li));
        }

        std::unique_ptr< link_interface > extract_link() {
            VAST_CHECK(!steps.empty(), "No conversions happened!");
            return std::make_unique< fat_link >(std::move(steps));
        }
    };

    link_ptr tower::apply(handle_t root, location_info_t &li, mlir::PassManager &pm) {
        auto bld = std::make_unique< link_builder >(li, storage, top());

        // We need to access some of the data after passes are ran.
        auto raw_bld = bld.get();
        pm.addInstrumentation(std::move(bld));

        // We need to do a clone, because we received a handle - this means that the module
        // is already stored and should not be modified.
        auto clone = root.mod->clone();

        // TODO: What if this fails?
        std::ignore = pm.run(clone);
        return raw_bld->extract_link();
    }

} // namespace vast::tw
