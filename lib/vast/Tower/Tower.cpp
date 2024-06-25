// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Tower/Tower.hpp"

namespace vast::tw {

    void dbg(operation op) {
        mlir::OpPrintingFlags flags;
        op->print(llvm::outs(), flags.enableDebugInfo(true, false));
    }

    struct link_builder : mlir::PassInstrumentation
    {
        location_info &li;
        module_storage &storage;

        // Start empty and after each callback add to it.
        conversion_path_t path = {};

        std::vector< handle_t > handles;

        explicit link_builder(location_info &li, module_storage &storage, handle_t root)
            : li(li), storage(storage), handles{ root } {}

        void runAfterPass(pass_ptr pass, operation op) override {
            auto mod = mlir::dyn_cast< vast_module >(op);
            VAST_CHECK(mod, "Pass inside tower was not run on module!");

            // Update locations so each operation now has a unique loc that also
            // encodes backlink.
            path.emplace_back(pass->getArgument().str());
            transform_locations(li, path, mod);

            owning_module_ref persistent = mlir::dyn_cast< vast_module >(op->clone());

            handles.emplace_back(storage.store(path, std::move(persistent)));
        }

        // From all the handles create a chain of one step transitions.
        unit_link_vector link_vector() {
            unit_link_vector out;
            for (std::size_t i = 1; i < handles.size(); ++i) {
                auto step =
                    std::make_unique< light_one_step_link >(handles[i - 1], handles[i], li);
                out.emplace_back(std::move(step));
            }
            return out;
        }

        std::unique_ptr< link_interface > extract_link() {
            auto unit_links = link_vector();
            return std::make_unique< fat_link >(std::move(unit_links));
        }
    };

    link_ptr tower::apply(handle_t root, location_info &li, mlir::PassManager &pm) {
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
