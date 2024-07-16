// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/Tower/PassUtils.hpp"

namespace vast::tw {

    std::string to_string(mlir::Pass *pass) {
        std::string buffer;
        llvm::raw_string_ostream os(buffer);
        pass->printAsTextualPipeline(os);
        return os.str();
    }

    std::string to_string(const conversion_passes_t &passes) {
        std::string out;
        for (auto p : passes)
            out += to_string(p) + ",";

        if (!out.empty())
            out.pop_back();
        return out;
    }

    void copy_passes(mlir::PassManager &pm, const conversion_passes_t &passes) {
        // Sadly I didn't find any better public API to clone passes between `mlir::PassManager`
        // instances. Pass does have a `clonePass` method but it is `protected` and same holds
        // for other utilities as well.
        auto str_pipeline = to_string(passes);
        auto status = mlir::parsePassPipeline(str_pipeline, pm);
        VAST_CHECK(mlir::succeeded(status), "Failed to parse pipeline string: {0}", str_pipeline);
    }

} // namespace vast::tw
