
// Copyright (c) 2024, Trail of Bits, Inc.

#include "vast/Util/Snapshots.hpp"

namespace vast::util {

    void with_snapshots::runAfterPass(pass_ptr pass, operation op) {
        if (!should_snapshot(pass)) {
            return;
        }

        auto os = make_output_stream(pass);
        (*os) << *op;
    }

    auto with_snapshots::make_output_stream(pass_ptr pass)
        -> output_stream_ptr
    {
        std::string output_file_name = file_prefix + "." + pass->getArgument().str();
        std::error_code error_code;
        auto os = std::make_shared< llvm::raw_fd_ostream >(output_file_name, error_code);
        VAST_CHECK(!error_code, "Cannot open file to store snapshot, error code: {0}", error_code.message());
        return os;
    }

} // namespace vast::util
