// Copyright (c) 2023, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

namespace vast {

#ifdef VAST_ENABLE_EXCEPTIONS
    error_throwing_stream& vast_error() {
        static error_throwing_stream stream;
        return stream;
    }
#else
    llvm::raw_ostream& vast_error() {
        return llvm::dbgs();
    }
#endif

    llvm::raw_ostream& vast_debug() {
        return llvm::dbgs();
    }

} // namespace vast
