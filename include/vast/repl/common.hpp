// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"

#include <exception>

namespace vast::repl {

    template< typename ... Args >
    [[noreturn]] void throw_error(Args && ... args) {
        throw std::runtime_error(llvm::formatv(std::forward< Args >(args) ...).str());
    }

} // namespace vast::repl
