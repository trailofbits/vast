// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include <stdexcept>
#include <string>
#include <system_error>

namespace vast::util
{
    /* This is a base class for exceptions which arise from improper use at the
     * user level (as opposed to programmer level). Use this if you expect an
     * exception to be seen by actual users. */
    struct error : std::runtime_error {
        int exit;

        explicit error(std::string err, int exit = 1)
            : std::runtime_error(err)
            , exit(exit) {}
    };

    struct system_error : std::system_error {
        explicit system_error(const char *w)
            : std::system_error(errno, std::system_category(), w) {}
    };

    struct finally {
        using fun = std::function< void() >;
        fun _fun;
        finally(fun f)
            : _fun(f) {}
        ~finally() {
            if (_fun)
                _fun();
        }
        void skip() { _fun = fun(); }
    };

} // namespace vast::util
