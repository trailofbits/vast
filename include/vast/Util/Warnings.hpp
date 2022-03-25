/*
 * Copyright (c) 2021 Trail of Bits, Inc.
 */

#pragma once

#define VAST_RELAX_WARNINGS \
  _Pragma( "clang diagnostic push" ) \
  _Pragma( "clang diagnostic ignored \"-Wsign-conversion\"" ) \
  _Pragma( "clang diagnostic ignored \"-Wconversion\"" ) \
  _Pragma( "clang diagnostic ignored \"-Wold-style-cast\"" ) \
  _Pragma( "clang diagnostic ignored \"-Wunused-parameter\"" ) \
  _Pragma( "clang diagnostic ignored \"-Wcast-align\"" ) \
  _Pragma( "clang diagnostic ignored \"-Wimplicit-int-conversion\"" ) \
  _Pragma( "clang diagnostic ignored \"-Wambiguous-reversed-operator\"" )

#define VAST_UNRELAX_WARNINGS \
  _Pragma( "clang diagnostic pop" )

VAST_RELAX_WARNINGS
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/Debug.h>
VAST_UNRELAX_WARNINGS

#include <sstream>
#include <stdexcept>
#include <utility>

#define DEBUG_TYPE "vast"

namespace vast {

    class Exception : public std::runtime_error {
       public:
          Exception(const std::string& what) : std::runtime_error(what) {}
          Exception(const char* what) : std::runtime_error(what) {}
    };

    template <typename T = Exception>
    class ExceptionThrower {
        std::stringstream stream;
        bool triggered, moved = false;

       public:
        ExceptionThrower(bool cond = true, std::stringstream ss = std::stringstream())
              : stream(std::move(ss)), triggered(cond) {}
        ~ExceptionThrower() noexcept(false) {
            if (triggered && !moved) {
              throw T(stream.str());
            }
        }

        template <typename V>
        ExceptionThrower<T> operator<<(V&& s) {
            moved = true;
            stream << std::forward<V>(s);
            return ExceptionThrower<T>(triggered, std::move(stream));
        }
    };


    #define VAST_THROW_IF(cond, msg) ExceptionThrower<Exception>((cond)) << msg

    #define VAST_THROW(msg) VAST_THROW_IF(true, msg)

    [[ noreturn,gnu::unused ]] static
    void unreachable_intrinsic(const char *msg,
                               const char *file = nullptr,
                               unsigned line = 0) {
        VAST_THROW(msg);
        (void)file, (void)line;
        __builtin_unreachable();
    }


    #define VAST_UNREACHABLE(fmt, ...) unreachable_intrinsic( llvm::formatv(fmt __VA_OPT__(,) __VA_ARGS__).str().c_str() );

    #define VAST_UNIMPLEMENTED VAST_UNREACHABLE("not implemented: {}", __PRETTY_FUNCTION__);

    #define VAST_DEBUG(fmt, ...) LLVM_DEBUG(llvm::dbgs() << llvm::formatv(fmt, __VA_OPT__(,) __VA_ARGS__));

    #define VAST_CHECK(cond, fmt, ...) if (!(cond)) { VAST_UNREACHABLE(fmt __VA_OPT__(,) __VA_ARGS__); }

    #define VAST_ASSERT(cond) if (!(cond)) { VAST_UNREACHABLE("assertion: " #cond " failed"); }
}
