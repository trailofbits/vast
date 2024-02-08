/*
 * Copyright (c) 2021 Trail of Bits, Inc.
 */

#pragma once

#define VAST_COMMON_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic ignored \"-Wsign-conversion\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wconversion\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wold-style-cast\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wunused-parameter\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wcast-align\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Woverloaded-virtual\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wctad-maybe-unsupported\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wdouble-promotion\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wshadow\"") \
  _Pragma( "GCC diagnostic ignored \"-Wunused-function\"") \
  _Pragma( "GCC diagnostic ignored \"-Wdeprecated-this-capture\"")

#define VAST_CLANG_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic ignored \"-Wambiguous-reversed-operator\"" )

#define VAST_GCC_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic ignored \"-Wuseless-cast\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wnull-dereference\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wmaybe-uninitialized\"" )

#ifdef __clang__
#define VAST_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic push" ) \
  VAST_COMMON_RELAX_WARNINGS \
  VAST_CLANG_RELAX_WARNINGS
#elif __GNUC__
#define VAST_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic push" ) \
  VAST_COMMON_RELAX_WARNINGS \
  VAST_GCC_RELAX_WARNINGS
#else
#define VAST_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic push" ) \
  VAST_COMMON_RELAX_WARNINGS
#endif

#define VAST_UNRELAX_WARNINGS \
  _Pragma( "GCC diagnostic pop" )

VAST_RELAX_WARNINGS
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/Debug.h>
VAST_UNRELAX_WARNINGS

#ifdef VAST_ENABLE_EXCEPTIONS
#include <stdexcept>
#include <sstream>
#endif

#define DEBUG_TYPE "vast"

namespace vast {

#ifdef VAST_ENABLE_EXCEPTIONS
    struct error_throwing_stream {
        explicit error_throwing_stream() : ss(buff) {}

        ~error_throwing_stream() noexcept(false) {
            ss.flush();
            throw std::runtime_error(buff);
        }

        template< typename T >
        error_throwing_stream& operator<<(T &&value) {
            ss << std::forward< T >(value);
            return *this;
        }

      private:
        std::string buff;
        llvm::raw_string_ostream ss;
    };

    error_throwing_stream vast_error();
#else
    llvm::raw_ostream& vast_error();
#endif

    llvm::raw_ostream& vast_debug();

    #define VAST_ERROR(...) do { \
      vast_error() << "[VAST error] " << llvm::formatv(__VA_ARGS__) << "\n"; \
    } while(0)


    #define VAST_UNREACHABLE(...) do { \
      VAST_ERROR(__VA_ARGS__); \
      llvm_unreachable(nullptr); \
    } while (0)

    #define VAST_TRAP LLVM_BUILTIN_TRAP

    #define VAST_FATAL(...) do { \
      vast_error() << "[VAST fatal] " << llvm::formatv(__VA_ARGS__) << "\n"; \
      VAST_TRAP; \
    } while(0)

    #define VAST_CHECK(cond, fmt, ...) \
        if (!(cond)) { VAST_FATAL(fmt __VA_OPT__(,) __VA_ARGS__); }

    #define VAST_UNIMPLEMENTED \
        VAST_FATAL("not implemented: {0}", __PRETTY_FUNCTION__)

    #define VAST_UNIMPLEMENTED_MSG(msg) \
        VAST_FATAL("not implemented: {0} because {1}", __PRETTY_FUNCTION__, msg)

    #define VAST_UNIMPLEMENTED_IF(cond) \
        if (cond) { VAST_FATAL("not implemented: {0}", __PRETTY_FUNCTION__); }


    #define VAST_TODO(fmt, ... ) \
        VAST_FATAL("[VAST TODO]: " # fmt __VA_OPT__(,) __VA_ARGS__ )

    #if !defined(NDEBUG)
        #define VAST_ASSERT(cond) \
            if (!(cond)) { \
                vast_error() << "[VAST assert] " << llvm::formatv(#cond) << " failed\n"; \
                VAST_TRAP; \
            }

        #define VAST_REPORT(...) do { \
          vast_debug() << "[VAST debug] " << llvm::formatv(__VA_ARGS__) << "\n"; \
        } while(0)

        #define VAST_REPORT_WITH_PREFIX(prefix, ...) do { \
          vast_debug() << "[VAST debug] " << prefix << llvm::formatv(__VA_ARGS__) << "\n"; \
        } while(0)

    #elif defined(VAST_RELEASE_WITH_ASSERTS)
        #define VAST_ASSERT(...) VAST_CHECK(__VA_ARGS__)
        #define VAST_REPORT(...)
        #define VAST_REPORT_WITH_PREFIX(...)
    #else
        #define VAST_ASSERT(...)
        #define VAST_REPORT(...)
        #define VAST_REPORT_WITH_PREFIX(...)
    #endif

    #define VAST_REPORT_IF(cond, ...) do { \
      if constexpr (cond) { \
        VAST_REPORT(__VA_ARGS__); \
      } \
    } while(0)

    #define VAST_REPORT_WITH_PREFIX_IF(cond, prefix, ...) do { \
      if constexpr (cond) { \
        VAST_REPORT_WITH_PREFIX(prefix, __VA_ARGS__); \
      } \
    } while(0)

} // namespace vast
