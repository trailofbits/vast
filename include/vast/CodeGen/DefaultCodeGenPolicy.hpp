#pragma once

#include "vast/CodeGen/CodeGenPolicy.hpp"
#include "vast/Frontend/Options.hpp"

namespace vast::cg {

    struct default_policy : policy_base
    {
        default_policy(cc::action_options &opts)
            : opts(opts)
            , default_missing_return(
                  opts.codegen.OptimizationLevel == 0 ? missing_return_policy::emit_trap
                                                      : missing_return_policy::emit_unreachable
              ) {}

        ~default_policy() = default;

        bool emit_strict_function_return([[maybe_unused]] const clang_function *decl
        ) const override {
            return opts.codegen.StrictReturn;
        };

        enum missing_return_policy
        missing_return_policy([[maybe_unused]] const clang_function *decl) const override {
            return default_missing_return;
        }

        bool skip_function_body([[maybe_unused]] const clang_function *decl) const override {
            return opts.front.SkipFunctionBodies;
        }

      protected:
        cc::action_options &opts;

      private:
        enum missing_return_policy default_missing_return;
    };

} // namespace vast::cg
