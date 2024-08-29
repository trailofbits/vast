#pragma once

#include "vast/CodeGen/CodeGenPolicy.hpp"
#include "vast/CodeGen/Common.hpp"
#include "vast/Frontend/Options.hpp"

namespace vast::cg {

    struct default_policy : codegen_policy
    {
        default_policy(cc::action_options &opts)
            : opts(opts)
        {}

        ~default_policy() = default;

        bool emit_strict_function_return(const clang_function * /* decl */) const override {
            return opts.codegen.StrictReturn;
        };

        missing_return_policy get_missing_return_policy(const clang_function * /* decl */) const override {
            return opts.codegen.OptimizationLevel == 0
                ? missing_return_policy::emit_trap
                : missing_return_policy::emit_unreachable;
        }

        bool skip_function_body(const clang_function * /* decl */) const override {
            return opts.front.SkipFunctionBodies;
        }

        bool skip_global_initializer(const clang_var_decl * /* decl */) const override {
            return false;
        };

      protected:
        cc::action_options &opts;
    };

} // namespace vast::cg
