// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenFunction.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/Basic/TargetInfo.h>
VAST_UNRELAX_WARNINGS

namespace vast::cg
{
    namespace detail {
        template< typename generator_t, typename action_t >
        auto generate(emition_kind emition, action_t &&action)
            -> std::unique_ptr< generator_t >
        {
            auto gen = std::make_unique< generator_t >();

            if (emition == emition_kind::immediate) {
                action(gen.get());
            } else {
                gen->defer([action = std::forward< action_t >(action), gen = gen.get()] {
                    action(gen);
                });
            }

            return gen;
        }

        template< typename generator_t >
        auto generate(emition_kind emition, clang_function *decl, mangler_t &mangler)
            -> std::unique_ptr< generator_t >
        {
            return generate< generator_t >(emition, [decl, &mangler] (auto *gen) {
                gen->emit(decl, mangler);
            });
        }

        template< typename generator_t >
        auto generate(emition_kind emition, clang_function *decl)
            -> std::unique_ptr< generator_t >
        {
            return generate< generator_t >(emition, [decl] (auto *gen) {
                gen->emit(decl);
            });
        }
    } // namespace detail

    void function_generator::emit(clang_function *decl, mangler_t &mangler) {
        hook(generate_prototype(decl, mangler));
        hook(generate_body(decl, emition_kind::deferred));
    }

    void prototype_generator::emit(clang_function *decl, mangler_t &mangler) {
        VAST_UNIMPLEMENTED;
    }

    void body_generator::emit(clang_function *decl) {
        VAST_UNIMPLEMENTED;
        emit_epilogue(decl);
    }

    void body_generator::emit_epilogue(clang_function *decl) {
        VAST_UNIMPLEMENTED;
    }

    auto generate_function(clang_function *decl, mangler_t &mangler)
        -> std::unique_ptr< function_generator >
    {
        return detail::generate< function_generator >(emition_kind::immediate, decl, mangler);
    }

    auto generate_prototype(clang_function *decl, mangler_t &mangler)
        -> std::unique_ptr< prototype_generator >
    {
        return detail::generate< prototype_generator >(emition_kind::immediate, decl, mangler);
    }

    auto generate_body(clang_function *decl, emition_kind emition)
        -> std::unique_ptr< body_generator >
    {
        return detail::generate< body_generator >(emition, decl);
    }

} // namespace vast::cg
