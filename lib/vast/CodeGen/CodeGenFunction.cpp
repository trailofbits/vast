// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenDriver.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/Basic/TargetInfo.h>
VAST_UNRELAX_WARNINGS

namespace vast::cg
{
    namespace detail {
        template< typename generator_t >
        std::unique_ptr< generator_t > generate(clang::FunctionDecl *decl, mangler_t &mangler) {
            auto gen = std::make_unique< generator_t >();
            gen->emit(decl, mangler);
            return gen;
        }
    } // namespace detail

    void function_generator::emit(clang::FunctionDecl *decl, mangler_t &mangler) {
        hook(generate_prototype(decl, mangler));
    }

    void prototype_generator::emit(clang::FunctionDecl *decl, mangler_t &mangler) {
        VAST_UNIMPLEMENTED;
    }

    std::unique_ptr< function_generator > generate_function(
        clang::FunctionDecl *decl, mangler_t &mangler
    ) {
        return detail::generate< function_generator >(decl, mangler);
    }

    std::unique_ptr< prototype_generator > generate_prototype(
        clang::FunctionDecl *decl, mangler_t &mangler
    ) {
        return detail::generate< prototype_generator >(decl, mangler);
    }

} // namespace vast::cg
