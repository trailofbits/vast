// Copyright (c) 2023, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/Basic/TargetOptions.h>
#include <clang/Driver/DriverDiagnostic.h>
#include <clang/Driver/Options.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/FrontendPluginRegistry.h>
#include <llvm/Option/OptTable.h>
VAST_UNRELAX_WARNINGS

#include "vast/Frontend/Action.hpp"
#include "vast/Frontend/CompilerInstance.hpp"

namespace vast::cc
{
    frontend_action_ptr create_frontend_action(const vast_args &vargs) {
        if (opt::emit_only_mlir(vargs)) {
            return std::make_unique< vast::cc::emit_mlir_action >(vargs);
        }

        if (vargs.has_option(opt::emit_llvm)) {
            return std::make_unique< vast::cc::emit_llvm_action >(vargs);
        }

        if (vargs.has_option(opt::emit_asm)) {
            return std::make_unique< vast::cc::emit_assembly_action >(vargs);
        }

        if (vargs.has_option(opt::emit_obj)) {
            return std::make_unique< vast::cc::emit_obj_action >(vargs);
        }

        return nullptr;
    }

    frontend_action_ptr create_frontend_action(compiler_instance &ci, const vast_args &vargs) {
        if (auto action = create_frontend_action(vargs)) {
            return action;
        }

        auto &opts = ci.getFrontendOpts();
        auto act   = opts.ProgramAction;
        using namespace clang::frontend;

        switch (act) {
            case ASTDump:  return std::make_unique< clang::ASTDumpAction >();
            case EmitAssembly: return std::make_unique< vast::cc::emit_assembly_action >(vargs);
            case EmitLLVM: return std::make_unique< vast::cc::emit_llvm_action >(vargs);
            case EmitObj: return std::make_unique< vast::cc::emit_obj_action >(vargs);
            default: VAST_UNIMPLEMENTED_MSG("unsupported frontend action");
        }

        VAST_UNIMPLEMENTED_MSG("not implemented frontend action");
    }

    bool execute_compiler_invocation(compiler_instance *ci, const vast_args &vargs) {
        auto &opts = ci->getFrontendOpts();

        // Honor -help.
        if (opts.ShowHelp) {
            clang::driver::getDriverOptTable().printHelp(
                llvm::outs(), "vast-front -cc1 [options] file...",
                "VAST Compiler: https://github.com/trailofbits/vast",
                /*Include=*/clang::driver::options::CC1Option,
                /*Exclude=*/0, /*ShowAllAliases=*/false);
            return true;
        }

        // Honor -version.
        //
        // FIXME: Use a better -version message?
        if (opts.ShowVersion) {
            llvm::cl::PrintVersionMessage();
            return true;
        }

        ci->LoadRequestedPlugins();

        // FIXME: Honor -mllvm.

        // FIXME: CLANG_ENABLE_STATIC_ANALYZER

        // If there were errors in processing arguments, don't do anything else.
        if (ci->getDiagnostics().hasErrorOccurred())
            return false;

        // Create and execute the frontend action.
        auto action = create_frontend_action(*ci, vargs);
        if (!action)
            return false;

        bool success = ci->ExecuteAction(*action);

        if (opts.DisableFree) {
            llvm::BuryPointer(std::move(action));
        }

        return success;
    }

} // namespace vast::cc
