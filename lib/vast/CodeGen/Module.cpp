// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/CodeGen/Module.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Verifier.h>
VAST_UNRELAX_WARNINGS

namespace vast::cg {

    void codegen_module::release() {
        // TODO: buildDeferred();
        // TODO: buildVTablesOpportunistically();
        // TODO: applyGlobalValReplacements();
        // TODO: applyReplacements();
        // TODO: checkAliases();
        // TODO: buildMultiVersionFunctions();
        // TODO: buildCXXGlobalInitFunc();
        // TODO: buildCXXGlobalCleanUpFunc();
        // TODO: registerGlobalDtorsWithAtExit();
        // TODO: buildCXXThreadLocalInitFunc();
        // TODO: ObjCRuntime
        if (actx.getLangOpts().CUDA) {
            llvm_unreachable("unsupported cuda module release");
        }
        // TODO: OpenMPRuntime
        // TODO: PGOReader
        // TODO: buildCtorList(GlobalCtors);
        // TODO: builtCtorList(GlobalDtors);
        // TODO: buildGlobalAnnotations();
        // TODO: buildDeferredUnusedCoverageMappings();
        // TODO: CIRGenPGO
        // TODO: CoverageMapping
        if (codegen_opts.SanitizeCfiCrossDso) {
            llvm_unreachable("unsupported SanitizeCfiCrossDso module release");
        }
        // TODO: buildAtAvailableLinkGuard();
        if (actx.getTargetInfo().getTriple().isWasm() && !actx.getTargetInfo().getTriple().isOSEmscripten()) {
            llvm_unreachable("unsupported WASM module release");
        }

        // Emit reference of __amdgpu_device_library_preserve_asan_functions to
        // preserve ASAN functions in bitcode libraries.
        if (lang_opts.Sanitize.has(clang::SanitizerKind::Address)) {
            llvm_unreachable("unsupported AddressSanitizer module release");
        }

        // TODO: buildLLVMUsed();// TODO: SanStats

        if (codegen_opts.Autolink) {
            // TODO: buildModuleLinkOptions
        }

        // TODO: FINISH THE REST OF THIS
    }

    bool codegen_module::verify_module() {
        return mlir::verify(mod).succeeded();
    }

    void codegen_module::build_global_decl(clang::GlobalDecl &/* decl */) {
        throw cc::compiler_error("build_global_decl not implemented");
    }

    void codegen_module::build_default_methods() {
        // Differently from deferred_decls_to_emit, there's no recurrent use of
        // deferred_decls_to_emit, so use it directly for emission.
        for (auto &decl : default_methods_to_emit) {
            build_global_decl(decl);
        }
    }

} // namespace vast::cg
