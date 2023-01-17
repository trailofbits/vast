// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/CodeGen/Module.hpp"

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

} // namespace vast::cg
