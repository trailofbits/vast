// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/CodeGen/Module.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Verifier.h>
VAST_UNRELAX_WARNINGS

namespace vast::cg {

    void codegen_module::release() {
        build_deferred();
        // TODO: buildVTablesOpportunistically();
        // TODO: applyGlobalValReplacements();
        apply_replacements();
        // TODO: checkAliases();
        // TODO: buildMultiVersionFunctions();
        // TODO: buildCXXGlobalInitFunc();
        // TODO: buildCXXGlobalCleanUpFunc();
        // TODO: registerGlobalDtorsWithAtExit();
        // TODO: buildCXXThreadLocalInitFunc();
        // TODO: ObjCRuntime
        if (actx.getLangOpts().CUDA) {
            throw cc::compiler_error("unsupported cuda module release");
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
            throw cc::compiler_error("unsupported SanitizeCfiCrossDso module release");
        }
        // TODO: buildAtAvailableLinkGuard();
        const auto &target_triplet = actx.getTargetInfo().getTriple();
        if (target_triplet.isWasm() && !target_triplet.isOSEmscripten()) {
            throw cc::compiler_error("unsupported WASM module release");
        }

        // Emit reference of __amdgpu_device_library_preserve_asan_functions to
        // preserve ASAN functions in bitcode libraries.
        if (lang_opts.Sanitize.has(clang::SanitizerKind::Address)) {
            throw cc::compiler_error("unsupported AddressSanitizer module release");
        }

        // TODO: buildLLVMUsed();// TODO: SanStats

        if (codegen_opts.Autolink) {
            // TODO: buildModuleLinkOptions
        }

        // TODO: FINISH THE REST OF THIS
    }

    void codegen_module::add_replacement(string_ref name, mlir::Operation *op) {
        replacements[name] = op;
    }

    void codegen_module::apply_replacements() {
        if (!replacements.empty()) {
            throw cc::compiler_error("unsupported function replacement in module release");
        }
    }

    bool codegen_module::verify_module() {
        return mlir::verify(mod).succeeded();
    }

    void codegen_module::build_global_decl(clang::GlobalDecl &/* decl */) {
        throw cc::compiler_error("build_global_decl not implemented");
    }

    void codegen_module::build_global(clang::GlobalDecl /*decl*/) {
        throw cc::compiler_error("build_global not implemented");
    }

    void codegen_module::build_top_level_decl(clang::Decl *decl) {
        // Ignore dependent declarations
        if (decl->isTemplated())
            return;

        // Consteval function shouldn't be emitted.
        if (auto *fn = llvm::dyn_cast< clang::FunctionDecl >(decl)) {
            if (fn->isConsteval()) {
                return;
            }
        }

        switch (decl->getKind()) {
            case clang::Decl::Var:
            case clang::Decl::Decomposition:
            case clang::Decl::VarTemplateSpecialization: {
                build_global(llvm::cast< clang::VarDecl >(decl));
                if (!llvm::isa< clang::DecompositionDecl >(decl)) {
                    throw cc::compiler_error("codegen for DecompositionDecl not implemented");
                }
                break;
            }
            case clang::Decl::CXXMethod:
            case clang::Decl::Function: {
                build_global(llvm::cast< clang::FunctionDecl >(decl));
                if (!codegen_opts.CoverageMapping) {
                    throw cc::compiler_error("codegen Coverage Mapping not supported");
                }
                break;
            }
            // case clang::Decl::Namespace:
            // case clang::Decl::ClassTemplateSpecialization:
            // case clang::Decl::CXXRecord:
            // case clang::Decl::UsingShadow:
            // case clang::Decl::ClassTemplate:
            // case clang::Decl::VarTemplate:
            // case clang::Decl::Concept:
            // case clang::Decl::VarTemplatePartialSpecialization:
            // case clang::Decl::FunctionTemplate:
            // case clang::Decl::TypeAliasTemplate:
            // case clang::Decl::Block:
            // case clang::Decl::Empty:
            // case clang::Decl::Binding:
            // case clang::Decl::Using:
            // case clang::Decl::UsingEnum:
            // case clang::Decl::NamespaceAlias:
            // case clang::Decl::UsingDirective:
            // case clang::Decl::CXXConstructor:
            // case clang::Decl::StaticAssert:
            // case clang::Decl::Typedef:
            // case clang::Decl::TypeAlias:
            // case clang::Decl::Record:
            // case clang::Decl::Enum:
            default: throw cc::compiler_error(
                std::string("codegen for '") + decl->getDeclKindName() + "' not implemented"
            );
        }
    }

    void codegen_module::build_default_methods() {
        // Differently from deferred_decls_to_emit, there's no recurrent use of
        // deferred_decls_to_emit, so use it directly for emission.
        for (auto &decl : default_methods_to_emit) {
            build_global_decl(decl);
        }
    }

    void codegen_module::build_deferred() {
        // Emit deferred declare target declarations
        if (lang_opts.OpenMP && !lang_opts.OpenMPSimd) {
            throw cc::compiler_error("build_deferred for openmp not implemented");
        }

        // Emit code for any potentially referenced deferred decls. Since a previously
        // unused static decl may become used during the generation of code for a
        // static function, iterate until no changes are made.
        if (!deferred_vtables.empty()) {
            throw cc::compiler_error("build_deferred for vtables not implemented");
        }

        // Emit CUDA/HIP static device variables referenced by host code only. Note we
        // should not clear CUDADeviceVarODRUsedByHost since it is still needed for
        // further handling.
        if (lang_opts.CUDA && lang_opts.CUDAIsDevice) {
            throw cc::compiler_error("build_deferred for cuda not implemented");
        }

        // Stop if we're out of both deferred vtables and deferred declarations.
        if (deferred_decls_tot_emit.empty())
            return;

        // Grab the list of decls to emit. If buildGlobalDefinition schedules more
        // work, it will not interfere with this.
        std::vector< clang::GlobalDecl > curr_decls_to_emit;
        curr_decls_to_emit.swap(deferred_decls_tot_emit);

        for (auto &decl : curr_decls_to_emit) {
            build_global_decl(decl);

            // FIXME: rework to worklist?
            // If we found out that we need to emit more decls, do that recursively.
            // This has the advantage that the decls are emitted in a DFS and related
            // ones are close together, which is convenient for testing.
            if (!deferred_vtables.empty() || !deferred_decls_tot_emit.empty()) {
                build_deferred();
                assert(deferred_vtables.empty() && deferred_decls_tot_emit.empty());
            }
        }
    }

} // namespace vast::cg
