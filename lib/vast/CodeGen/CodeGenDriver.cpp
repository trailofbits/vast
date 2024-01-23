// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenDriver.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/Basic/TargetInfo.h>
VAST_UNRELAX_WARNINGS

namespace vast::cg
{
    defer_handle_of_top_level_decl::defer_handle_of_top_level_decl(
        codegen_driver &codegen, bool emit_deferred
    )
        : codegen(codegen), emit_deferred(emit_deferred)
    {
        ++codegen.deferred_top_level_decls;
    }

    defer_handle_of_top_level_decl::~defer_handle_of_top_level_decl() {
        unsigned level = --codegen.deferred_top_level_decls;
        if (level == 0 && emit_deferred) {
            codegen.build_deferred_decls();
        }
    }


    meta_generator_ptr make_meta_generator(codegen_context &cgctx, const cc::vast_args &vargs) {
        // TODO make configurable based on vast args
        if (vargs.has_option(cc::opt::locs_as_meta_ids)) {
            return std::make_unique< id_meta_gen >(&cgctx.actx, &cgctx.mctx);
        }
        return std::make_unique< default_meta_gen >(&cgctx.actx, &cgctx.mctx);
    }

    void codegen_driver::finalize() {
        codegen.emit_data_layout();
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
        if (lang().CUDA) {
            VAST_UNIMPLEMENTED_MSG("cuda module release");
        }
        // TODO: OpenMPRuntime
        // TODO: PGOReader
        // TODO: buildCtorList(GlobalCtors);
        // TODO: builtCtorList(GlobalDtors);
        // TODO: buildGlobalAnnotations();
        // TODO: buildDeferredUnusedCoverageMappings();
        // TODO: VASTGenPGO
        // TODO: CoverageMapping
        // TODO: if (codegen_opts.SanitizeCfiCrossDso) {
        // TODO: buildAtAvailableLinkGuard();
        const auto &target_triplet = acontext().getTargetInfo().getTriple();
        if (target_triplet.isWasm() && !target_triplet.isOSEmscripten()) {
            VAST_UNIMPLEMENTED_MSG("WASM module release");
        }

        // Emit reference of __amdgpu_device_library_preserve_asan_functions to
        // preserve ASAN functions in bitcode libraries.
        if (lang().Sanitize.has(clang::SanitizerKind::Address)) {
            VAST_UNIMPLEMENTED_MSG("AddressSanitizer module release");
        }

        // TODO: buildLLVMUsed();// TODO: SanStats

        // TODO: if (codegen_opts.Autolink) {
            // TODO: buildModuleLinkOptions
        // }

        // TODO: FINISH THE REST OF THIS
    }

    bool codegen_driver::verify_module() const {
        return codegen.verify_module();
    }

    void codegen_driver::build_deferred_decls() {
        if (deferred_inline_member_func_defs.empty())
            return;

        // Emit any deferred inline method definitions. Note that more deferred
        // methods may be added during this loop, since ASTConsumer callbacks can be
        // invoked if AST inspection results in declarations being added.
        auto deferred = deferred_inline_member_func_defs;
        deferred_inline_member_func_defs.clear();

        defer_handle_of_top_level_decl defer(*this);
        for (auto decl : deferred) {
            handle_top_level_decl(decl);
        }

        // Recurse to handle additional deferred inline method definitions.
        build_deferred_decls();
    }

    void codegen_driver::handle_top_level_decl(clang::DeclGroupRef decls) {
        defer_handle_of_top_level_decl defer(*this);

        for (auto decl : decls) {
            handle_top_level_decl(decl);
        }
    }

    void codegen_driver::handle_top_level_decl(clang::Decl *decl) {
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
                if (llvm::isa< clang::DecompositionDecl >(decl)) {
                    VAST_UNIMPLEMENTED_MSG("codegen for DecompositionDecl");
                }
                break;
            }
            case clang::Decl::CXXMethod:
            case clang::Decl::Function: {
                build_global(llvm::cast< clang::FunctionDecl >(decl));
                if (opts.codegen.CoverageMapping) {
                    VAST_UNIMPLEMENTED_MSG("coverage mapping not supported");
                }
                break;
            }
            default:
                codegen.Visit(decl);
        }
    }

    operation codegen_driver::build_global_definition(clang::GlobalDecl glob) {
        const auto *decl = llvm::cast< clang::ValueDecl >(glob.getDecl());

        if (const auto *fn = llvm::dyn_cast< clang::FunctionDecl >(decl)) {
            if (fn->isMultiVersion()) {
                VAST_UNIMPLEMENTED_MSG("codegen for multi version function");
            }

            if (const auto *method = llvm::dyn_cast< clang::CXXMethodDecl >(decl)) {
                VAST_UNIMPLEMENTED_MSG("cxx methods");
            }

            return build_global_function_definition(glob);
        }

        if (const auto *var = llvm::dyn_cast< clang::VarDecl >(decl))
            return build_global_var_definition(var, !var->hasDefinition());

        VAST_FATAL("Invalid argument to buildGlobalDefinition()");

    }

    operation codegen_driver::build_global_function_declaration(clang::GlobalDecl decl) {
        return codegen.build_function_prototype(decl);
    }

    operation codegen_driver::build_global_function_definition(clang::GlobalDecl decl) {
        auto fn = mlir::cast< hl::FuncOp >(build_global_function_declaration(decl));

        const auto *function_decl = llvm::cast< clang::FunctionDecl >(decl.getDecl());

        // Already emitted.
        if (!fn.isDeclaration()) {
            return fn;
        }

        // TODO setGVProperties
        // TODO MaubeHandleStaticInExternC
        // TODO maybeSetTrivialComdat
        // TODO setLLVMFunctionFEnvAttributes

        fn = build_function_body(fn, decl);

        // TODO: setNonAliasAttributes
        // TODO: SetLLVMFunctionAttributesForDeclaration

        if (function_decl->getAttr< clang::ConstructorAttr >()) {
            VAST_UNIMPLEMENTED_MSG("ctor emition");
        }

        if (function_decl->getAttr< clang::DestructorAttr >()) {
            VAST_UNIMPLEMENTED_MSG("dtor emition");
        }

        return fn;
    }

    operation codegen_driver::build_global_var_definition(const clang::VarDecl *decl, bool tentative) {
        VAST_UNIMPLEMENTED_IF(lang().OpenCL || lang().OpenMPIsTargetDevice);

        VAST_UNIMPLEMENTED_IF(decl->needsDestruction(acontext()) == clang::QualType::DK_cxx_destructor);

        VAST_UNIMPLEMENTED_IF(lang().CUDA);
        VAST_UNIMPLEMENTED_IF(lang().OpenMP);

        return codegen.Visit(decl);
    }

    operation codegen_driver::build_global_decl(const clang::GlobalDecl &/* decl */) {
        VAST_UNIMPLEMENTED;
    }

    operation codegen_driver::build_global(clang::GlobalDecl decl) {
        const auto *glob = llvm::cast< clang::ValueDecl >(decl.getDecl());

        VAST_UNIMPLEMENTED_IF(lang().CUDA);
        VAST_UNIMPLEMENTED_IF(lang().OpenMP);

        if (const auto *fn = llvm::dyn_cast< clang::FunctionDecl >(glob)) {
            // In contrast to clang codegen we emit also declarations.
            if (!fn->doesThisDeclarationHaveABody()) {
                if (!fn->doesDeclarationForceExternallyVisibleDefinition()) {
                    return build_global_function_declaration(decl);
                }

                VAST_UNIMPLEMENTED_MSG("FunctionDecl");
            }
        } else {
            const auto *var = llvm::cast< clang::VarDecl >(glob);
            VAST_CHECK(var->isFileVarDecl(), "Cannot emit local var decl as global.");
            if (var->isThisDeclarationADefinition() != clang::VarDecl::Definition &&
                !acontext().isMSStaticDataMemberInlineDefinition(var)
            ) {
                VAST_UNIMPLEMENTED_IF(lang().OpenMP);
                // If this declaration may have caused an inline variable definition
                // to change linkage, make sure that it's emitted.
                // TODO probably use GetAddrOfGlobalVar(var) below?
                VAST_UNIMPLEMENTED_IF(
                    acontext().getInlineVariableDefinitionKind(var) ==
                    clang::ASTContext::InlineVariableDefinitionKind::Strong
                );

                return build_global_var_definition(
                    var,
                    var->isThisDeclarationADefinition() == clang::VarDecl::TentativeDefinition
                );
            }
            if (var->getStorageClass() == clang::StorageClass::SC_Static) {
                return build_global_var_definition(
                    var,
                    var->isThisDeclarationADefinition() == clang::VarDecl::TentativeDefinition
                );
            }
        }

        // Defer code generation to first use when possible, e.g. if this is an inline
        // function. If the global mjust always be emitted, do it eagerly if possible
        // to benefit from cache locality.
        if (may_be_emitted_eagerly(glob)) {
            // Emit the definition if it can't be deferred.
            return build_global_definition(glob);
        }

        // If we're deferring emission of a C++ variable with an initializer, remember
        // the order in which it appeared on the file.
        if (lang().CPlusPlus && clang::isa< clang::VarDecl >(glob) &&
            clang::cast< clang::VarDecl >(glob)->hasInit()
        ) {
            VAST_UNIMPLEMENTED_MSG("build_global CXX GlobalVar");
            // DelayedCXXInitPosition[glob] = CXXGlobalInits.size();
            // CXXGlobalInits.push_back(nullptr);
        }

        auto mangled_name = cgctx.get_mangled_name(decl);
        if (cgctx.get_global_value(mangled_name) != nullptr) {
            // The value has already been used and should therefore be emitted.
            cgctx.add_deferred_decl_to_emit(decl);
        } else {
            // The value must be emitted, but cannot be emitted eagerly.
            VAST_ASSERT(!may_be_emitted_eagerly(glob));
            cgctx.add_deferred_decl_to_emit(decl);
        }

        return {};
    }

    bool codegen_driver::may_be_emitted_eagerly(const clang::ValueDecl *glob) {
        VAST_UNIMPLEMENTED_IF(lang().OpenMP);

        if (const auto *fn = llvm::dyn_cast< clang::FunctionDecl >(glob)) {
            // Implicit template instantiations may change linkage if they are later
            // explicitly instantiated, so they should not be emitted eagerly.
            constexpr auto implicit = clang::TSK_ImplicitInstantiation;
            VAST_UNIMPLEMENTED_IF(fn->getTemplateSpecializationKind() == implicit);
            VAST_UNIMPLEMENTED_IF(fn->isTemplated());
            return true;
        }


        if (const auto *vr = llvm::dyn_cast< clang::VarDecl >(glob)) {
            // A definition of an inline constexpr static data member may change
            // linkage later if it's redeclared outside the class.
            constexpr auto weak_unknown = clang::ASTContext::InlineVariableDefinitionKind::WeakUnknown;
            VAST_UNIMPLEMENTED_IF(acontext().getInlineVariableDefinitionKind(vr) == weak_unknown);
            return true;
        }

        VAST_UNIMPLEMENTED_MSG("unsupported value decl");
    }

    void codegen_driver::build_default_methods() {
        // Differently from deferred_decls_to_emit, there's no recurrent use of
        // deferred_decls_to_emit, so use it directly for emission.
        for (const auto &decl : cgctx.default_methods_to_emit) {
            build_global_decl(decl);
        }
    }

    void codegen_driver::build_deferred() {
        // Emit deferred declare target declarations
        VAST_UNIMPLEMENTED_IF(lang().OpenMP && !lang().OpenMPSimd);

        const auto &deferred_vtables = cgctx.deferred_vtables;
        const auto &deferred_decls_to_emit = cgctx.deferred_decls_to_emit;

        // Emit code for any potentially referenced deferred decls. Since a previously
        // unused static decl may become used during the generation of code for a
        // static function, iterate until no changes are made.
        VAST_UNIMPLEMENTED_IF(!deferred_vtables.empty());

        // Emit CUDA/HIP static device variables referenced by host code only. Note we
        // should not clear CUDADeviceVarODRUsedByHost since it is still needed for
        // further handling.
        VAST_UNIMPLEMENTED_IF(lang().CUDA && lang().CUDAIsDevice);

        // Stop if we're out of both deferred vtables and deferred declarations.
        if (deferred_decls_to_emit.empty()) {
            return;
        }

        // Grab the list of decls to emit. If build_global_definition schedules more
        // work, it will not interfere with this.
        auto curr_decls_to_emit = cgctx.receive_deferred_decls_to_emit();
        for (auto &decl : curr_decls_to_emit) {
            build_global_decl(decl);

            // FIXME: rework to worklist?
            // If we found out that we need to emit more decls, do that recursively.
            // This has the advantage that the decls are emitted in a DFS and related
            // ones are close together, which is convenient for testing.
            if (!deferred_vtables.empty() || !deferred_decls_to_emit.empty()) {
                build_deferred();
                VAST_ASSERT(deferred_vtables.empty() && deferred_decls_to_emit.empty());
            }
        }
    }

    void codegen_driver::add_replacement(string_ref name, mlir::Operation *op) {
        replacements[name] = op;
    }

    void codegen_driver::apply_replacements() {
        if (!replacements.empty()) {
            VAST_UNIMPLEMENTED_MSG(" function replacement in module release");
        }
    }

} // namespace vast::cg
