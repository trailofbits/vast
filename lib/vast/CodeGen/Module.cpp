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

    x86_avx_abi_level avx_level(const clang::TargetInfo &target) {
        const auto &abi = target.getABI();

        if (abi == "avx512")
            return x86_avx_abi_level::avx512;
        if (abi == "avx")
            return x86_avx_abi_level::avx;
        return x86_avx_abi_level::none;
    }

    using target_info_ptr = std::unique_ptr< target_info_t >;
    target_info_ptr get_target_info_impl(
        const clang::TargetInfo &target, types_generator &types
    ) {
        const auto &triple = target.getTriple();
        const auto &abi = target.getABI();

        auto aarch64_info = [&] {
            if (abi == "aapcs" || abi == "darwinpcs") {
                cc::compiler_error("Only Darwin supported for aarch64");
            }

            auto abi_kind = aarch64_abi_info::abi_kind::darwin_pcs;
            return target_info_ptr(new aarch64_target_info(types, abi_kind));
        };

        auto x86_64_info = [&] {
            auto os = triple.getOS();
            if (os == llvm::Triple::Linux) {
                return target_info_ptr(
                    new x86_64_target_info(types, avx_level(target))
                );
            }

            if (os == llvm::Triple::Darwin) {
                return target_info_ptr(
                    new darwin_x86_64_target_info(types, avx_level(target))
                );
            }

            throw cc::compiler_error("Unsupported x86_64 OS type.");
        };

        switch (triple.getArch()) {
            case llvm::Triple::aarch64: return aarch64_info();
            case llvm::Triple::x86_64:  return x86_64_info();
            default: throw cc::compiler_error("Target not yet supported.");
        }
    }

    const target_info_t &codegen_module::get_target_info() {
        if (!target_info) {
            target_info = get_target_info_impl(target, types);
        }

        return *target_info;
    }

    void codegen_module::add_replacement(string_ref name, mlir::Operation *op) {
        replacements[name] = op;
    }

    void codegen_module::apply_replacements() {
        if (!replacements.empty()) {
            throw cc::compiler_error("unsupported function replacement in module release");
        }
    }

    vast::hl::FuncOp codegen_module::get_addr_of_function(
        clang::GlobalDecl /* decl */, mlir_type /* ty */,
        bool /* for_vtable */, bool /* dontdefer */,
        global_emit /* is_for_definition */
    ) {
        throw cc::compiler_error("get_addr_of_function not implemented");
    }

    mlir::Operation *codegen_module::get_addr_of_function(
        clang::GlobalDecl /* decl */, global_emit /* is_for_definition */
    ) {
        throw cc::compiler_error("get_addr_of_function not implemented");
    }

    void codegen_module::update_completed_type(const clang::TagDecl */* decl */) {
        throw cc::compiler_error("update_completed_type not implemented");
    }

    bool codegen_module::should_emit_function(clang::GlobalDecl /* decl */) {
        // TODO: implement this -- requires defining linkage for vast
        return true;
    }

    void codegen_module::build_global_definition(clang::GlobalDecl glob, mlir::Operation *op) {
        const auto *decl = llvm::cast< clang::ValueDecl >(glob.getDecl());

        if (const auto *fn = llvm::dyn_cast< clang::FunctionDecl >(decl)) {
            // At -O0, don't generate vast for functions with available_externally linkage.
            if (!should_emit_function(glob))
                return;

            if (fn->isMultiVersion()) {
                throw cc::compiler_error("codegen for multi version function not implemented");
            }

            if (const auto *method = llvm::dyn_cast< clang::CXXMethodDecl >(decl)) {
                throw cc::compiler_error("methods not implemented");
            }

            return build_global_function_definition(glob, op);
        }

        if (const auto *var = llvm::dyn_cast< clang::VarDecl >(decl))
            return build_global_var_definition(var, !var->hasDefinition());

        llvm_unreachable("Invalid argument to buildGlobalDefinition()");

    }

    void codegen_module::build_global_function_definition(clang::GlobalDecl decl, mlir::Operation *op) {
        // auto const *fn_decl = llvm::cast< clang::FunctionDecl >(decl.getDecl());

        // Compute the function info and vast type.
        const auto &fty_info = types.arrange_global_decl(decl);
        auto ty = types.get_function_type(fty_info);

        // Get or create the prototype for the function.
        // if (!V || (V.getValueType() != Ty))
        // TODO: Figure out what to do here? llvm uses a GlobalValue for the FuncOp in mlir
        op = get_addr_of_function(
            decl, ty, /*ForVTable=*/false, /*DontDefer=*/true,
            global_emit::definition
        );

        auto fn = mlir::cast< vast::hl::FuncOp >(op);
        // Already emitted.
        if (!fn.isDeclaration()) {
            return;
        }

        // setFunctionLinkage(GD, Fn);
        // // TODO setGVProperties
        // // TODO MaubeHandleStaticInExternC
        // // TODO maybeSetTrivialComdat
        // // TODO setLLVMFunctionFEnvAttributes

        // CIRGenFunction CGF{*this, builder};
        // CurCGF = &CGF;
        // {
        //     mlir::OpBuilder::InsertionGuard guard(builder);
        //     CGF.generateCode(GD, Fn, FI);
        // }
        // CurCGF = nullptr;

        // // TODO: setNonAliasAttributes
        // // TODO: SetLLVMFunctionAttributesForDeclaration

        // assert(!D->getAttr<ConstructorAttr>() && "not implemented");
        // assert(!D->getAttr<DestructorAttr>() && "not implemented");
        // assert(!D->getAttr<AnnotateAttr>() && "not implemented");
    }

    void codegen_module::build_global_var_definition(const clang::VarDecl */* decl */, bool /* tentative */) {
        throw cc::compiler_error("build_global_var_definition not implemented");
    }

    void codegen_module::build_global_decl(clang::GlobalDecl &/* decl */) {
        throw cc::compiler_error("build_global_decl not implemented");
    }

    bool codegen_module::must_be_emitted(const clang::ValueDecl *glob) {
        // Never defer when EmitAllDecls is specified.
        assert(!lang_opts.EmitAllDecls && "EmitAllDecls not implemented");
        assert(!codegen_opts.KeepStaticConsts && "KeepStaticConsts not implemented");

        return actx.DeclMustBeEmitted(glob);
    }

    bool codegen_module::may_be_emitted_eagerly(const clang::ValueDecl *glob) {
        assert(!lang_opts.OpenMP && "not supported");

        if (const auto *fn = llvm::dyn_cast< clang::FunctionDecl >(glob)) {
            // Implicit template instantiations may change linkage if they are later
            // explicitly instantiated, so they should not be emitted eagerly.
            constexpr auto implicit = clang::TSK_ImplicitInstantiation;
            assert(fn->getTemplateSpecializationKind() != implicit && "not implemented");
            assert(!fn->isTemplated() && "templates not implemented");
            return true;
        }


        if (const auto *vr = llvm::dyn_cast< clang::VarDecl >(glob)) {
            // A definition of an inline constexpr static data member may change
            // linkage later if it's redeclared outside the class.
            constexpr auto weak_unknown = clang::ASTContext::InlineVariableDefinitionKind::WeakUnknown;
            assert(actx.getInlineVariableDefinitionKind(vr) != weak_unknown && "not implemented");
            return true;
        }

        throw cc::compiler_error("unsupported value decl");
    }

    void codegen_module::build_global(clang::GlobalDecl decl) {
        // builder.
        const auto *glob = llvm::cast< clang::ValueDecl >(decl.getDecl());

        assert(!glob->hasAttr< clang::WeakRefAttr >() && "not implemented");
        assert(!glob->hasAttr< clang::AliasAttr >() && "not implemented");
        assert(!glob->hasAttr< clang::IFuncAttr >() && "not implemented");
        assert(!glob->hasAttr< clang::CPUDispatchAttr >() && "not implemented");
        assert(!lang_opts.CUDA && "not implemented");
        assert(!lang_opts.OpenMP && "not implemented");

        // Ignore declarations, they will be emitted on their first use.
        if (const auto *fn = llvm::dyn_cast< clang::FunctionDecl >(glob)) {
            // Forward declarations are emitted lazily on first use.
            if (!fn->doesThisDeclarationHaveABody()) {
                if (!fn->doesDeclarationForceExternallyVisibleDefinition())
                    return;
                throw cc::compiler_error("build_global FunctionDecl not implemented");
                // auto mangled_name = getMangledName(decl);

                // // Compute the function info and CIR type.
                // const auto &FI = getTypes().arrangeGlobalDeclaration(decl);
                // mlir_type Ty = getTypes().GetFunctionType(FI);

                // GetOrCreateCIRFunction(MangledName, Ty, decl, /*ForVTable=*/false,
                //                         /*DontDefer=*/false);
                // return;
            }
        } else {
            throw cc::compiler_error("build_global VarDecl not implemented");

            // const auto *var = llvm::cast< clang::VarDecl>(glob);
            // assert(var->isFileVarDecl() && "Cannot emit local var decl as global.");
            // if (var->isThisDeclarationADefinition() != VarDecl::Definition &&
            //     !astCtx.isMSStaticDataMemberInlineDefinition(var)) {
            //     assert(!getLangOpts().OpenMP && "not implemented");
            //     // If this declaration may have caused an inline variable definition
            //     // to change linkage, make sure that it's emitted.
            //     // TODO probably use GetAddrOfGlobalVar(var) below?
            //     assert((astCtx.getInlineVariableDefinitionKind(var) !=
            //             ASTContext::InlineVariableDefinitionKind::Strong) &&
            //             "not implemented");
            //     return;
            // }
        }

        // Defer code generation to first use when possible, e.g. if this is an inline
        // function. If the global mjust always be emitted, do it eagerly if possible
        // to benefit from cache locality.
        if (must_be_emitted(glob) && may_be_emitted_eagerly(glob)) {
            // Emit the definition if it can't be deferred.
            return build_global_definition(glob);
        }

        // // If we're deferring emission of a C++ variable with an initializer, remember
        // // the order in which it appeared on the file.
        // if (getLangOpts().CPlusPlus && isa<VarDecl>(glob) &&
        //     cast<VarDecl>(glob)->hasInit()) {
        //     DelayedCXXInitPosition[glob] = CXXGlobalInits.size();
        //     CXXGlobalInits.push_back(nullptr);
        // }

        // llvm::StringRef MangledName = getMangledName(GD);
        // if (getGlobalValue(MangledName) != nullptr) {
        //     // The value has already been used and should therefore be emitted.
        //     addDeferredDeclToEmit(GD);
        // } else if (MustBeEmitted(glob)) {
        //     // The value must be emitted, but cannot be emitted eagerly.
        //     assert(!MayBeEmittedEagerly(glob));
        //     addDeferredDeclToEmit(GD);
        // } else {
        //     // Otherwise, remember that we saw a deferred decl with this name. The first
        //     // use of the mangled name will cause it to move into DeferredDeclsToEmit.
        //     DeferredDecls[MangledName] = GD;
        // }
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
                if (llvm::isa< clang::DecompositionDecl >(decl)) {
                    throw cc::compiler_error("codegen for DecompositionDecl not implemented");
                }
                break;
            }
            case clang::Decl::CXXMethod:
            case clang::Decl::Function: {
                build_global(llvm::cast< clang::FunctionDecl >(decl));
                if (codegen_opts.CoverageMapping) {
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
