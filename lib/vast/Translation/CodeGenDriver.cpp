// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Translation/CodeGenDriver.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/Basic/TargetInfo.h>
VAST_UNRELAX_WARNINGS

#include "vast/Translation/Error.hpp"

// FIXME: get rid of dependency from upper layer
#include "vast/CodeGen/TypeInfo.hpp"

namespace vast::cg
{
    defer_handle_of_top_level_decl::defer_handle_of_top_level_decl(
        codegen_driver &codegen, bool emit_deferred
    )
        : codegen(codegen), emit_deferred(emit_deferred)
    {
        ++codegen.deffered_top_level_decls;
    }

    defer_handle_of_top_level_decl::~defer_handle_of_top_level_decl() {
        unsigned level = --codegen.deffered_top_level_decls;
        if (level == 0 && emit_deferred) {
            codegen.build_deferred_decls();
        }
    }

    void codegen_driver::finalize() {
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
            throw cg::unimplemented("cuda module release");
        }
        // TODO: OpenMPRuntime
        // TODO: PGOReader
        // TODO: buildCtorList(GlobalCtors);
        // TODO: builtCtorList(GlobalDtors);
        // TODO: buildGlobalAnnotations();
        // TODO: buildDeferredUnusedCoverageMappings();
        // TODO: CIRGenPGO
        // TODO: CoverageMapping
        // TODO: if (codegen_opts.SanitizeCfiCrossDso) {
        // TODO: buildAtAvailableLinkGuard();
        const auto &target_triplet = actx.getTargetInfo().getTriple();
        if (target_triplet.isWasm() && !target_triplet.isOSEmscripten()) {
            throw cg::unimplemented("WASM module release");
        }

        // Emit reference of __amdgpu_device_library_preserve_asan_functions to
        // preserve ASAN functions in bitcode libraries.
        if (lang().Sanitize.has(clang::SanitizerKind::Address)) {
            throw cg::unimplemented("AddressSanitizer module release");
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

    void codegen_driver::handle_translation_unit(acontext_t &/* acontext */) {
        finalize();
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
                    throw cg::unimplemented("codegen for DecompositionDecl");
                }
                break;
            }
            case clang::Decl::CXXMethod:
            case clang::Decl::Function: {
                build_global(llvm::cast< clang::FunctionDecl >(decl));
                if (options.coverage_mapping) {
                    throw cg::unimplemented("coverage mapping not supported");
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
            default: throw cg::unimplemented(std::string("codegen for '") + decl->getDeclKindName());
            (void) mctx;
        }
    }

    function_processing_lock codegen_driver::make_lock(const function_info_t *fninfo) {
        return function_processing_lock(type_conv, fninfo);
    }

    owning_module_ref codegen_driver::freeze() { return codegen.freeze(); }

    vast::hl::FuncOp codegen_driver::get_addr_of_function(
        clang::GlobalDecl /* decl */, mlir_type /* ty */,
        bool /* for_vtable */, bool /* dontdefer */,
        global_emit /* is_for_definition */
    ) {
        throw cg::unimplemented("get_addr_of_function");
    }

    mlir::Operation *codegen_driver::get_addr_of_function(
        clang::GlobalDecl /* decl */, global_emit /* is_for_definition */
    ) {
        throw cg::unimplemented("get_addr_of_function");
    }

    void codegen_driver::build_global_definition(clang::GlobalDecl glob, mlir::Operation *op) {
        const auto *decl = llvm::cast< clang::ValueDecl >(glob.getDecl());

        if (const auto *fn = llvm::dyn_cast< clang::FunctionDecl >(decl)) {
            // At -O0, don't generate vast for functions with available_externally linkage.
            if (!should_emit_function(glob))
                return;

            if (fn->isMultiVersion()) {
                throw cg::unimplemented("codegen for multi version function");
            }

            if (const auto *method = llvm::dyn_cast< clang::CXXMethodDecl >(decl)) {
                throw cg::unimplemented("cxx methods");
            }

            return build_global_function_definition(glob, op);
        }

        if (const auto *var = llvm::dyn_cast< clang::VarDecl >(decl))
            return build_global_var_definition(var, !var->hasDefinition());

        llvm_unreachable("Invalid argument to buildGlobalDefinition()");

    }

    void codegen_driver::build_global_function_definition(clang::GlobalDecl decl, mlir::Operation *op) {
        // auto const *fn_decl = llvm::cast< clang::FunctionDecl >(decl.getDecl());

        // Compute the function info and vast type.
        const auto &fty_info = type_info.arrange_global_decl(decl);
        /* auto ty = */ type_conv.get_function_type(fty_info);
        throw cg::unimplemented("build_global_function_definition type emition ");

        // Get or create the prototype for the function.
        // if (!V || (V.getValueType() != Ty))
        // TODO: Figure out what to do here? llvm uses a GlobalValue for the FuncOp in mlir
        // op = get_addr_of_function(
        //     decl, ty, /*ForVTable=*/false, /*DontDefer=*/true,
        //     global_emit::definition
        // );

        // auto fn = mlir::cast< vast::hl::FuncOp >(op);
        // // Already emitted.
        // if (!fn.isDeclaration()) {
        //     return;
        // }

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

    bool codegen_driver::should_emit_function(clang::GlobalDecl /* decl */) {
        // TODO: implement this -- requires defining linkage for vast
        return true;
    }

    void codegen_driver::build_global_var_definition(const clang::VarDecl */* decl */, bool /* tentative */) {
        throw cg::unimplemented("build_global_var_definition");
    }

    void codegen_driver::build_global_decl(clang::GlobalDecl &/* decl */) {
        throw cg::unimplemented("build_global_decl");
    }

    void codegen_driver::build_global(clang::GlobalDecl decl) {
        const auto *glob = llvm::cast< clang::ValueDecl >(decl.getDecl());

        assert(!glob->hasAttr< clang::WeakRefAttr >() && "not implemented");
        assert(!glob->hasAttr< clang::AliasAttr >() && "not implemented");
        assert(!glob->hasAttr< clang::IFuncAttr >() && "not implemented");
        assert(!glob->hasAttr< clang::CPUDispatchAttr >() && "not implemented");

        assert(!lang().CUDA && "not implemented");
        assert(!lang().OpenMP && "not implemented");

        // Ignore declarations, they will be emitted on their first use.
        if (const auto *fn = llvm::dyn_cast< clang::FunctionDecl >(glob)) {
            // Forward declarations are emitted lazily on first use.
            if (!fn->doesThisDeclarationHaveABody()) {
                if (!fn->doesDeclarationForceExternallyVisibleDefinition())
                    return;
                throw cg::unimplemented("build_global FunctionDecl");
                // auto mangled_name = getMangledName(decl);

                // // Compute the function info and CIR type.
                // const auto &FI = getTypes().arrangeGlobalDeclaration(decl);
                // mlir_type Ty = getTypes().GetFunctionType(FI);

                // GetOrCreateFunction(MangledName, Ty, decl, /*ForVTable=*/false,
                //                         /*DontDefer=*/false);
                // return;
            }
        } else {
            throw cg::unimplemented("build_global VarDecl");

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

    bool codegen_driver::must_be_emitted(const clang::ValueDecl *glob) {
        // Never defer when EmitAllDecls is specified.
        assert(!lang().EmitAllDecls && "EmitAllDecls not implemented");
        assert(!options.keep_static_consts && "KeepStaticConsts not implemented");

        return actx.DeclMustBeEmitted(glob);
    }

    bool codegen_driver::may_be_emitted_eagerly(const clang::ValueDecl *glob) {
        assert(!lang().OpenMP && "not supported");

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

        throw cg::unimplemented("unsupported value decl");
    }

    void codegen_driver::build_default_methods() {
        // Differently from deferred_decls_to_emit, there's no recurrent use of
        // deferred_decls_to_emit, so use it directly for emission.
        for (auto &decl : default_methods_to_emit) {
            build_global_decl(decl);
        }
    }

    void codegen_driver::build_deferred() {
        // Emit deferred declare target declarations
        if (lang().OpenMP && !lang().OpenMPSimd) {
            throw cg::unimplemented("build_deferred for openmp");
        }

        // Emit code for any potentially referenced deferred decls. Since a previously
        // unused static decl may become used during the generation of code for a
        // static function, iterate until no changes are made.
        if (!deferred_vtables.empty()) {
            throw cg::unimplemented("build_deferred for vtables");
        }

        // Emit CUDA/HIP static device variables referenced by host code only. Note we
        // should not clear CUDADeviceVarODRUsedByHost since it is still needed for
        // further handling.
        if (lang().CUDA && lang().CUDAIsDevice) {
            throw cg::unimplemented("build_deferred for cuda");
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

    void codegen_driver::add_replacement(string_ref name, mlir::Operation *op) {
        replacements[name] = op;
    }

    void codegen_driver::apply_replacements() {
        if (!replacements.empty()) {
            throw cg::unimplemented(" function replacement in module release");
        }
    }

} // namespace vast::cg
