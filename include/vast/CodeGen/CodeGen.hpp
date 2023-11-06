// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/Frontend/ASTUnit.h>
#include <mlir/IR/Verifier.h>
#include <mlir/InitAllDialects.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/DefaultVisitor.hpp"
#include "vast/CodeGen/UnsupportedVisitor.hpp"
#include "vast/CodeGen/UnreachableVisitor.hpp"
#include "vast/CodeGen/FallBackVisitor.hpp"

#include "vast/Dialect/Dialects.hpp"
#include "vast/CodeGen/DataLayout.hpp"

#include "vast/Frontend/Options.hpp"

namespace vast::cg
{
    template< typename From, typename Symbol >
    using scoped_symbol_table = llvm::ScopedHashTableScope< From, Symbol >;

    using typedefs_scope   = scoped_symbol_table< const clang::TypedefDecl *, hl::TypeDefOp >;
    using typedecls_scope  = scoped_symbol_table< const clang::TypeDecl *, hl::TypeDeclOp >;
    using enumdecls_scope  = scoped_symbol_table< const clang::EnumDecl *, hl::EnumDeclOp >;
    using enumconsts_scope = scoped_symbol_table< const clang::EnumConstantDecl *, hl::EnumConstantOp >;
    using lables_scope     = scoped_symbol_table< const clang::LabelDecl*, hl::LabelDeclOp >;
    using functions_scope  = scoped_symbol_table< mangled_name_ref, hl::FuncOp >;
    using vars_scope       = scoped_symbol_table< const clang::VarDecl *, Value >;

    struct scope_t {
        typedefs_scope   typedefs;
        typedecls_scope  typedecls;
        enumdecls_scope  enumdecls;
        enumconsts_scope enumconsts;
        lables_scope     labels;
        functions_scope  funcdecls;
        vars_scope       globs;
    };

    template< typename derived_t >
    using default_visitor_stack = fallback_visitor< derived_t,
        default_visitor, unsup_visitor, unreach_visitor
    >;

    //
    // CodeGen
    //
    template<
        typename context_t,
        template< typename > typename visitor_config,
        typename meta_generator
    >
    struct codegen_instance
    {
        using visitor_t = visitor_instance< context_t, visitor_config, meta_generator >;
        using var_table = typename context_t::var_table;

        codegen_instance(context_t &cgctx)
            : cgctx(cgctx), meta(&cgctx.actx, &cgctx.mctx)
        {
            mlir::registerAllDialects(cgctx.mctx);
            vast::registerAllDialects(cgctx.mctx);
            cgctx.mctx.loadAllAvailableDialects();

            scope = std::unique_ptr< scope_t >( new scope_t{
                .typedefs   = cgctx.typedefs,
                .typedecls  = cgctx.typedecls,
                .enumdecls  = cgctx.enumdecls,
                .enumconsts = cgctx.enumconsts,
                .labels     = cgctx.labels,
                .funcdecls  = cgctx.funcdecls,
                .globs      = cgctx.vars
            });

            visitor = std::make_unique< visitor_t >(
                cgctx, meta
            );
        }

        void emit_data_layout() {
            hl::emit_data_layout(cgctx.mctx, cgctx.mod, cgctx.data_layout());
        }

        bool verify_module() const {
            return mlir::verify(cgctx.mod.get()).succeeded();
        }

        template< typename Op, typename... Args >
        auto make(Args &&...args) {
            return visitor->template create< Op >(std::forward< Args >(args)...);
        }

        // FIXME: Remove from driver whan we obliterate function type convertion
        // from driver
        mlir_type convert(qual_type type) {
            return visitor->Visit(type);
        }
        mlir_type make_lvalue(mlir_type type) {
            if (type.isa< hl::LValueType >()) {
                return type;
            }
            return hl::LValueType::get(&cgctx.mctx, type);
        }

        void update_completed_type(clang::TagDecl *decl) {
            VAST_UNIMPLEMENTED;
        }

        operation build_function_prototype(clang::GlobalDecl decl, mlir_type fty) {
            return visitor->build_function_prototype(decl, fty);
        }

        // FIXME: should be part of scope
        operation get_global_value(mangled_name_ref name) {
            return cgctx.get_global_value(name);
        }

        // FIXME: should be part of scope
        mlir_value get_global_value(const clang::Decl *decl) {
            return cgctx.get_global_value(decl);
        }

        // FIXME: should be part of scope
        void add_deferred_decl_to_emit(clang::GlobalDecl decl) {
            cgctx.add_deferred_decl_to_emit(decl);
        }

        // FIXME: should be part of scope
        const auto& default_methods_to_emit() const {
            return cgctx.default_methods_to_emit;
        }

        // FIXME: should be part of scope
        const auto& deferred_decls_to_emit() const {
            return cgctx.deferred_decls_to_emit;
        }

        // FIXME: should be part of scope
        const auto& deferred_vtables() const {
            return cgctx.deferred_vtables;
        }

        // FIXME: should be part of scope
        void set_deferred_decl(mangled_name_ref name, clang::GlobalDecl decl) {
            cgctx.deferred_decls[name] = decl;
        }

        // FIXME: should be part of scope
        const auto& deferred_decls() const {
            return cgctx.deferred_decls;
        }

        // FIXME: should be part of scope
        auto receive_deferred_decls_to_emit() {
            std::vector< clang::GlobalDecl > current;
            current.swap(cgctx.deferred_decls_to_emit);
            return current;
        }

        // FIXME: should be part of scope
        mangled_name_ref get_mangled_name(clang::GlobalDecl decl) {
            return cgctx.get_mangled_name(decl);
        }

        auto insert_at_end(hl::FuncOp fn) {
            auto guard = visitor->insertion_guard();
            visitor->set_insertion_point_to_end(&fn.getBody());
            return std::move(guard);
        }

        // TODO: This is currently just a dumb stub. But we want to be able to clearly
        // assert where we arn't doing things that we know we should and will crash
        // as soon as we add a DebugInfo type to this class.
        std::nullptr_t *get_debug_info() { return nullptr; }

        void start_function(
            clang::GlobalDecl glob,
            hl::FuncOp fn,
            const function_info_t &fty_info,
            const function_arg_list &args,
            loc_t loc,
            const cc::action_options &opts
        ) {
            const auto *decl = glob.getDecl();
            const auto *function_decl = clang::dyn_cast_or_null< clang::FunctionDecl >(decl);
            if (function_decl && function_decl->usesSEHTry()) {
                VAST_UNIMPLEMENTED;
            }

            const auto &lang = cgctx.actx.getLangOpts();

            // auto curr_function_decl = decl ? decl->getNonClosureContext() : nullptr;

            // TODO: Sanitizers
            // TODO: XRay
            // TODO: PGO

            unsigned entry_count = 0, entry_offset = 0;
            if (const auto *attr = decl ? decl->getAttr< clang::PatchableFunctionEntryAttr >() : nullptr) {
                VAST_UNIMPLEMENTED;
            } else {
                entry_count  = opts.codegen.PatchableFunctionEntryCount;
                entry_offset = opts.codegen.PatchableFunctionEntryOffset;
            }

            if (entry_count && entry_offset <= entry_count) {
                VAST_UNIMPLEMENTED;
            }

            // Add no-jump-tables value.
            if (opts.codegen.NoUseJumpTables) {

                VAST_UNIMPLEMENTED;
            }

            // Add no-inline-line-tables value.
            if (opts.codegen.NoInlineLineTables) {
                VAST_UNIMPLEMENTED;
            }

            // TODO: Add profile-sample-accurate value.
            if (opts.codegen.ProfileSampleAccurate) {
                VAST_UNIMPLEMENTED;
            }

            if (decl && decl->hasAttr< clang::CFICanonicalJumpTableAttr >()) {
                VAST_UNIMPLEMENTED;
            }

            if (decl && decl->hasAttr< clang::NoProfileFunctionAttr >()) {
                VAST_UNIMPLEMENTED;
            }

            if (function_decl && lang.OpenCL) {
                VAST_UNIMPLEMENTED;
            }

            // If we are checking function types, emit a function type signature as
            // prologue data.
            // if (function_decl && lang.CPlusPlus && SanOpts.has(SanitizerKind::Function)) {
            //     VAST_UNIMPLEMENTED;
            // }

            // If we're checking nullability, we need to know whether we can check the
            // return value. Initialize the flag to 'true' and refine it in
            // buildParmDecl.
            // if (SanOpts.has(SanitizerKind::NullabilityReturn)) {
            //     VAST_UNIMPLEMENTED;
            // }

            // If we're in C++ mode and the function name is "main", it is guaranteed to
            // be norecurse by the standard (3.6.1.3 "The function main shall not be
            // used within a program").
            //
            // OpenCL C 2.0 v2.2-11 s6.9.i:
            //     Recursion is not supported.
            //
            // SYCL v1.2.1 s3.10:
            //     kernels cannot include RTTI information, exception cases, recursive
            //     code, virtual functions or make use of C++ libraries that are not
            //     compiled for the device.
            auto norecurse = [&] () -> bool {
                return function_decl
                    && ((lang.CPlusPlus && function_decl->isMain())
                    || lang.OpenCL
                    || lang.SYCLIsDevice
                    || (lang.CUDA && function_decl->hasAttr< clang::CUDAGlobalAttr >()));
            };

            if (norecurse()) {
                ; // TODO: support norecurse attr
            }

            // TODO: fp rounding and exception behavior

            // TODO: stackrealign attr

            auto &entry_block = fn.getBlocks().front();

            // TODO: allocapt insertion? probably don't need for VAST

            // TODO: return value checking

            if (get_debug_info()) {
                VAST_UNIMPLEMENTED;
            }

            // if (ShouldInstrumentFunction()) {
            //     VAST_UNIMPLEMENTED;
            // }

            // Since emitting the mcount call here impacts optimizations such as
            // function inlining, we just add an attribute to insert a mcount call in
            // backend. The attribute "counting-function" is set to mcount function name
            // which is architecture dependent.
            // if (options.InstrumentForProfiling) {
            //     VAST_UNIMPLEMENTED;
            // }

            if (opts.codegen.PackedStack) {
                VAST_UNIMPLEMENTED;
            }

            if (opts.codegen.WarnStackSize != UINT_MAX) {
                VAST_UNIMPLEMENTED;
            }

            // TODO: emitstartehspec

            // TODO: prologuecleanupdepth

            if (lang.OpenMP && decl) {
                VAST_UNIMPLEMENTED;
            }

            // TODO: build_function_prolog

            {
                // Set the insertion point in the builder to the beginning of the
                // function body, it will be used throughout the codegen to create
                // operations in this function.

                // TODO: this should live in `build_function_prolog`
                // Declare all the function arguments in the symbol table.
                for (const auto [ast_param, mlir_param] : llvm::zip(args, entry_block.getArguments())) {
                    // TODO set alignment
                    // TODO set name
                    mlir_param.setLoc(meta.location(ast_param));
                    declare(ast_param, mlir_value(mlir_param));
                }

            }

            if (decl && clang::isa< clang::CXXMethodDecl>(decl) &&
                clang::cast< clang::CXXMethodDecl>(decl)->isInstance()
            ) {
                VAST_UNIMPLEMENTED_MSG( "emit prologue of cxx methods" );
            }

            // If any of the arguments have a variably modified type, make sure to emit
            // the type size.
            for (auto arg : args) {
                const clang::VarDecl *var_decl = arg;

                // Dig out the type as written from ParmVarDecls; it's unclear whether the
                // standard (C99 6.9.1p10) requires this, but we're following the
                // precedent set by gcc.
                auto type = [&] {
                    if (const auto *parm_var_decl = dyn_cast< clang::ParmVarDecl >(var_decl)) {
                        return parm_var_decl->getOriginalType();
                    }
                    return var_decl->getType();
                } ();

                if (type->isVariablyModifiedType()) {
                    VAST_UNIMPLEMENTED;
                }
            }

            // Emit a location at the end of the prologue.
            if (get_debug_info()) {
                VAST_UNIMPLEMENTED;
            }

            // TODO: Do we need to handle this in two places like we do with
            // target-features/target-cpu?
            if (const auto *vec_width = function_decl->getAttr< clang::MinVectorWidthAttr >()) {
                VAST_UNIMPLEMENTED;
            }
        }

        logical_result build_compound_stmt_without_scope(const clang::CompoundStmt &stmt) {
            for (auto *curr : stmt.body()) {
                if (build_stmt(curr, /* use current scope */ false).failed()) {
                    return mlir::failure();
                }
            }

            return mlir::success();
        }

        logical_result build_stmt(const clang::Stmt *stmt, bool /* use_current_scope */) {
            // FIXME: consolidate with clang codegene
            visitor->Visit(stmt);
            return mlir::success();
        }

        logical_result build_function_body(const clang::Stmt *body) {
            // TODO: incrementProfileCounter(Body);

            // We start with function level scope for variables.
            llvm::ScopedHashTableScope var_scope(cgctx.vars);

            auto result = logical_result::success();
            if (const auto stmt = clang::dyn_cast< clang::CompoundStmt >(body)) {
                result = build_compound_stmt_without_scope(*stmt);
            } else {
                result = build_stmt(body, /* use current scope */ true);
            }

            // This is checked after emitting the function body so we know if there are
            // any permitted infinite loops.
            // TODO: if (checkIfFunctionMustProgress())
            // CurFn->addFnAttr(llvm::Attribute::MustProgress);
            return result;
        }

        hl::FuncOp emit_function_prologue(
            hl::FuncOp fn, clang::GlobalDecl decl,  const function_info_t &fty_info,
            function_arg_list args, const cc::action_options &options
        ) {
            VAST_CHECK(fn, "generating code for a null function");
            const auto function_decl = clang::cast< clang::FunctionDecl >(decl.getDecl());

            auto guard = visitor->insertion_guard();

            if (function_decl->isInlineBuiltinDeclaration()) {
                VAST_UNIMPLEMENTED_MSG("emit body of inline builtin declaration");
            } else {
                // Detect the unusual situation where an inline version is shadowed by a
                // non-inline version. In that case we should pick the external one
                // everywhere. That's GCC behavior too. Unfortunately, I cannot find a way
                // to detect that situation before we reach codegen, so do some late
                // replacement.
                for (const auto *prev = function_decl->getPreviousDecl(); prev; prev = prev->getPreviousDecl()) {
                    if (LLVM_UNLIKELY(prev->isInlineBuiltinDeclaration())) {
                        VAST_UNIMPLEMENTED_MSG("emit body of inline builtin declaration");
                    }
                }
            }

            // Check if we should generate debug info for this function.
            if (function_decl->hasAttr< clang::NoDebugAttr >()) {
                VAST_UNIMPLEMENTED_MSG("emit no debug meta");
            }

            // The function might not have a body if we're generating thunks for a
            // function declaration.
            // FIXME: use meta location instead
            // auto body_range = [&] () -> clang::SourceRange {
            //     if (auto *body = function_decl->getBody())
            //         return body->getSourceRange();
            //     else
            //         return function_decl->getLocation();
            // } ();

            // TODO: CurEHLocation

            // Use the location of the start of the function to determine where the
            // function definition is located. By default we use the location of the
            // declaration as the location for the subprogram. A function may lack a
            // declaration in the source code if it is created by code gen. (examples:
            // _GLOBAL__I_a, __cxx_global_array_dtor, thunk).
            auto loc = meta.location(function_decl);

            // If this is a function specialization then use the pattern body as the
            // location for the function.
            if (const auto *spec = function_decl->getTemplateInstantiationPattern()) {
                if (spec->hasBody(spec)) {
                    loc = meta.location(spec);
                }
            }

            // FIXME: maybe move to codegen visitor
            if (auto body = function_decl->getBody()) {
                // LLVM codegen: Coroutines always emit lifetime markers
                // Hide this under request for lifetime emission so that we can write
                // tests when the time comes, but VAST should be intrinsically scope
                // accurate, so no need to tie coroutines to such markers.
                if (clang::isa< clang::CoroutineBodyStmt >(body)) {
                    VAST_UNIMPLEMENTED_MSG("emit lifetime markers");
                }
            }

            // Create a scope in the symbol table to hold variable declarations.
            llvm::ScopedHashTableScope var_scope(cgctx.vars);
            {
                auto body = function_decl->getBody();
                auto begin_loc = meta.location(body);
                auto end_loc = meta.location(body);

                VAST_CHECK(fn.isDeclaration(), "Function already has body?");
                auto *entry_block = fn.addEntryBlock();
                visitor->set_insertion_point_to_start(entry_block);

                lexical_scope_context lex_ccope{begin_loc, end_loc, entry_block};
                lexical_scope_guard scope_guard{*this, &lex_ccope};

                // Emit the standard function prologue.
                start_function(decl, fn, fty_info, args, loc, options);

                for (const auto lab : filter< clang::LabelDecl >(function_decl->decls())) {
                    visitor->Visit(lab);
                }

                // Initialize lexical scope information.

                // Save parameters for coroutine function.
                if (body && clang::isa_and_nonnull< clang::CoroutineBodyStmt >(body)) {
                    VAST_UNIMPLEMENTED_MSG("coroutine parameters");
                }

                // Generate the body of the function.
                // TODO: PGO.assignRegionCounters

                const auto &lang = cgctx.actx.getLangOpts();

                if (clang::isa< clang::CXXDestructorDecl >(function_decl)) {
                    VAST_UNIMPLEMENTED;
                } else if (clang::isa< clang::CXXConstructorDecl >(function_decl)) {
                    VAST_UNIMPLEMENTED;
                } else if (lang.CUDA && !lang.CUDAIsDevice && function_decl->hasAttr< clang::CUDAGlobalAttr >()) {
                    VAST_UNIMPLEMENTED;
                } else if (auto method = clang::dyn_cast< clang::CXXMethodDecl >(function_decl); method && method->isLambdaStaticInvoker()) {
                    VAST_UNIMPLEMENTED;
                } else if (function_decl->isDefaulted() && clang::isa< clang::CXXMethodDecl >(function_decl) &&
                    (clang::cast< clang::CXXMethodDecl >(function_decl)->isCopyAssignmentOperator() ||
                     clang::cast< clang::CXXMethodDecl >(function_decl)->isMoveAssignmentOperator())
                ) {
                    VAST_UNIMPLEMENTED;
                } else if (body) {
                    if (mlir::failed(build_function_body(body))) {
                        VAST_UNREACHABLE("failed function body codegen");
                    }
                } else {
                    VAST_UNIMPLEMENTED_MSG("no definition for emitted function");
                }
            }

            return fn;
        }

        void emit_implicit_return_zero(hl::FuncOp fn, const clang::FunctionDecl *decl) {
            auto guard = insert_at_end(fn);
            auto loc   = meta.location(decl);

            auto fty = fn.getFunctionType();
            auto zero = visitor->constant(loc, fty.getResult(0), apsint(0));
            make< core::ImplicitReturnOp >(loc, zero);
        }

        void emit_implicit_void_return(hl::FuncOp fn, const clang::FunctionDecl *decl) {
            VAST_CHECK( decl->getReturnType()->isVoidType(),
                "Can't emit implicit void return in non-void function."
            );

            auto guard = insert_at_end(fn);

            auto loc = meta.location(decl);
            make< core::ImplicitReturnOp >(loc, visitor->constant(loc));
        }

        void emit_trap(hl::FuncOp fn, const clang::FunctionDecl *decl) {
            // TODO fix when we support builtin function (emit enreachable for now)
            emit_unreachable(fn, decl);
        }

        void emit_unreachable(hl::FuncOp fn, const clang::FunctionDecl *decl) {
            auto guard = insert_at_end(fn);
            auto loc = meta.location(decl);
            make< hl::UnreachableOp >(loc);
        }

        hl::FuncOp declare(const clang::FunctionDecl *decl, auto vast_decl_builder) {
            return cgctx.declare(decl, vast_decl_builder);
        }

        mlir_value declare(const clang::VarDecl *decl, mlir_value vast_value) {
            return cgctx.declare(decl, vast_value);
        }

        mlir_value declare(const clang::VarDecl *decl, auto vast_decl_builder) {
            return cgctx.declare(decl, vast_decl_builder);
        }

        hl::LabelDeclOp declare(const clang::LabelDecl *decl, auto vast_decl_builder) {
            return cgctx.declare(decl, vast_decl_builder);
        }

        hl::TypeDefOp declare(const clang::TypedefDecl *decl, auto vast_decl_builder) {
            return cgctx.declare(decl, vast_decl_builder);
        }

        hl::TypeDeclOp declare(const clang::TypeDecl *decl, auto vast_decl_builder) {
            return cgctx.declare(decl, vast_decl_builder);
        }

        hl::EnumDeclOp declare(const clang::EnumDecl *decl, auto vast_decl_builder) {
            return cgctx.declare(decl, vast_decl_builder);
        }

        hl::EnumConstantOp declare(const clang::EnumConstantDecl *decl, auto vast_decl_builder) {
            return cgctx.declare(decl, vast_decl_builder);
        }

        context_t &cgctx;

        meta_generator meta;
        std::unique_ptr< scope_t > scope;
        std::unique_ptr< visitor_t > visitor;
    };

    using default_codegen       = codegen_instance< codegen_context, default_visitor_stack, default_meta_gen >;
    using codegen_with_meta_ids = codegen_instance< codegen_context, default_visitor_stack, id_meta_gen >;

} // namespace vast::cg
