// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Expr.h>
#include <clang/Frontend/ASTUnit.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Verifier.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"

#include "vast/Translation/CodeGenVisitor.hpp"
#include "vast/Translation/CodeGenFallBackVisitor.hpp"

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/Meta/MetaDialect.hpp"
#include "vast/Dialect/Dialects.hpp"

#include "vast/Translation/DataLayout.hpp"
#include "vast/Translation/CodeGenMeta.hpp"
#include "vast/Translation/CodeGenOptions.hpp"
#include "vast/Translation/Types.hpp"

namespace vast::cg
{
    namespace detail {
        static inline mcontext_t& codegen_context_setup(mcontext_t &ctx) {
            mlir::registerAllDialects(ctx);
            vast::registerAllDialects(ctx);

            ctx.loadAllAvailableDialects();
            return ctx;
        };

    } // namespace detail

    //
    // CodeGenUnit
    //
    // It takes care of translation of single translation unit or declaration.
    //
    template< typename CodeGenVisitor >
    struct CodeGenBase
    {
        using MetaGenerator = typename CodeGenVisitor::MetaGeneratorType;

        CodeGenBase(mcontext_t *mctx, MetaGenerator &meta)
            : _mctx(mctx), _meta(meta), _cgctx(nullptr), _module(nullptr)
        {
            detail::codegen_context_setup(*_mctx);
        }

        CodeGenBase(acontext_t *actx, mcontext_t *mctx, MetaGenerator &meta)
            : CodeGenBase(mctx, meta)
        {
            setup_codegen(*actx);
        }


        owning_module_ref emit_module(clang::ASTUnit *unit) {
            append_to_module(unit);
            return freeze();
        }

        owning_module_ref emit_module(clang::Decl *decl) {
            append_to_module(decl);
            return freeze();
        }

        void append_to_module(clang::ASTUnit *unit) { append_impl(unit); }

        void append_to_module(clang::Decl *decl) { append_impl(decl); }

        void append_to_module(clang::Stmt *stmt) { append_impl(stmt); }

        void append_to_module(clang::Expr *expr) { append_impl(expr); }

        void append_to_module(clang::Type *type) { append_impl(type); }

        owning_module_ref freeze() {
            hl::emit_data_layout(*_mctx, _module, _cgctx->data_layout());
            return std::move(_module);
        }

        operation build_function_prototype(clang::GlobalDecl decl, mlir_type fty) {
            return _visitor->build_function_prototype(decl, fty);
        }

        template< typename From, typename Symbol >
        using ScopedSymbolTable = llvm::ScopedHashTableScope< From, Symbol >;

        using TypeDefsScope      = ScopedSymbolTable< const clang::TypedefDecl *, hl::TypeDefOp >;
        using TypeDeclsScope     = ScopedSymbolTable< const clang::TypeDecl *, hl::TypeDeclOp >;
        using EnumDeclsScope     = ScopedSymbolTable< const clang::EnumDecl *, hl::EnumDeclOp >;
        using EnumConstantsScope = ScopedSymbolTable< const clang::EnumConstantDecl *, hl::EnumConstantOp >;
        using FunctionsScope     = ScopedSymbolTable< const clang::FunctionDecl *, hl::FuncOp >;
        using VariablesScope     = ScopedSymbolTable< const clang::VarDecl *, Value >;

        struct CodegenScope {
            TypeDefsScope      typedefs;
            TypeDeclsScope     typedecls;
            EnumDeclsScope     enumdecls;
            EnumConstantsScope enumconsts;
            FunctionsScope     funcdecls;
            VariablesScope     globs;
        };

        bool verify_module() const {
            return mlir::verify(_module.get()).succeeded();
        }

        operation get_global_value(string_ref name) {
            return _cgctx->get_global_value(name);
        }

        mlir_value get_global_value(const clang::Decl *decl) {
            return _cgctx->get_global_value(decl);
        }

        const std::vector< clang::GlobalDecl >& default_methods_to_emit() const {
            return _cgctx->default_methods_to_emit;
        }

        const std::vector< clang::GlobalDecl >& deferred_decls_to_emit() const {
            return _cgctx->deferred_decls_to_emit;
        }

        const std::vector< const clang::CXXRecordDecl * >& deferred_vtables() const {
            return _cgctx->deferred_vtables;
        }

        const std::map< string_ref, clang::GlobalDecl >& deferred_decls() const {
            return _cgctx->deferred_decls;
        }

        std::vector< clang::GlobalDecl >&& receive_deferred_decls_to_emit() {
            std::vector< clang::GlobalDecl > current;
            current.swap(_cgctx->deferred_decls_to_emit);
            return std::move(current);
        }

        lexical_scope_context * current_lexical_scope() {
            return _cgctx->current_lexical_scope;
        }

        void set_current_lexical_scope(lexical_scope_context *scope) {
            _cgctx->current_lexical_scope = scope;
        }

        mlir_type convert(qual_type type) { return _visitor->Visit(type); }

        CodeGenContext::VarTable& variables_symbol_table() { return _cgctx->vars; }

        // correspond to clang::CodeGenFunction::GenerateCode
        hl::FuncOp emit_function_prologue(
            hl::FuncOp fn, clang::GlobalDecl decl,  const function_info_t &fty_info,
            function_arg_list args, const codegen_options &options
        ) {
            assert(fn && "generating code for a null function");
            const auto function_decl = clang::cast< clang::FunctionDecl >(decl.getDecl());

            if (function_decl->isInlineBuiltinDeclaration()) {
                throw cg::unimplemented("emit body of inline builtin declaration");
            } else {
                // Detect the unusual situation where an inline version is shadowed by a
                // non-inline version. In that case we should pick the external one
                // everywhere. That's GCC behavior too. Unfortunately, I cannot find a way
                // to detect that situation before we reach codegen, so do some late
                // replacement.
                for (const auto *prev = function_decl->getPreviousDecl(); prev; prev = prev->getPreviousDecl()) {
                    if (LLVM_UNLIKELY(prev->isInlineBuiltinDeclaration())) {
                        throw cg::unimplemented("emit body of inline builtin declaration");
                    }
                }
            }

            // Check if we should generate debug info for this function.
            if (function_decl->hasAttr< clang::NoDebugAttr >()) {
                throw cg::unimplemented("emit no debug meta");
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
            auto loc = meta_location(function_decl);

            // If this is a function specialization then use the pattern body as the
            // location for the function.
            if (const auto *spec = function_decl->getTemplateInstantiationPattern()) {
                if (spec->hasBody(spec)) {
                    loc = meta_location(spec);
                }
            }

            // FIXME: maybe move to codegen visitor
            if (auto body = function_decl->getBody()) {
                // LLVM codegen: Coroutines always emit lifetime markers
                // Hide this under request for lifetime emission so that we can write
                // tests when the time comes, but CIR should be intrinsically scope
                // accurate, so no need to tie coroutines to such markers.
                if (clang::isa< clang::CoroutineBodyStmt >(body)) {
                    throw cg::unimplemented("emit lifetime markers");
                }

                // Initialize helper which will detect jumps which can cause invalid
                // lifetime markers.
                if (options.should_emit_lifetime_markers) {
                    throw cg::unimplemented("emit lifetime markers");
                }
            }

            // Create a scope in the symbol table to hold variable declarations.
            llvm::ScopedHashTableScope var_scope(variables_symbol_table());
            {
                auto body = function_decl->getBody();
                auto begin_loc = meta_location(body);
                auto end_loc = meta_location(body);

                assert(fn.isDeclaration() && "Function already has body?");
                auto *entry_block = fn.addEntryBlock();
                _visitor->set_insertion_point_to_start(entry_block);

                lexical_scope_context lex_ccope{begin_loc, end_loc, entry_block};
                lexical_scope_guard scope_guard{*this, &lex_ccope};

                // Emit the standard function prologue.
                start_function(decl, fn, fty_info, args, loc, options);

                // Initialize lexical scope information.

                // Save parameters for coroutine function.
                if (body && clang::isa_and_nonnull< clang::CoroutineBodyStmt >(body)) {
                    throw cg::unimplemented("coroutine parameters");
                }

                // Generate the body of the function.
                // TODO: PGO.assignRegionCounters

                const auto &lang = _cgctx->actx.getLangOpts();

                if (clang::isa< clang::CXXDestructorDecl >(function_decl))
                    llvm_unreachable("NYI");
                else if (clang::isa< clang::CXXConstructorDecl >(function_decl))
                    llvm_unreachable("NYI");
                else if (lang.CUDA && !lang.CUDAIsDevice && function_decl->hasAttr< clang::CUDAGlobalAttr >())
                    llvm_unreachable("NYI");
                else if (auto method = clang::dyn_cast< clang::CXXMethodDecl >(function_decl); method && method->isLambdaStaticInvoker()) {
                    llvm_unreachable("NYI");
                } else if (function_decl->isDefaulted() && clang::isa< clang::CXXMethodDecl >(function_decl) &&
                    (clang::cast< clang::CXXMethodDecl >(function_decl)->isCopyAssignmentOperator() ||
                     clang::cast< clang::CXXMethodDecl >(function_decl)->isMoveAssignmentOperator())
                ) {
                    llvm_unreachable("NYI");
                } else if (body) {
                    if (mlir::failed(build_function_body(body))) {
                        throw cg::codegen_error("failed function body codegen");
                    }
                } else {
                    llvm_unreachable("no definition for emitted function");
                }
            }

            return fn;
        }

        // TODO: This is currently just a dumb stub. But we want to be able to clearly
        // assert where we arne't doing things that we know we should and will crash
        // as soon as we add a DebugInfo type to this class.
        std::nullptr_t *get_debug_info() { return nullptr; }

        void start_function(
            clang::GlobalDecl glob,
            hl::FuncOp fn,
            const function_info_t &fty_info,
            const function_arg_list &args,
            mlir::Location loc,
            const codegen_options &options
        ) {
            const auto *decl = glob.getDecl();
            const auto *function_decl = clang::dyn_cast_or_null< clang::FunctionDecl >(decl);
            if (function_decl && function_decl->usesSEHTry()) {
                llvm_unreachable("NYI");
            }

            const auto &lang = _cgctx->actx.getLangOpts();

            // auto curr_function_decl = decl ? decl->getNonClosureContext() : nullptr;

            // TODO: Sanitizers
            // TODO: XRay
            // TODO: PGO

            //
            unsigned entry_count = 0, entry_offset = 0;
            if (const auto *attr = decl ? decl->getAttr< clang::PatchableFunctionEntryAttr >() : nullptr) {
                llvm_unreachable("NYI");
            } else {
                entry_count = options.patchable_function_entry_count;
                entry_offset = options.patchable_function_entry_offset;
            }
            if (entry_count && entry_offset <= entry_count) {
                llvm_unreachable("NYI");
            }

            // Add no-jump-tables value.
            if (options.no_use_jump_tables)
                llvm_unreachable("NYI");

            // Add no-inline-line-tables value.
            if (options.no_inline_line_tables)
                llvm_unreachable("NYI");

            // TODO: Add profile-sample-accurate value.

            if (decl && decl->hasAttr< clang::CFICanonicalJumpTableAttr >())
                llvm_unreachable("NYI");

            if (decl && decl->hasAttr< clang::NoProfileFunctionAttr >())
                llvm_unreachable("NYI");

            if (function_decl && lang.OpenCL) {
                llvm_unreachable("NYI");
            }

            // If we are checking function types, emit a function type signature as
            // prologue data.
            // if (function_decl && lang.CPlusPlus && SanOpts.has(SanitizerKind::Function)) {
            //     llvm_unreachable("NYI");
            // }

            // If we're checking nullability, we need to know whether we can check the
            // return value. Initialize the flag to 'true' and refine it in
            // buildParmDecl.
            // if (SanOpts.has(SanitizerKind::NullabilityReturn)) {
            //     llvm_unreachable("NYI");
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
                llvm_unreachable("NYI");
            }

            // if (ShouldInstrumentFunction()) {
            //     llvm_unreachable("NYI");
            // }

            // Since emitting the mcount call here impacts optimizations such as
            // function inlining, we just add an attribute to insert a mcount call in
            // backend. The attribute "counting-function" is set to mcount function name
            // which is architecture dependent.
            // if (options.InstrumentForProfiling) {
            //     llvm_unreachable("NYI");
            // }

            if (options.packed_stack) {
                llvm_unreachable("NYI");
            }

            if (options.warn_stack_size != UINT_MAX) {
                llvm_unreachable("NYI");
            }

            // TODO: emitstartehspec

            // TODO: prologuecleanupdepth

            if (lang.OpenMP && decl) {
                llvm_unreachable("NYI");
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
                    mlir_param.setLoc(meta_location(ast_param));
                    declare(ast_param, mlir_value(mlir_param));
                }

            }

            if (decl && clang::isa< clang::CXXMethodDecl>(decl) &&
                clang::cast< clang::CXXMethodDecl>(decl)->isInstance()
            ) {
                throw cg::unimplemented( "emit prologue of cxx methods" );
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
                    llvm_unreachable("NYI");
                }
            }

            // Emit a location at the end of the prologue.
            if (get_debug_info()) {
                llvm_unreachable("NYI");
            }

            // TODO: Do we need to handle this in two places like we do with
            // target-features/target-cpu?
            if (const auto *vec_width = function_decl->getAttr< clang::MinVectorWidthAttr >()) {
                llvm_unreachable("NYI");
            }
        }

        logical_result build_function_body(const clang::Stmt *body) {
            // TODO: incrementProfileCounter(Body);

            // We start with function level scope for variables.
            llvm::ScopedHashTableScope var_scope(variables_symbol_table());

            auto result = logical_result::success();
            if (const auto stmt = clang::dyn_cast< clang::CompoundStmt >(body))
                result = build_compound_stmt_without_scope(*stmt);
            else
                result = build_stmt(body, /* use current scope */ true);

            // This is checked after emitting the function body so we know if there are
            // any permitted infinite loops.
            // TODO: if (checkIfFunctionMustProgress())
            // CurFn->addFnAttr(llvm::Attribute::MustProgress);
            return result;
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
            _visitor->Visit(stmt);
            return mlir::success();
        }

        template< typename Token >
        mlir::Location meta_location(Token token) const {
            return _meta.get(token).location();
        }

        hl::FuncOp declare(const clang::FunctionDecl *decl, auto vast_decl_builder) {
            return _cgctx->declare(decl, vast_decl_builder);
        }

        mlir_value declare(const clang::VarDecl *decl, mlir_value vast_value) {
            return _cgctx->declare(decl, vast_value);
        }

        mlir_value declare(const clang::VarDecl *decl, auto vast_decl_builder) {
            return _cgctx->declare(decl, vast_decl_builder);
        }

        hl::LabelDeclOp declare(const clang::LabelDecl *decl, auto vast_decl_builder) {
            return _cgctx->declare(decl, vast_decl_builder);
        }

        hl::TypeDefOp declare(const clang::TypedefDecl *decl, auto vast_decl_builder) {
            return _cgctx->declare(decl, vast_decl_builder);
        }

        hl::TypeDeclOp declare(const clang::TypeDecl *decl, auto vast_decl_builder) {
            return _cgctx->declare(decl, vast_decl_builder);
        }

        hl::EnumDeclOp declare(const clang::EnumDecl *decl, auto vast_decl_builder) {
            return _cgctx->declare(decl, vast_decl_builder);
        }

        hl::EnumConstantOp declare(const clang::EnumConstantDecl *decl, auto vast_decl_builder) {
            return _cgctx->declare(decl, vast_decl_builder);
        }

        bool has_insertion_block() {
            return _visitor->has_insertion_block();
        }

        void clear_insertion_point() {
            _visitor->clear_insertion_point();
        }

    private:

        void setup_codegen(acontext_t &actx) {
            if (_cgctx)
                return;

            // TODO(Heno): fix module location
            _module = { vast_module::create(mlir::UnknownLoc::get(_mctx)) };

            _cgctx = std::make_unique< CodeGenContext >(*_mctx, actx, _module);

            _scope = std::unique_ptr< CodegenScope >( new CodegenScope{
                .typedefs   = _cgctx->typedefs,
                .typedecls  = _cgctx->typedecls,
                .enumdecls  = _cgctx->enumdecls,
                .enumconsts = _cgctx->enumconsts,
                .funcdecls  = _cgctx->funcdecls,
                .globs      = _cgctx->vars
            });

            _visitor = std::make_unique< CodeGenVisitor >(*_cgctx, _meta);
        }

        template< typename AST >
        void append_impl(AST ast) {
            setup_codegen(ast->getASTContext());
            process(ast, *_visitor);
        }

        static bool process_root_decl(void * context, const clang::Decl *decl) {
            CodeGenVisitor &visitor = *static_cast<CodeGenVisitor*>(context);
            return visitor.Visit(decl), true;
        }

        void process(clang::ASTUnit *unit, CodeGenVisitor &visitor) {
            unit->visitLocalTopLevelDecls(&visitor, process_root_decl);
        }

        void process(clang::Decl *decl, CodeGenVisitor &visitor) {
            visitor.Visit(decl);
        }

        mcontext_t *_mctx;
        MetaGenerator &_meta;

        std::unique_ptr< CodeGenContext > _cgctx;
        std::unique_ptr< CodegenScope >   _scope;
        std::unique_ptr< CodeGenVisitor > _visitor;

        owning_module_ref _module;
    };

    template< typename Derived >
    using DefaultCodeGenVisitorConfig = CodeGenFallBackVisitorMixin< Derived,
        DefaultCodeGenVisitorMixin,
        DefaultFallBackVisitorMixin
    >;

    //
    // DefaultCodeGen
    //
    // Uses `DefaultMetaGenerator` and `DefaultCodeGenVisitorMixin`
    // with `DefaultFallBack` for the generation.
    //
    template<
        template< typename >
        typename VisitorConfig = DefaultCodeGenVisitorConfig,
        typename MetaGenerator = DefaultMetaGenerator
    >
    struct DefaultCodeGen
    {
        using Visitor = CodeGenVisitor< VisitorConfig, MetaGenerator >;

        using Base = CodeGenBase< Visitor >;

        DefaultCodeGen(acontext_t *actx, mcontext_t *mctx)
            : meta(actx, mctx), codegen(actx, mctx, meta)
        {}

        owning_module_ref emit_module(clang::ASTUnit *unit) {
            return codegen.emit_module(unit);
        }

        owning_module_ref emit_module(clang::Decl *decl) {
            return codegen.emit_module(decl);
        }

        bool verify_module() const { return codegen.verify_module(); }

        owning_module_ref freeze() { return codegen.freeze(); }

        mlir_type convert(qual_type type) { return codegen.convert(type); }

        operation build_function_prototype(clang::GlobalDecl decl, mlir_type fty) {
            return codegen.build_function_prototype(decl, fty);
        }

        operation get_global_value(string_ref name) {
            return codegen.get_global_value(name);
        }

        mlir_value get_global_value(const clang::Decl *decl) {
            return codegen.get_global_value(decl);
        }

        const std::vector< clang::GlobalDecl >& default_methods_to_emit() const {
            return codegen.default_methods_to_emit();
        }

        const std::vector< clang::GlobalDecl >& deferred_decls_to_emit() const {
            return codegen.deferred_decls_to_emit();
        }

        const std::vector< const clang::CXXRecordDecl * >& deferred_vtables() const {
            return codegen.deferred_vtables();
        }

        const std::map< string_ref, clang::GlobalDecl >& deferred_decls() const {
            return codegen.deferred_decls();
        }

        std::vector< clang::GlobalDecl >&& receive_deferred_decls_to_emit() {
            return codegen.receive_deferred_decls_to_emit();
        }

        CodeGenContext::VarTable & variables_symbol_table() {
            return codegen.variables_symbol_table();
        }

        template< typename Token >
        mlir::Location meta_location(Token token) const {
            return codegen.template meta_location(token);
        }

        hl::FuncOp emit_function_prologue(
            hl::FuncOp fn, clang::GlobalDecl decl,  const function_info_t &fty_info,
            function_arg_list args, const codegen_options &options
        ) {
            return codegen.emit_function_prologue(fn, decl, fty_info, args, options);
        }

        hl::FuncOp declare(const clang::FunctionDecl *decl, auto vast_decl_builder) {
            return codegen.declare(decl, vast_decl_builder);
        }

        mlir_value declare(const clang::VarDecl *decl, mlir_value vast_value) {
            return codegen.declare(decl, vast_value);
        }

        mlir_value declare(const clang::VarDecl *decl, auto vast_decl_builder) {
            return codegen.declare(decl, vast_decl_builder);
        }

        hl::LabelDeclOp declare(const clang::LabelDecl *decl, auto vast_decl_builder) {
            return codegen.declare(decl, vast_decl_builder);
        }

        hl::TypeDefOp declare(const clang::TypedefDecl *decl, auto vast_decl_builder) {
            return codegen.declare(decl, vast_decl_builder);
        }

        hl::TypeDeclOp declare(const clang::TypeDecl *decl, auto vast_decl_builder) {
            return codegen.declare(decl, vast_decl_builder);
        }

        hl::EnumDeclOp declare(const clang::EnumDecl *decl, auto vast_decl_builder) {
            return codegen.declare(decl, vast_decl_builder);
        }

        hl::EnumConstantOp declare(const clang::EnumConstantDecl *decl, auto vast_decl_builder) {
            return codegen.declare(decl, vast_decl_builder);
        }

        bool has_insertion_block() {
            return codegen.has_insertion_block();
        }

        void clear_insertion_point() {
            codegen.clear_insertion_point();
        }

        MetaGenerator meta;
        CodeGenBase< Visitor > codegen;
    };

    using CodeGenWithMetaIDs = DefaultCodeGen< DefaultCodeGenVisitorConfig, IDMetaGenerator >;

} // namespace vast::cg
