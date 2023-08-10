// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/DeclVisitor.h>
#include <clang/AST/Attr.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Frontend/FrontendDiagnostic.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/CodeGenVisitorLens.hpp"
#include "vast/CodeGen/CodeGenFunction.hpp"
#include "vast/CodeGen/Mangler.hpp"
#include "vast/CodeGen/Util.hpp"

#include "vast/Util/Scopes.hpp"

#include "vast/Dialect/HighLevel/HighLevelLinkage.hpp"

#include "vast/CodeGen/FunctionInfo.hpp"

namespace vast::cg {

    template< typename Derived >
    struct CodeGenDeclVisitor
        : clang::ConstDeclVisitor< CodeGenDeclVisitor< Derived >, operation >
        , CodeGenVisitorLens< CodeGenDeclVisitor< Derived >, Derived >
        , CodeGenBuilder< CodeGenDeclVisitor< Derived >, Derived >
    {
        using LensType = CodeGenVisitorLens< CodeGenDeclVisitor< Derived >, Derived >;

        using LensType::derived;
        using LensType::context;
        using LensType::mcontext;
        using LensType::acontext;

        using LensType::meta_location;

        using LensType::name_mangler;

        using LensType::visit;
        using LensType::visit_as_lvalue_type;

        using Builder = CodeGenBuilder< CodeGenDeclVisitor< Derived >, Derived >;

        using Builder::builder;
        using Builder::make_value_builder;

        using Builder::insertion_guard;

        using Builder::set_insertion_point_to_start;
        using Builder::set_insertion_point_to_end;

        using Builder::get_current_function;

        using Builder::constant;

        template< typename Op, typename... Args >
        auto make(Args &&...args) {
            return this->template create< Op >(std::forward< Args >(args)...);
        }

        static bool is_defaulted_method(const clang::FunctionDecl *function_decl)  {
            if (function_decl->isDefaulted() && clang::isa< clang::CXXMethodDecl >(function_decl)) {
                auto method = clang::cast< clang::CXXMethodDecl >(function_decl);
                return method->isCopyAssignmentOperator() || method->isMoveAssignmentOperator();
            }

            return false;
        }

        // Effectively create the vast instruction, properly handling insertion points.
        vast_function create_vast_function(
            mlir::Location loc, mangled_name_ref mangled_name, mlir::FunctionType fty, const clang::FunctionDecl *function_decl
        ) {
            // At the point we need to create the function, the insertion point
            // could be anywhere (e.g. callsite). Do not rely on whatever it might
            // be, properly save, find the appropriate place and restore.
            auto guard = insertion_guard();
            auto linkage = hl::get_function_linkage(function_decl);

            // make function header, that will be later filled with function body
            // or returned as declaration in the case of external function
            auto fn = context().declare(mangled_name, [&] () {
                return make< hl::FuncOp >(loc, mangled_name.name, fty, linkage);
            });

            VAST_CHECK(fn.isDeclaration(), "expected empty body");

            mlir::SymbolTable::setSymbolVisibility(
                fn, get_visibility_from_linkage(linkage)
            );

            return fn;
        }

        bool record_conflicting_definition(clang::GlobalDecl glob) {
            return context().diagnosed_conflicting_definitions.insert(glob).second;
        }

        vast_function get_or_create_vast_function(
            mangled_name_ref mangled_name, mlir_type type, clang::GlobalDecl glob, global_emition emit
        ) {
            VAST_UNIMPLEMENTED_IF(emit.for_vtable);
            VAST_UNIMPLEMENTED_IF(emit.thunk);

            const auto *decl = glob.getDecl();

            // Any attempts to use a MultiVersion function should result in retrieving the
            // iFunc instead. Name mangling will handle the rest of the changes.
            if (const auto *fn = clang::cast_or_null< clang::FunctionDecl >(decl)) {
                VAST_UNIMPLEMENTED_IF(acontext().getLangOpts().OpenMPIsDevice);
                VAST_UNIMPLEMENTED_IF(fn->isMultiVersion());
            }

            // Lookup the entry, lazily creating it if necessary.
            auto *entry = context().get_global_value(mangled_name);
            if (entry) {
                VAST_UNIMPLEMENTED_IF(!mlir::isa< hl::FuncOp >(entry));
                VAST_UNIMPLEMENTED_IF(context().weak_ref_references.erase(entry));

                // Handle dropped DLL attributes.
                if (decl && !decl->hasAttr< clang::DLLImportAttr >() && !decl->hasAttr< clang::DLLExportAttr >()) {
                    // TODO: Entry->setDLLStorageClass
                    // setDSOLocal(Entry);
                }

                // If there are two attempts to define the same mangled name, issue an error.
                auto fn = mlir::cast< hl::FuncOp >(entry);
                if (is_for_definition(emit) && fn && !fn.isDeclaration()) {
                    // Check that glob is not yet in DiagnosedConflictingDefinitions is required
                    // to make sure that we issue and error only once.
                    if (auto other = name_mangler().lookup_representative_decl(mangled_name)) {
                        if (glob.getCanonicalDecl().getDecl()) {
                            if (record_conflicting_definition(glob)) {
                                auto &diags = acontext().getDiagnostics();
                                // FIXME: this should not be responsibility of visitor
                                diags.Report(decl->getLocation(), clang::diag::err_duplicate_mangled_name) << mangled_name.name;
                                diags.Report(other->getDecl()->getLocation(), clang::diag::note_previous_definition);
                            }
                        }
                    }
                }

                if (fn && fn.getFunctionType() == type) {
                    return fn;
                }

                VAST_UNREACHABLE("NYI");

                // TODO: clang checks here if this is a llvm::GlobalAlias... how will we
                // support this?
            }

            // This function doesn't have a complete type (for example, the return type is
            // an incomplete struct). Use a fake type instead, and make sure not to try to
            // set attributes.
            bool is_incomplete_function = false;

            mlir::FunctionType fty;
            if (type.isa< mlir::FunctionType >()) {
                fty = type.cast< mlir::FunctionType >();
            } else {
                VAST_UNIMPLEMENTED_MSG("functions with incomplete types");
                is_incomplete_function = true;
            }

            auto *function_decl = llvm::cast< clang::FunctionDecl >(decl);
            VAST_CHECK(function_decl, "Only FunctionDecl supported so far.");

            // TODO: CodeGen includeds the linkage (ExternalLinkage) and only passes the
            // mangled_name if entry is nullptr
            auto fn = create_vast_function(meta_location(function_decl), mangled_name, fty, function_decl);

            if (entry) {
                VAST_UNIMPLEMENTED;
            }

            // TODO: This might not be valid, seems the uniqueing system doesn't make
            // sense for MLIR
            // VAST_ASSERT(F->getName().getStringRef() == MangledName && "name was uniqued!");

            if (decl) {
                ; // TODO: set function attributes from the declaration
            }

            // TODO: set function attributes from the missing attributes param

            // TODO: Handle extra attributes

            if (emit.defer) {
                // All MSVC dtors other than the base dtor are linkonce_odr and delegate to
                // each other bottoming out wiht the base dtor. Therefore we emit non-base
                // dtors on usage, even if there is no dtor definition in the TU.
                if (decl && clang::isa< clang::CXXDestructorDecl >(decl)) {
                    VAST_UNIMPLEMENTED;
                }

                // This is the first use or definition of a mangled name. If there is a
                // deferred decl with this name, remember that we need to emit it at the end
                // of the file.
                // FIXME: encapsulate this eventually
                auto &deferred = context().deferred_decls;
                if (auto ddi = deferred.find(mangled_name); ddi != deferred.end()) {
                    // Move the potentially referenced deferred decl to the
                    // DeferredDeclsToEmit list, and remove it from DeferredDecls (since we
                    // don't need it anymore).

                    context().add_deferred_decl_to_emit(ddi->second);
                    deferred.erase(ddi);

                    // Otherwise, there are cases we have to worry about where we're using a
                    // declaration for which we must emit a definition but where we might not
                    // find a top-level definition.
                    //   - member functions defined inline in their classes
                    //   - friend functions defined inline in some class
                    //   - special member functions with implicit definitions
                    // If we ever change our AST traversal to walk into class methods, this
                    // will be unnecessary.
                    //
                    // We also don't emit a definition for a function if it's going to be an
                    // entry in a vtable, unless it's already marked as used.
                } else if (acontext().getLangOpts().CPlusPlus && decl) {
                    // Look for a declaration that's lexically in a record.
                    const auto *function_decl = clang::cast< clang::FunctionDecl >(decl)->getMostRecentDecl();
                    for (; function_decl; function_decl = function_decl->getPreviousDecl()) {
                        if (clang::isa< clang::CXXRecordDecl >(function_decl->getLexicalDeclContext())) {
                            if (function_decl->doesThisDeclarationHaveABody()) {
                                if (is_defaulted_method(function_decl)) {
                                    context().add_default_methods_to_emit(glob.getWithDecl(function_decl));
                                } else {
                                    context().add_deferred_decl_to_emit(glob.getWithDecl(function_decl));
                                }
                                break;
                            }
                        }
                    }
                }
            }

            if (!is_incomplete_function) {
                VAST_ASSERT(fn.getFunctionType() == type);
                return fn;
            }

            VAST_UNREACHABLE("codegen of incomplete function");
        }

        vast_function get_addr_of_function(
            clang::GlobalDecl decl, mlir_type fty, global_emition emit
        ) {
            VAST_UNIMPLEMENTED_IF(emit.for_vtable);

            // TODO: is this true for vast?
            VAST_CHECK(!clang::cast< clang::FunctionDecl >(decl.getDecl())->isConsteval(),
                "consteval function should never be emitted"
            );

            VAST_CHECK(fty, "missing funciton type");
            // TODO: do we need this:
            // if (!type) {
            //     const auto *fn = clang::cast< clang::FunctionDecl >(decl.getDecl());
            //     type = type_conv.get_function_type(fn->getType());
            // }

            VAST_UNIMPLEMENTED_IF(clang::dyn_cast< clang::CXXDestructorDecl >(decl.getDecl()));

            auto mangled_name = context().get_mangled_name(decl);
            return get_or_create_vast_function(mangled_name, fty, decl, emit);
        }

        // Implelements buildGlobalFunctionDefinition of vast codegen
        operation build_function_prototype(clang::GlobalDecl decl, mlir_type fty) {
            // Get or create the prototype for the function.
            // TODO: Figure out what to do here? llvm uses a GlobalValue for the FuncOp in mlir
            return get_addr_of_function(decl, fty, deferred_emit_definition);
        }

        clang::GlobalDecl get_gdecl(const clang::FunctionDecl *decl) {
            return decl;
        }

        clang::GlobalDecl get_gdecl(const clang::CXXConstructorDecl *decl) {
            return clang::GlobalDecl(decl, clang::CXXCtorType::Ctor_Complete);
        }

        clang::GlobalDecl get_gdecl(const clang::CXXDestructorDecl *decl) {
            return clang::GlobalDecl(decl, clang::CXXDtorType::Dtor_Complete);
        }

        // FIXME: remove as this duplicates logic from codegen driver
        template< typename Decl >
        operation VisitFunctionLikeDecl(const Decl *decl) {
            auto gdecl = get_gdecl(decl);
            auto mangled = context().get_mangled_name(gdecl);

            if (auto fn = context().lookup_function(mangled, false /* emit no error */)) {
                return fn;
            }

            auto guard = insertion_guard();
            auto is_definition = decl->isThisDeclarationADefinition();

            // emit definition instead of declaration
            if (!is_definition && decl->getDefinition()) {
                return visit(decl->getDefinition());
            }

            auto is_terminator = [] (auto &op) {
                return op.template hasTrait< mlir::OpTrait::IsTerminator >() ||
                       mlir::isa< hl::ReturnOp >(op);
            };

            auto declare_function_params = [&, this] (auto entry) {
                // In MLIR the entry block of the function must have the same
                // argument list as the function itself.
                // FIXME: driver solves this already
                auto params = llvm::zip(decl->getDefinition()->parameters(), entry->getArguments());
                for (const auto &[arg, earg] : params) {
                    context().declare(arg, mlir_value(earg));
                }
            };

            auto emit_function_terminator = [&] (auto fn) {
                auto loc = fn.getLoc();
                if (decl->getReturnType()->isVoidType()) {
                    make< hl::ReturnOp >(loc);
                } else {
                    if (decl->isMain()) {
                        // return zero if no return is present in main
                        auto type = fn.getFunctionType();
                        auto zero = constant(loc, type.getResult(0), apsint(0));
                        make< hl::ReturnOp >(loc, zero);
                    } else {
                        make< hl::UnreachableOp >(loc);
                    }
                }
            };

            auto emit_function_body = [&] (auto fn) {
                auto entry = fn.addEntryBlock();
                set_insertion_point_to_start(entry);

                if (decl->hasBody()) {
                    declare_function_params(entry);

                    // emit label declarations
                    llvm::ScopedHashTableScope labels_scope(context().labels);

                    for (const auto label : filter< clang::LabelDecl >(decl->decls()))
                        this->visit(label);

                    visit(decl->getBody());
                }

                auto &fn_blocks  = fn.getBlocks();
                auto &last_block = fn_blocks.back();
                auto &ops        = last_block.getOperations();
                set_insertion_point_to_end(&last_block);

                auto last_op = &ops.back();

                // Making sure, that if the operation is enclosed in a trailing
                // scope, then the termiantor is evaluated in this scope (which
                // will then be spliced by subsequent pass)
                auto next_scope = [](operation op) -> hl::ScopeOp {
                    if (op)
                        return mlir::dyn_cast< hl::ScopeOp >(op);
                    return {};
                };

                auto process_scope = [&](hl::ScopeOp scope) -> operation {
                    auto parent = scope->getParentRegion();
                    if (parent->hasOneBlock()
                        && parent->back().begin() == std::prev(parent->back().end()))
                    {
                        set_insertion_point_to_end(&scope.getBody());
                        return get_last_op(scope);
                    }
                    return {};
                };

                if (!ops.empty()) {
                    while (auto scope = next_scope(last_op)) {
                        last_op = process_scope(scope);
                    }
                }

                if (ops.empty()
                    || !last_op
                    || !is_terminator(*last_op))
                {
                    emit_function_terminator(fn);
                }
            };

            llvm::ScopedHashTableScope scope(context().vars);

            auto linkage = hl::get_function_linkage(gdecl);

            auto fn = context().declare(mangled, [&] () {
                auto loc  = meta_location(decl);
                auto type = visit(decl->getFunctionType()).template cast< mlir::FunctionType >();
                // make function header, that will be later filled with function body
                // or returned as declaration in the case of external function
                return make< hl::FuncOp >(loc, mangled.name, type, linkage);
            });

            if (!is_definition) {
                // MLIR requires declrations to have private visibility
                fn.setVisibility(mlir::SymbolTable::Visibility::Private);
                return fn;
            }

            fn.setVisibility(get_visibility_from_linkage(linkage));

            if (fn.empty()) {
                emit_function_body(fn);
            }

            return fn;
        }

        operation VisitFunctionDecl(const clang::FunctionDecl *decl) {
            return VisitFunctionLikeDecl(decl);
        }

        operation VisitCXXConstructorDecl(const clang::CXXConstructorDecl *decl) {
            return VisitFunctionLikeDecl(decl);
        }

        operation VisitCXXDestructorDecl(const clang::CXXDestructorDecl *decl) {
            return VisitFunctionLikeDecl(decl);
        }

        //
        // Variable Declration
        //

        hl::StorageClass VisitStorageClass(const clang::VarDecl *decl) const {
            switch (decl->getStorageClass()) {
                case clang::SC_None: return hl::StorageClass::sc_none;
                case clang::SC_Auto: return hl::StorageClass::sc_auto;
                case clang::SC_Static: return hl::StorageClass::sc_static;
                case clang::SC_Extern: return hl::StorageClass::sc_extern;
                case clang::SC_PrivateExtern: return hl::StorageClass::sc_private_extern;
                case clang::SC_Register: return hl::StorageClass::sc_register;
            }
            VAST_UNREACHABLE("unknown storage class");
        }

        hl::TSClass VisitThreadStorageClass(const clang::VarDecl *decl) const {
            switch (decl->getTSCSpec()) {
                case clang::TSCS_unspecified: return hl::TSClass::tsc_none;
                case clang::TSCS___thread: return hl::TSClass::tsc_gnu_thread;
                case clang::TSCS_thread_local: return hl::TSClass::tsc_cxx_thread;
                case clang::TSCS__Thread_local: return hl::TSClass::tsc_c_thread;
            }
            VAST_UNREACHABLE("unknown storage class");
        }

        operation VisitVarDecl(const clang::VarDecl *decl) {
            return context().declare(decl, [&] {
                auto type = decl->getType();
                bool has_allocator = type->isVariableArrayType();
                bool has_init = decl->getInit();

                auto initializer = make_value_builder(decl->getInit());

                auto array_allocator = [decl, this](auto &bld, auto loc) {
                    if (auto type = clang::dyn_cast< clang::VariableArrayType >(decl->getType())) {
                        make_value_builder(type->getSizeExpr())(bld, loc);
                    }
                };

                auto var = this->template make_operation< hl::VarDeclOp >()
                    .bind(meta_location(decl))                                  // location
                    .bind(visit_as_lvalue_type(type))                           // type
                    .bind(context().decl_name(decl->getUnderlyingDecl()))       // name
                    .bind_region_if(has_init, std::move(initializer))           // initializer
                    .bind_region_if(has_allocator, std::move(array_allocator))  // array allocator
                    .freeze();

                if (auto sc = VisitStorageClass(decl); sc != hl::StorageClass::sc_none) {
                    var.setStorageClass(sc);
                }

                if (auto tsc = VisitThreadStorageClass(decl); tsc != hl::TSClass::tsc_none) {
                    var.setThreadStorageClass(tsc);
                }

                return var;
            }).getDefiningOp();
        }

        operation VisitParmVarDecl(const clang::ParmVarDecl *decl) {
            if (auto var = context().vars.lookup(decl))
                return var.getDefiningOp();
            context().error("error: missing parameter declaration " + decl->getName());
            return nullptr;
        }

        // operation VisitImplicitParamDecl(const clang::ImplicitParamDecl *decl)

        // operation VisitLinkageSpecDecl(const clang::LinkageSpecDecl *decl)

        operation VisitTranslationUnitDecl(const clang::TranslationUnitDecl *tu) {
            auto loc = meta_location(tu);
            return this->template make_scoped< TranslationUnitScope >(loc, [&] {
                for (const auto &decl : tu->decls()) {
                    visit(decl);
                }
            });
        }

        // operation VisitTypedefNameDecl(const clang::TypedefNameDecl *decl)

        inline void walk_type(clang::QualType type, invocable< clang::Type * > auto &&yield) {
            if (yield(type)) {
                return;
            }

            if (auto arr = clang::dyn_cast< clang::ArrayType >(type)) {
                walk_type(arr->getElementType(), yield);
            }

            if (auto ptr = clang::dyn_cast< clang::PointerType >(type)) {
                walk_type(ptr->getPointeeType(), yield);
            }
        }

        static inline std::string parse_annotation(acontext_t &ctx, const clang::Attr *attr) {
            // Clang does not provide a nicer interface :(
            std::string buff; llvm::raw_string_ostream stream(buff);
            auto policy = ctx.getPrintingPolicy();
            attr->printPretty(stream, policy);
            llvm::StringRef ref(buff);
            ref.consume_front(" __attribute__((annotate(\"");
            ref.consume_back("\")))");
            return ref.str();
        }

        void attach_attributes(const clang::Decl *from, auto &to) {
            if (!from->hasAttrs())
                return;

            auto &actx = acontext();
            for (auto attr: from->getAttrs()) {
                auto annot = mlir::StringAttr::get(to->getContext(), parse_annotation(actx, attr));
                to->setAttr("annotation", hl::AnnotationAttr::get(annot));
            }
        }

        operation VisitTypedefDecl(const clang::TypedefDecl *decl) {
            return context().declare(decl, [&] {
                auto type = [&, this] () -> mlir::Type {
                    auto underlying = decl->getUnderlyingType();
                    if (auto fty = clang::dyn_cast< clang::FunctionType >(underlying)) {
                        return visit(fty);
                    }

                    // predeclare named underlying types if necessery
                    walk_type(underlying, [=, this](auto ty) {
                        if (auto tag = clang::dyn_cast< clang::TagType >(ty)) {
                            this->visit(tag->getDecl());
                            return true; // stop recursive walk
                        }

                        return false;
                    });

                    return visit(underlying);
                };

                // create typedef operation
                auto def = this->template make_operation< hl::TypeDefOp >()
                    .bind(meta_location(decl)) // location
                    .bind(decl->getName())     // name
                    .bind(type())              // type
                    .freeze();

                attach_attributes(decl /* from */, def /* to */);
                return def;
            });
        }

        // operation VisitTypeAliasDecl(const clang::TypeAliasDecl *decl)

        operation VisitLabelDecl(const clang::LabelDecl *decl) {
            return context().declare(decl, [&] {
                return this->template make_operation< hl::LabelDeclOp >()
                    .bind(meta_location(decl))  // location
                    .bind(decl->getName())      // name
                    .freeze();
            });
        }

        operation VisitEmptyDecl(const clang::EmptyDecl *decl) {
            return this->template make_operation< hl::EmptyDeclOp >()
                .bind(meta_location(decl)) // location
                .freeze();
        }

        //
        // Enum Declarations
        //
        operation VisitEnumDecl(const clang::EnumDecl *decl) {
            if (!decl->isFirstDecl()) {
                auto prev = decl->getPreviousDecl();

                if (!decl->isComplete()) {
                    return context().enumdecls.lookup(prev);
                }

                while (prev) {
                    if (auto prev_op = context().enumdecls.lookup(prev)) {
                        VAST_ASSERT(!prev->isComplete());
                        prev_op.setType(visit(decl->getIntegerType()));
                        auto guard = insertion_guard();
                        set_insertion_point_to_start(&prev_op.getConstants().front());
                        for (auto con : decl->enumerators()) {
                            visit(con);
                        }
                        return prev_op;
                    }
                    prev = prev->getPreviousDecl();
                }
            }

            return context().declare(decl, [&] {
                if (!decl->isComplete()) {
                    return this->template make_operation< hl::EnumDeclOp >()
                        .bind(meta_location(decl))                           // location
                        .bind(decl->getName())                               // name
                        .freeze();
                }

                auto constants = [&] (auto &bld, auto loc) {
                    for (auto con : decl->enumerators()) {
                        visit(con);
                    }
                };

                return this->template make_operation< hl::EnumDeclOp >()
                    .bind(meta_location(decl))                              // location
                    .bind(decl->getName())                                  // name
                    .bind(visit(decl->getIntegerType()))                    // type
                    .bind(constants)                                        // constants
                    .freeze();
            });
        }

        operation VisitEnumConstantDecl(const clang::EnumConstantDecl *decl) {
            return context().declare(decl, [&] {
                auto initializer = make_value_builder(decl->getInitExpr());

                auto type = visit(decl->getType());

                return this->template make_operation< hl::EnumConstantOp >()
                    .bind(meta_location(decl))                              // location
                    .bind(decl->getName())                                  // name
                    .bind(type)                                             // type
                    .bind(decl->getInitVal())                               // value
                    .bind_if(decl->getInitExpr(), std::move(initializer))   // initializer
                    .freeze();
            });
        }

        hl::AccessSpecifier convert_access(clang::AccessSpecifier spec) {
            switch(spec) {
                case clang::AccessSpecifier::AS_public:
                    return hl::AccessSpecifier::as_public;
                case clang::AccessSpecifier::AS_protected:
                    return hl::AccessSpecifier::as_protected;
                case clang::AccessSpecifier::AS_private:
                    return hl::AccessSpecifier::as_private;
                case clang::AccessSpecifier::AS_none:
                    return hl::AccessSpecifier::as_none;
            }
            VAST_UNREACHABLE("unknown access specifier");
        }

        //
        // Record Declaration
        //
        template< typename Op, typename Decl >
        operation make_record_decl(const Decl *decl) {
            auto loc  = meta_location(decl);
            auto name = context().decl_name(decl);

            // declare the type first to allow recursive type definitions
            if (!decl->isCompleteDefinition()) {
                return context().declare(decl, [&] {
                    return this->template make_operation< hl::TypeDeclOp >()
                        .bind(meta_location(decl)) // location
                        .bind(decl->getName())     // name
                        .freeze();
                });
            }

            auto fields = [&](auto &bld, auto loc) {
                for (auto child: decl->decls()) {
                    if (auto field = clang::dyn_cast< clang::FieldDecl >(child)) {
                        visit(field);
                    } else if (auto access = clang::dyn_cast< clang::AccessSpecDecl >(child)) {
                        visit(access);
                    } else if (auto var = clang::dyn_cast< clang::VarDecl >(child)) {
                        visit(var);
                    } else if (auto ctor = clang::dyn_cast< clang::CXXConstructorDecl >(child)) {
                        visit(ctor);
                    } else if (auto dtor = clang::dyn_cast< clang::CXXDestructorDecl >(child)) {
                        visit(dtor);
                    } else if (auto func = clang::dyn_cast< clang::FunctionDecl >(child)) {
                        auto name = func->getDeclName();
                        if (name.getNameKind() != clang::DeclarationName::NameKind::Identifier) {
                            // TODO(frabert): cannot mangle non-identifiers for now
                            continue;
                        }
                        visit(func);
                    }
                }
            };

            if constexpr (std::is_same_v< Decl, clang::CXXRecordDecl >) {
                auto bases = [&](auto &bld, auto loc) {
                    for (auto &base : decl->bases()) {
                        auto loc = meta_location(base);
                        make< hl::CxxBaseSpecifierOp >(
                            loc,
                            visit(base.getType()),
                            convert_access(base.getAccessSpecifier()),
                            base.isVirtual());
                    }
                };

                return make< Op >(loc, name, bases, fields);
            }

            return make< Op >(loc, name, fields);
        }

        operation VisitRecordDecl(const clang::RecordDecl *decl) {
            if (decl->isUnion()) {
                return make_record_decl< hl::UnionDeclOp >(decl);
            } else {
                return make_record_decl< hl::StructDeclOp >(decl);
            }
        }

        operation VisitCXXRecordDecl(const clang::CXXRecordDecl *decl) {
            if (decl->isClass()) {
                return make_record_decl< hl::ClassDeclOp >(decl);
            }
            return make_record_decl< hl::CxxStructDeclOp >(decl);
        }

        operation VisitAccessSpecDecl(const clang::AccessSpecDecl *decl) {
            auto loc = meta_location(decl);
            return make< hl::AccessSpecifierOp >(
                loc,
                convert_access(decl->getAccess()));
        }

        operation VisitFieldDecl(const clang::FieldDecl *decl) {
            // define field type if the field defines a new nested type
            if (auto tag = decl->getType()->getAsTagDecl()) {
                if (tag->isThisDeclarationADefinition()) {
                    if (!context().tag_names.count(tag)) {
                        visit(tag);
                    }
                }
            }

            return this->template make_operation< hl::FieldDeclOp >()
                .bind(meta_location(decl))              // location
                .bind(context().get_decl_name(decl))    // name
                .bind(visit(decl->getType()))           // type
                .bind(decl->getBitWidth() ? context().u32(decl->getBitWidthValue(acontext())) : nullptr) // bitfield
                .freeze();

        }
    };

} // namespace vast::cg
