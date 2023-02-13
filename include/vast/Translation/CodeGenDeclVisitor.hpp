// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/DeclVisitor.h>
#include <clang/AST/Attr.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Frontend/FrontendDiagnostic.h>
VAST_UNRELAX_WARNINGS

#include "vast/Translation/CodeGenMeta.hpp"
#include "vast/Translation/CodeGenBuilder.hpp"
#include "vast/Translation/CodeGenVisitorBase.hpp"
#include "vast/Translation/CodeGenVisitorLens.hpp"
#include "vast/Translation/CodeGenFunction.hpp"
#include "vast/Translation/Mangler.hpp"
#include "vast/Translation/Util.hpp"

#include "vast/Dialect/HighLevel/HighLevelLinkage.hpp"

#include "vast/Translation/Error.hpp"

namespace vast::cg {

    template< typename Derived >
    struct CodeGenDeclVisitorMixin
        : clang::ConstDeclVisitor< CodeGenDeclVisitorMixin< Derived >, operation >
        , CodeGenVisitorLens< CodeGenDeclVisitorMixin< Derived >, Derived >
        , CodeGenBuilderMixin< CodeGenDeclVisitorMixin< Derived >, Derived >
    {
        using LensType = CodeGenVisitorLens< CodeGenDeclVisitorMixin< Derived >, Derived >;

        using LensType::derived;
        using LensType::context;
        using LensType::mcontext;
        using LensType::acontext;

        using LensType::meta_location;

        using LensType::name_mangler;

        using LensType::visit;
        using LensType::visit_as_lvalue_type;

        using Builder = CodeGenBuilderMixin< CodeGenDeclVisitorMixin< Derived >, Derived >;

        using Builder::builder;
        using Builder::make_value_builder;

        using Builder::set_insertion_point_to_start;
        using Builder::set_insertion_point_to_end;

        using Builder::get_current_function;

        using Builder::constant;
        using Builder::declare;

        template< typename Op, typename... Args >
        auto make(Args &&...args) {
            return this->template create< Op >(std::forward< Args >(args)...);
        }

        template< typename T >
        void filter(const auto &decls, auto &&yield) {
            for ( auto decl : decls) {
                if (auto s = clang::dyn_cast< T >(decl)) {
                    yield(s);
                }
            }
        }

        static bool is_defaulted_method(const clang::FunctionDecl *function_decl)  {
            if (function_decl->isDefaulted() && clang::isa< clang::CXXMethodDecl >(function_decl)) {
                auto method = clang::cast< clang::CXXMethodDecl >(function_decl);
                return method->isCopyAssignmentOperator() || method->isMoveAssignmentOperator();
            }

            return false;
        }

        // Effectively create the CIR instruction, properly handling insertion points.
        vast_function create_vast_function(
            mlir::Location loc, mangled_name_ref mangled_name, mlir::FunctionType fty, const clang::FunctionDecl *function_decl
        ) {
            // At the point we need to create the function, the insertion point
            // could be anywhere (e.g. callsite). Do not rely on whatever it might
            // be, properly save, find the appropriate place and restore.
            InsertionGuard guard(builder());
            auto linkage = hl::get_function_linkage(function_decl);

            // make function header, that will be later filled with function body
            // or returned as declaration in the case of external function
            auto fn = make< hl::FuncOp >(loc, mangled_name.name, fty, linkage);
            assert(fn.isDeclaration() && "expected empty body");

            mlir::SymbolTable::setSymbolVisibility(
                fn, mlir::SymbolTable::Visibility::Private
            );

            return fn;
        }

        bool record_conflicting_definition(clang::GlobalDecl glob) {
            return context().diagnosed_conflicting_definitions.insert(glob).second;
        }

        vast_function get_or_create_vast_function(
            mangled_name_ref mangled_name, mlir_type type, clang::GlobalDecl glob, global_emition emit
        ) {
            assert(!emit.for_vtable && "NYI");
            assert(!emit.thunk && "NYI");

            const auto *decl = glob.getDecl();

            // Any attempts to use a MultiVersion function should result in retrieving the
            // iFunc instead. Name mangling will handle the rest of the changes.
            if (const auto *fn = clang::cast_or_null< clang::FunctionDecl >(decl)) {
                if (acontext().getLangOpts().OpenMPIsDevice)
                    llvm_unreachable("open MP NYI");
                if (fn->isMultiVersion())
                    llvm_unreachable("NYI");
            }

            // Lookup the entry, lazily creating it if necessary.
            auto *entry = context().get_global_value(mangled_name.name);
            if (entry) {
                if ( !mlir::isa< hl::FuncOp >(entry) ) {
                    throw cg::unimplemented( "only supports FuncOp for now" );
                }

                if (context().weak_ref_references.erase(entry)) {
                    llvm_unreachable("NYI");
                }

                // Handle dropped DLL attributes.
                if (decl && !decl->hasAttr< clang::DLLImportAttr>() && !decl->hasAttr< clang::DLLExportAttr >()) {
                    llvm_unreachable("NYI");
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

                llvm_unreachable("NYI");

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
                throw cg::unimplemented("functions with incomplete types");
                is_incomplete_function = true;
            }

            auto *function_decl = llvm::cast< clang::FunctionDecl >(decl);
            assert(function_decl && "Only FunctionDecl supported so far.");

            // TODO: CodeGen includeds the linkage (ExternalLinkage) and only passes the
            // mangled_name if entry is nullptr
            auto fn = create_vast_function(meta_location(function_decl), mangled_name, fty, function_decl);

            if (entry) {
                llvm_unreachable("NYI");
            }

            // TODO: This might not be valid, seems the uniqueing system doesn't make
            // sense for MLIR
            // assert(F->getName().getStringRef() == MangledName && "name was uniqued!");

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
                    llvm_unreachable("NYI");
                }

                // This is the first use or definition of a mangled name. If there is a
                // deferred decl with this name, remember that we need to emit it at the end
                // of the file.
                // FIXME: encapsulate this eventually
                auto &deffered = context().deferred_decls;
                if (auto ddi = deffered.find(mangled_name.name); ddi != deffered.end()) {
                    // Move the potentially referenced deferred decl to the
                    // DeferredDeclsToEmit list, and remove it from DeferredDecls (since we
                    // don't need it anymore).

                    context().add_deferred_decl_to_emit(ddi->second);
                    deffered.erase(ddi);

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
                assert(fn.getFunctionType() == type);
                return fn;
            }

            throw cg::unimplemented("codegen of incomplete function");
        }

        vast_function get_addr_of_function(
            clang::GlobalDecl decl, mlir_type fty, global_emition emit
        ) {
            assert(!emit.for_vtable && "NYI");

            // TODO: is this true for vast?
            assert(!clang::cast< clang::FunctionDecl >(decl.getDecl())->isConsteval() &&
                "consteval function should never be emitted"
            );

            assert(fty && "missing funciton type");
            // TODO: do we need this:
            // if (!type) {
            //     const auto *fn = clang::cast< clang::FunctionDecl >(decl.getDecl());
            //     type = type_conv.get_function_type(fn->getType());
            // }

            assert(!clang::dyn_cast< clang::CXXDestructorDecl >( decl.getDecl() ) && "NYI");


            auto mangled_name = name_mangler().get_mangled_name(decl, acontext().getTargetInfo(), /* module name hash */ "");
            return get_or_create_vast_function(mangled_name, fty, decl, emit);
        }

        // Implelements buildGlobalFunctionDefinition of cir codegen
        operation build_function_prototype(clang::GlobalDecl decl, mlir_type fty) {
            // Get or create the prototype for the function.
            // TODO: Figure out what to do here? llvm uses a GlobalValue for the FuncOp in mlir
            return get_addr_of_function(decl, fty, deferred_emit_definition);
        }

        operation VisitFunctionDecl(const clang::FunctionDecl *decl) {
            InsertionGuard guard(builder());
            auto is_definition = decl->doesThisDeclarationHaveABody();

            // emit definition instead of declaration
            if (!is_definition && decl->getDefinition()) {
                return visit(decl->getDefinition());
            }

            auto is_terminator = [] (auto &op) {
                return op.template hasTrait< mlir::OpTrait::IsTerminator >();
            };

            auto declare_function_params = [&, this] (auto entry) {
                // In MLIR the entry block of the function must have the same
                // argument list as the function itself.
                auto params = llvm::zip(decl->getDefinition()->parameters(), entry->getArguments());
                for (const auto &[arg, earg] : params) {
                    this->declare(arg, earg);
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

                    filter< clang::LabelDecl >(decl->decls(), [this] (auto lab) {
                        this->visit(lab);
                    });

                    visit(decl->getBody());
                }

                // TODO make as pass
                splice_trailing_scopes(fn);

                auto &last_block = fn.getBlocks().back();
                auto &ops        = last_block.getOperations();
                set_insertion_point_to_end(&last_block);

                if (ops.empty() || !is_terminator(ops.back())) {
                    emit_function_terminator(fn);
                }
            };

            llvm::ScopedHashTableScope scope(context().vars);

            auto linkage = hl::get_function_linkage(decl);

            auto fn = declare(decl, [&] () {
                auto loc  = meta_location(decl);
                auto type = visit(decl->getFunctionType()).template cast< mlir::FunctionType >();
                // make function header, that will be later filled with function body
                // or returned as declaration in the case of external function
                return make< hl::FuncOp >(loc, decl->getName(), type, linkage);
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
            return declare(decl, [&] {
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
                    .bind(decl->getUnderlyingDecl()->getName())                 // name
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
            return declare(decl, [&] {
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
            return declare(decl, [&] {
                return this->template make_operation< hl::LabelDeclOp >()
                    .bind(meta_location(decl))  // location
                    .bind(decl->getName())      // name
                    .freeze();
            });
        }

        //
        // Enum Declarations
        //
        operation VisitEnumDecl(const clang::EnumDecl *decl) {
            return declare(decl, [&] {
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
            return declare(decl, [&] {
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

        //
        // Record Declaration
        //
        template< typename Decl >
        operation make_record_decl(const clang::RecordDecl *decl) {
            auto loc  = meta_location(decl);
            auto name = context().decl_name(decl);

            // declare the type first to allow recursive type definitions
            if (!decl->isCompleteDefinition()) {
                return declare(decl, [&] {
                    return this->template make_operation< hl::TypeDeclOp >()
                        .bind(meta_location(decl)) // location
                        .bind(decl->getName())     // name
                        .freeze();
                });
            }

            // generate record definition
            if (decl->field_empty()) {
                return make< Decl >(loc, name);
            }

            auto fields = [&](auto &bld, auto loc) {
                for (auto field : decl->fields()) {
                    auto field_type = field->getType();
                    if (clang::isa< clang::ElaboratedType >(field_type)) {
                        visit(field_type->getAsTagDecl());
                    }
                    visit(field);
                }
            };

            return make< Decl >(loc, name, fields);
        }

        operation VisitRecordDecl(const clang::RecordDecl *decl) {
            if (decl->isUnion()) {
                return make_record_decl< hl::UnionDeclOp >(decl);
            } else {
                return make_record_decl< hl::StructDeclOp >(decl);
            }
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
