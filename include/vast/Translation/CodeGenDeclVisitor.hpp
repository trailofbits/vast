// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/DeclVisitor.h>
#include <clang/AST/Attr.h>
VAST_UNRELAX_WARNINGS

#include "vast/Translation/CodeGenMeta.hpp"
#include "vast/Translation/CodeGenBuilder.hpp"
#include "vast/Translation/CodeGenVisitorBase.hpp"
#include "vast/Translation/CodeGenVisitorLens.hpp"
#include "vast/Translation/Util.hpp"

namespace vast::hl {

    template< typename Derived >
    struct CodeGenDeclVisitorMixin
        : clang::ConstDeclVisitor< CodeGenDeclVisitorMixin< Derived >, Operation* >
        , CodeGenVisitorLens< CodeGenDeclVisitorMixin< Derived >, Derived >
        , CodeGenBuilderMixin< CodeGenDeclVisitorMixin< Derived >, Derived >
    {
        using LensType = CodeGenVisitorLens< CodeGenDeclVisitorMixin< Derived >, Derived >;

        using LensType::derived;
        using LensType::context;
        using LensType::mcontext;

        using LensType::meta_location;

        using LensType::visit;
        using LensType::visit_as_lvalue_type;

        using Builder = CodeGenBuilderMixin< CodeGenDeclVisitorMixin< Derived >, Derived >;

        using Builder::op_builder;

        using Builder::make_value_builder;
        using Builder::start_scoped_builder;

        using Builder::set_insertion_point_to_start;
        using Builder::set_insertion_point_to_end;

        using Builder::constant;

        using Builder::define_type;
        using Builder::declare_type;
        using Builder::declare_enum;
        using Builder::declare_enum_constant;

        template< typename Op, typename... Args >
        auto create(Args &&...args) {
            return op_builder().template create< Op >(std::forward< Args >(args)...);
        }

        Operation* VisitFunctionDecl(const clang::FunctionDecl *decl) {
            auto name = decl->getName();
            auto is_definition = decl->doesThisDeclarationHaveABody();

            // emit definition instead of declaration
            if (!is_definition && decl->getDefinition()) {
                return visit(decl->getDefinition());
            }

            // return already seen definition
            if (auto fn = context().functions.lookup(name)) {
                return fn;
            }

            auto builder_scope = start_scoped_builder();
            llvm::ScopedHashTableScope scope(context().vars);

            auto loc  = meta_location(decl);
            auto type = visit(decl->getFunctionType()).template cast< mlir::FunctionType >();
            // create function header, that will be later filled with function body
            // or returned as declaration in the case of external function
            auto fn = create< mlir::FuncOp >(loc, name, type);
            if (failed(context().functions.declare(name, fn))) {
                context().error("error: multiple declarations of a same function" + name);
            }

            if (!is_definition) {
                fn.setVisibility( mlir::FuncOp::Visibility::Private );
                return fn;
            }

            // emit function body
            auto entry = fn.addEntryBlock();
            set_insertion_point_to_start(entry);

            if (decl->hasBody()) {
                // In MLIR the entry block of the function must have the same
                // argument list as the function itself.
                auto params = llvm::zip(decl->getDefinition()->parameters(), entry->getArguments());
                for (const auto &[arg, earg] : params) {
                    if (failed(context().vars.declare(arg, earg)))
                        context().error("error: multiple declarations of a same symbol" + arg->getName());
                }

                visit(decl->getBody());
            }

            // TODO make as pass
            splice_trailing_scopes(fn);

            auto &last_block = fn.getBlocks().back();
            auto &ops        = last_block.getOperations();
            set_insertion_point_to_end(&last_block);

            if (ops.empty() || !ops.back().template hasTrait< mlir::OpTrait::IsTerminator >()) {
                if (decl->getReturnType()->isVoidType()) {
                    create< ReturnOp >(loc);
                } else {
                    if (decl->isMain()) {
                        // return zero if no return is present in main
                        auto zero = constant(loc, type.getResult(0), apint(0));
                        create< ReturnOp >(loc, zero);
                    } else {
                        create< UnreachableOp >(loc);
                    }
                }
            }

            return fn;
        }

        //
        // Variable Declration
        //

        StorageClass VisitStorageClass(const clang::VarDecl *decl) {
            switch (decl->getStorageClass()) {
                case clang::SC_None: return StorageClass::sc_none;
                case clang::SC_Auto: return StorageClass::sc_auto;
                case clang::SC_Static: return StorageClass::sc_static;
                case clang::SC_Extern: return StorageClass::sc_extern;
                case clang::SC_PrivateExtern: return StorageClass::sc_private_extern;
                case clang::SC_Register: return StorageClass::sc_register;
            }
        }

        TSClass VisitThreadStorageClass(const clang::VarDecl *decl) {
            switch (decl->getTSCSpec()) {
                case clang::TSCS_unspecified: return TSClass::tsc_none;
                case clang::TSCS___thread: return TSClass::tsc_gnu_thread;
                case clang::TSCS_thread_local: return TSClass::tsc_cxx_thread;
                case clang::TSCS__Thread_local: return TSClass::tsc_c_thread;
            }
        }

        Operation* VisitVarDecl(const clang::VarDecl *decl) {
            auto type = decl->getType();
            bool has_allocator = type->isVariableArrayType();
            bool has_init =  decl->getInit();

            auto initializer = make_value_builder(decl->getInit());

            auto array_allocator = [decl, this](auto &bld, auto loc) {
                if (auto type = clang::dyn_cast< clang::VariableArrayType >(decl->getType())) {
                    make_value_builder(type->getSizeExpr())(bld, loc);
                }
            };

            auto var = this->template make_operation< VarDecl >()
                .bind(meta_location(decl))                                  // location
                .bind(visit_as_lvalue_type(type))                           // type
                .bind(decl->getUnderlyingDecl()->getName())                 // name
                .bind_region_if(has_init, std::move(initializer))           // initializer
                .bind_region_if(has_allocator, std::move(array_allocator))  // array allocator
                .freeze();

            if (auto sc = VisitStorageClass(decl); sc != StorageClass::sc_none) {
                var.setStorageClass(sc);
            }

            if (auto tsc = VisitThreadStorageClass(decl); tsc != TSClass::tsc_none) {
                var.setThreadStorageClass(tsc);
            }

            if (failed(context().vars.declare(decl, var))) {
                context().error("error: multiple declarations of a same symbol " + decl->getName());
            }

            return var;
        }

        Operation* VisitParmVarDecl(const clang::ParmVarDecl *decl) {
            if (auto var = context().vars.lookup(decl))
                return var.getDefiningOp();
            context().error("error: missing parameter declaration " + decl->getName());
            return nullptr;
        }

        // Operation* VisitImplicitParamDecl(const clang::ImplicitParamDecl *decl)

        // Operation* VisitLinkageSpecDecl(const clang::LinkageSpecDecl *decl)

        Operation* VisitTranslationUnitDecl(const clang::TranslationUnitDecl *tu) {
            auto loc = meta_location(tu);
            return this->template make_scoped< TranslationUnitScope >(loc, [&] {
                for (const auto &decl : tu->decls()) {
                    visit(decl);
                }
            });
        }

        // Operation* VisitTypedefNameDecl(const clang::TypedefNameDecl *decl)

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

        static inline std::string parse_annotation(AContext &ctx, const clang::Attr *attr) {
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
            if (from->hasAttrs()) {
                auto &actx = context().getASTContext();
                for (auto attr: from->getAttrs()) {
                    auto annot = mlir::StringAttr::get(to->getContext(), parse_annotation(actx, attr));
                    to->setAttr("annotation", AnnotationAttr::get(annot));
                }
            }
        }

        Operation* VisitTypedefDecl(const clang::TypedefDecl *decl) {
            auto name = decl->getName();

            auto type = [&]() -> mlir::Type {
                auto underlying = decl->getUnderlyingType();
                if (auto fty = clang::dyn_cast< clang::FunctionType >(underlying)) {
                    return visit(fty);
                }

                // predeclare named underlying types if necessery
                walk_type(underlying, [&](auto ty) {
                    if (auto tag = clang::dyn_cast< clang::TagType >(ty)) {
                        visit(tag->getDecl());
                        return true; // stop recursive walk
                    }

                    return false;
                });

                return visit(underlying);
            }();

            auto def = define_type(meta_location(decl), type, name);
            attach_attributes(decl /* from */, def /* to */);
            return def;
        }

        // Operation* VisitTypeAliasDecl(const clang::TypeAliasDecl *decl)

        // Operation* VisitLabelDecl(const clang::LabelDecl *decl)

        //
        // Enum Declarations
        //
        Operation* VisitEnumDecl(const clang::EnumDecl *decl) {
            auto name = context().decl_name(decl);
            auto type = visit(decl->getIntegerType());

            auto constants = [&] (auto &bld, auto loc) {
                for (auto con : decl->enumerators()) {
                    visit(con);
                }
            };

            return declare_enum(meta_location(decl), name, type, constants);
        }

        Operation* VisitEnumConstantDecl(const clang::EnumConstantDecl *decl) {
            auto initializer = make_value_builder(decl->getInitExpr());

            auto enum_constant = this->template make_operation< EnumConstantOp >()
                .bind(meta_location(decl))                              // location
                .bind(decl->getName())                                  // name
                .bind(decl->getInitVal())                               // value
                .bind_if(decl->getInitExpr(), std::move(initializer))   // initializer
                .freeze();

            return declare_enum_constant(enum_constant);
        }

        //
        // Record Declaration
        //
        template< typename Decl >
        Operation* make_record_decl(const clang::RecordDecl *decl) {
            auto loc  = meta_location(decl);
            auto name = context().decl_name(decl);
            // declare the type first to allow recursive type definitions
            if (!decl->isCompleteDefinition()) {
                return declare_type(loc, name);;
            }

            // generate record definition
            if (decl->field_empty()) {
                return create< Decl >(loc, name);
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

            return create< Decl >(loc, name, fields);
        }

        Operation* VisitRecordDecl(const clang::RecordDecl *decl) {
            if (decl->isUnion()) {
                return make_record_decl< UnionDeclOp >(decl);
            } else {
                return make_record_decl< StructDeclOp >(decl);
            }
        }

        Operation* VisitFieldDecl(const clang::FieldDecl *decl) {
            auto loc  = meta_location(decl);
            auto name = context().get_decl_name(decl);

            // define field type if the field defines a new nested type
            if (auto tag = decl->getType()->getAsTagDecl()) {
                if (tag->isThisDeclarationADefinition()) {
                    if (!context().tag_names.count(tag)) {
                        visit(tag);
                    }
                }
            }
            auto type = visit(decl->getType());
            return create< FieldDeclOp >(loc, name, type);
        }
    };

} // namespace vast::hl
