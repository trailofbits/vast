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
#include "vast/Dialect/HighLevel/HighLevelLinkage.hpp"
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
        using LensType::acontext;

        using LensType::meta_location;

        using LensType::visit;
        using LensType::visit_as_lvalue_type;

        using Builder = CodeGenBuilderMixin< CodeGenDeclVisitorMixin< Derived >, Derived >;

        using Builder::op_builder;
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

        Operation* VisitFunctionDecl(const clang::FunctionDecl *decl) {
            InsertionGuard guard(op_builder());
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
                    declare(arg, earg);
                }
            };

            auto emit_function_terminator = [&] (auto fn) {
                auto loc = fn.getLoc();
                if (decl->getReturnType()->isVoidType()) {
                    make< ReturnOp >(loc);
                } else {
                    if (decl->isMain()) {
                        // return zero if no return is present in main
                        auto type = fn.getFunctionType();
                        auto zero = constant(loc, type.getResult(0), apsint(0));
                        make< ReturnOp >(loc, zero);
                    } else {
                        make< UnreachableOp >(loc);
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
                    filter< clang::LabelDecl >(decl->decls(), [&, this] (auto lab) {
                        visit(lab);
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

            auto linkage = get_function_linkage(decl);

            auto fn = declare(decl, [&] () {
                auto loc  = meta_location(decl);
                auto type = visit(decl->getFunctionType()).template cast< mlir::FunctionType >();
                // make function header, that will be later filled with function body
                // or returned as declaration in the case of external function
                return make< FuncOp >(loc, decl->getName(), type, linkage);
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

        StorageClass VisitStorageClass(const clang::VarDecl *decl) const {
            switch (decl->getStorageClass()) {
                case clang::SC_None: return StorageClass::sc_none;
                case clang::SC_Auto: return StorageClass::sc_auto;
                case clang::SC_Static: return StorageClass::sc_static;
                case clang::SC_Extern: return StorageClass::sc_extern;
                case clang::SC_PrivateExtern: return StorageClass::sc_private_extern;
                case clang::SC_Register: return StorageClass::sc_register;
            }
            VAST_UNREACHABLE("unknown storage class");
        }

        TSClass VisitThreadStorageClass(const clang::VarDecl *decl) const {
            switch (decl->getTSCSpec()) {
                case clang::TSCS_unspecified: return TSClass::tsc_none;
                case clang::TSCS___thread: return TSClass::tsc_gnu_thread;
                case clang::TSCS_thread_local: return TSClass::tsc_cxx_thread;
                case clang::TSCS__Thread_local: return TSClass::tsc_c_thread;
            }
            VAST_UNREACHABLE("unknown storage class");
        }

        Operation* VisitVarDecl(const clang::VarDecl *decl) {
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

                auto var = this->template make_operation< VarDeclOp >()
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

                return var;
            }).getDefiningOp();
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
            if (!from->hasAttrs())
                return;

            auto &actx = acontext();
            for (auto attr: from->getAttrs()) {
                auto annot = mlir::StringAttr::get(to->getContext(), parse_annotation(actx, attr));
                to->setAttr("annotation", AnnotationAttr::get(annot));
            }
        }

        Operation* VisitTypedefDecl(const clang::TypedefDecl *decl) {
            return declare(decl, [&] {
                auto type = [&, this] () -> mlir::Type {
                    auto underlying = decl->getUnderlyingType();
                    if (auto fty = clang::dyn_cast< clang::FunctionType >(underlying)) {
                        return visit(fty);
                    }

                    // predeclare named underlying types if necessery
                    walk_type(underlying, [&, this](auto ty) {
                        if (auto tag = clang::dyn_cast< clang::TagType >(ty)) {
                            visit(tag->getDecl());
                            return true; // stop recursive walk
                        }

                        return false;
                    });

                    return visit(underlying);
                };

                // create typedef operation
                auto def = this->template make_operation< TypeDefOp >()
                    .bind(meta_location(decl)) // location
                    .bind(decl->getName())     // name
                    .bind(type())              // type
                    .freeze();

                attach_attributes(decl /* from */, def /* to */);
                return def;
            });
        }

        // Operation* VisitTypeAliasDecl(const clang::TypeAliasDecl *decl)

        Operation* VisitLabelDecl(const clang::LabelDecl *decl) {
            return declare(decl, [&] {
                return this->template make_operation< LabelDeclOp >()
                    .bind(meta_location(decl))  // location
                    .bind(decl->getName())      // name
                    .freeze();
            });
        }

        //
        // Enum Declarations
        //
        Operation* VisitEnumDecl(const clang::EnumDecl *decl) {
            return declare(decl, [&] {
                auto constants = [&] (auto &bld, auto loc) {
                    for (auto con : decl->enumerators()) {
                        visit(con);
                    }
                };

                return this->template make_operation< EnumDeclOp >()
                    .bind(meta_location(decl))                              // location
                    .bind(decl->getName())                                  // name
                    .bind(visit(decl->getIntegerType()))                    // type
                    .bind(constants)                                        // constants
                    .freeze();
            });
        }

        Operation* VisitEnumConstantDecl(const clang::EnumConstantDecl *decl) {
            return declare(decl, [&] {
                auto initializer = make_value_builder(decl->getInitExpr());

                auto type = visit(decl->getType());

                return this->template make_operation< EnumConstantOp >()
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
        Operation* make_record_decl(const clang::RecordDecl *decl) {
            auto loc  = meta_location(decl);
            auto name = context().decl_name(decl);

            // declare the type first to allow recursive type definitions
            if (!decl->isCompleteDefinition()) {
                return declare(decl, [&] {
                    return this->template make_operation< TypeDeclOp >()
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

        Operation* VisitRecordDecl(const clang::RecordDecl *decl) {
            if (decl->isUnion()) {
                return make_record_decl< UnionDeclOp >(decl);
            } else {
                return make_record_decl< StructDeclOp >(decl);
            }
        }

        Operation* VisitFieldDecl(const clang::FieldDecl *decl) {
            // define field type if the field defines a new nested type
            if (auto tag = decl->getType()->getAsTagDecl()) {
                if (tag->isThisDeclarationADefinition()) {
                    if (!context().tag_names.count(tag)) {
                        visit(tag);
                    }
                }
            }

            return this->template make_operation< FieldDeclOp >()
                .bind(meta_location(decl))              // location
                .bind(context().get_decl_name(decl))    // name
                .bind(visit(decl->getType()))           // type
                .bind(decl->getBitWidth() ? context().u32(decl->getBitWidthValue(acontext())) : nullptr) // bitfield
                .freeze();

        }
    };

} // namespace vast::hl
