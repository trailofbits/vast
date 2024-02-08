// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/ScopedHashTable.h>
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

#include "vast/Dialect/Core/Linkage.hpp"

namespace vast::cg {

    template< typename derived_t >
    struct default_decl_visitor
        : decl_visitor_base< default_decl_visitor< derived_t > >
        , visitor_lens< derived_t, default_decl_visitor >
    {
        using lens = visitor_lens< derived_t, default_decl_visitor >;

        using lens::derived;
        using lens::context;
        using lens::mcontext;
        using lens::acontext;

        using lens::name_mangler;

        using lens::mlir_builder;

        using lens::visit;
        using lens::visit_as_lvalue_type;
        using lens::visit_function_type;

        using lens::make_value_builder;

        using lens::insertion_guard;

        using lens::set_insertion_point_to_start;
        using lens::set_insertion_point_to_end;

        using lens::meta_location;

        using lens::constant;

        using excluded_attr_list = util::type_list<
              clang::WeakAttr
            , clang::SelectAnyAttr
            , clang::CUDAGlobalAttr
        >;

        template< typename op_t, typename... args_t >
        auto make(args_t &&...args) {
            return mlir_builder().template create< op_t >(std::forward< args_t >(args)...);
        }

        auto visit_decl_attrs(const clang::Decl *decl, operation op) {
            // getAttrs on decl without attrs triggers an assertion in clang
            if (decl->hasAttrs()) {
                mlir::NamedAttrList attrs = op->getAttrs();
                for (auto attr : exclude_attrs< excluded_attr_list >(decl->getAttrs())) {
                    auto visited = visit(attr);

                    auto spelling = attr->getSpelling();
                    // Bultin attr doesn't have spelling because it can not be written in code
                    if (auto builtin = clang::dyn_cast< clang::BuiltinAttr >(attr)) {
                        spelling = "builtin";
                    }

                    if (auto prev = attrs.getNamed(spelling)) {
                        VAST_CHECK(visited == prev.value().getValue(), "Conflicting redefinition of attribute {0}", spelling);
                    }

                    attrs.set(spelling, visited);
                }
                op->setAttrs(attrs);
            }
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
            VAST_UNIMPLEMENTED_MSG("unknown storage class");
        }

        hl::TSClass VisitThreadStorageClass(const clang::VarDecl *decl) const {
            switch (decl->getTSCSpec()) {
                case clang::TSCS_unspecified: return hl::TSClass::tsc_none;
                case clang::TSCS___thread: return hl::TSClass::tsc_gnu_thread;
                case clang::TSCS_thread_local: return hl::TSClass::tsc_cxx_thread;
                case clang::TSCS__Thread_local: return hl::TSClass::tsc_c_thread;
            }
            VAST_UNIMPLEMENTED_MSG("unknown thread storage class");
        }

        operation VisitVarDecl(const clang::VarDecl *decl) {
            auto var_decl = context().declare(decl, [&] {
                auto type = decl->getType();
                bool has_allocator = type->isVariableArrayType();
                bool has_init = decl->getInit();
                auto array_allocator = [decl, this](auto &bld, auto loc) {
                    if (auto type = clang::dyn_cast< clang::VariableArrayType >(decl->getType())) {
                        make_value_builder(type->getSizeExpr())(bld, loc);
                    }
                };

                auto var = this->template make_operation< hl::VarDeclOp >()
                    .bind(meta_location(decl))                                  // location
                    .bind(visit_as_lvalue_type(type))                           // type
                    .bind(context().decl_name(decl->getUnderlyingDecl()))       // name
                    // The initializer region is filled later as it might
                    // have references to the VarDecl we are currently
                    // visiting - int *x = malloc(sizeof(*x))
                    .bind_region_if(has_init, [](auto, auto){})                 // initializer
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

            if (decl->hasInit()) {
                auto declared = mlir::dyn_cast< hl::VarDeclOp >(var_decl);
                auto &initializer = declared.getInitializer();

                assert(initializer.hasOneBlock());
                // If the initializer isn't empty it means that we are revisiting
                // already declared variable. Skip the initialization as we
                // only want to return the variable.
                if (initializer.front().empty()) {
                    auto guard = insertion_guard();
                    set_insertion_point_to_start(&declared.getInitializer());

                    auto value_builder = make_value_builder(decl->getInit());
                    value_builder(mlir_builder(), meta_location(decl));
                }
            }

            return var_decl;
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
            return derived().template make_scoped< TranslationUnitScope >(loc, [&] {
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
            VAST_UNIMPLEMENTED_MSG("unknown access specifier");
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

    template< typename derived_t >
    struct decl_visitor_with_attrs : default_decl_visitor< derived_t >
    {
        using base = default_decl_visitor< derived_t >;

        using base::visit_decl_attrs;

        auto Visit(const clang::Decl *decl) -> operation {
            if (auto op = base::Visit(decl)) {
                visit_decl_attrs(decl, op);
                return op;
            }
            return {};
        }
    };
} // namespace vast::cg
