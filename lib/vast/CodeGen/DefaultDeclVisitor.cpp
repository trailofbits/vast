// Copyright (c) 2024, Trail of Bits, Inc.

#include "vast/CodeGen/DefaultDeclVisitor.hpp"

#include "vast/Dialect/Core/Linkage.hpp"

#include "vast/CodeGen/CodeGenFunction.hpp"

#include "vast/Util/Maybe.hpp"
#include "vast/CodeGen/Util.hpp"

namespace vast::cg
{
    bool unsupported(const clang_function *decl) {
        if (clang::isa< clang::CXXConstructorDecl >(decl)) {
            VAST_REPORT("Unsupported constructor declaration");
            return true;
        }

        if (clang::isa< clang::CXXDestructorDecl >(decl)) {
            VAST_REPORT("Unsupported destructor declaration");
            return true;
        }

        if (llvm::dyn_cast< clang::CXXMethodDecl >(decl)) {
            VAST_REPORT("Unsupported C++ method declaration");
            return true;
        }

        if (decl->isDefaulted()) {
            VAST_REPORT("Unsupported defaulted functions");
            return true;
        }

        if (decl->getAttr< clang::ConstructorAttr >()) {
            VAST_REPORT("Unsupported function with constructor attribute");
            return true;
        }

        if (decl->getAttr< clang::DestructorAttr >()) {
            VAST_REPORT("Unsupported function with destructor attribute");
            return true;
        }

        if (decl->isMultiVersion()) {
            VAST_REPORT("Unsupported function with multi-version attribute");
            return true;
        }

        if (decl->usesSEHTry()) {
            VAST_REPORT("Unsupported function with SEH try attribute");
            return true;
        }

        if (decl->getAttr< clang::PatchableFunctionEntryAttr >()) {
            VAST_REPORT("Unsupported function with patchable function entry attribute");
            return true;
        }

        if (decl->hasAttr< clang::CFICanonicalJumpTableAttr >()) {
            VAST_REPORT("Unsupported function with CFI canonical jump table attribute");
            return true;
        }

        if (decl->hasAttr< clang::NoProfileFunctionAttr >()) {
            VAST_REPORT("Unsupported function with no profile function attribute");
            return true;
        }

        if (decl->hasAttr< clang::MinVectorWidthAttr >()) {
            VAST_REPORT("Unsupported function with min vector width attribute");
            return true;
        }

        if (decl->hasAttr< clang::NoDebugAttr >()) {
            VAST_REPORT("Unsupported function with no debug attribute");
            return true;
        }

        if (decl->hasAttr< clang::CUDAGlobalAttr >()) {
            VAST_REPORT("Unsupported function with no cuda global attribute");
            return true;
        }

        auto &actx = decl->getASTContext();
        auto &lang = actx.getLangOpts();

        if (lang.OpenCL || lang.OpenMPIsTargetDevice || lang.CUDA || lang.CUDAIsDevice || lang.OpenMP) {
            VAST_REPORT("Unsupported function declaration in the language");
            return true;
        }

        // Add no-jump-tables value.
        // if (opts.codegen.NoUseJumpTables) {
        //     return false;
        // }

        // // Add no-inline-line-tables value.
        // if (opts.codegen.NoInlineLineTables) {
        //     return false;
        // }

        // // TODO: Add profile-sample-accurate value.
        // if (opts.codegen.ProfileSampleAccurate) {
        //     return false;
        // }

        // Detect the unusual situation where an inline version is shadowed by a
        // non-inline version. In that case we should pick the external one
        // everywhere. That's GCC behavior too. Unfortunately, I cannot find a way
        // to detect that situation before we reach codegen, so do some late
        // replacement.
        for (const auto *prev = decl->getPreviousDecl(); prev; prev = prev->getPreviousDecl()) {
            if (LLVM_UNLIKELY(prev->isInlineBuiltinDeclaration())) {
                VAST_REPORT("Unsupported inline builtin declaration");
                return true;
            }
        }


        return false;
    }

    //
    // Function Declaration
    //

    mlir_visibility get_function_visibility(const clang_function *decl, linkage_kind linkage) {
        if (decl->isThisDeclarationADefinition()) {
            return core::get_visibility_from_linkage(linkage);
        }
        if (decl->doesDeclarationForceExternallyVisibleDefinition()) {
            return mlir_visibility::Public;
        }
        return mlir_visibility::Private;
    }

    operation default_decl_visitor::visit_prototype(const clang_function *decl) {
        if (unsupported(decl)) {
            return {};
        }

        auto set_visibility = [&] (vast_function fn) {
            auto visibility = get_function_visibility(decl, fn.getLinkage());
            mlir::SymbolTable::setSymbolVisibility(fn, visibility);
            return fn;
        };

        auto set_attrs = [&] (vast_function fn) {
            auto attrs = visit_attrs(decl);
            attrs.append(fn->getAttrs());
            fn->setAttrs(attrs);
            return fn;
        };

        return bld.compose< vast_function >()
            .bind(self.location(decl))
            .bind(self.symbol(decl))
            .bind_dyn_cast< vast_function_type >(self.visit(decl->getFunctionType(), decl->isVariadic()))
            .bind(core::get_function_linkage(decl))
            .freeze_as_maybe() // construct vast_function
            .transform(set_visibility)
            .transform(set_attrs)
            .take();
    }

    mlir_attr_list default_decl_visitor::visit_attrs(const clang_function *decl) {
        if (!decl->hasAttrs()) {
            return {};
        }

        // These are already handled by linkage attributes
        using excluded_attr_list = util::type_list<
              clang::WeakAttr
            , clang::SelectAnyAttr
            , clang::CUDAGlobalAttr
        >;

        mlir_attr_list attrs;
        for (auto attr : exclude_attrs< excluded_attr_list >(decl->getAttrs())) {
            auto visited = self.visit(attr);

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

        return attrs;
    }

    //
    // Variable Declaration
    //

    hl::StorageClass storage_class(const clang_var_decl *decl) {
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

    hl::TSClass thread_storage_class(const clang_var_decl *decl) {
        switch (decl->getTSCSpec()) {
            case clang::TSCS_unspecified: return hl::TSClass::tsc_none;
            case clang::TSCS___thread: return hl::TSClass::tsc_gnu_thread;
            case clang::TSCS_thread_local: return hl::TSClass::tsc_cxx_thread;
            case clang::TSCS__Thread_local: return hl::TSClass::tsc_c_thread;
        }
        VAST_UNIMPLEMENTED_MSG("unknown thread storage class");
    }

    bool unsupported(const clang_var_decl *decl) {
        auto &actx = decl->getASTContext();
        auto lang = actx.getLangOpts();

        if (decl->hasInit() && lang.CPlusPlus) {
            VAST_REPORT("Unsupported variable declaration with initializer in C++");
            return false;
        }

        if (lang.OpenCL || lang.OpenMPIsTargetDevice || lang.CUDA || lang.OpenMP) {
            VAST_REPORT("Unsupported variable declaration in the language");
            return false;
        }

        if (decl->needsDestruction(actx) == clang::QualType::DK_cxx_destructor) {
            VAST_REPORT("Unsupported variable declaration with destructor");
            return false;
        }

        return false;
    }

    operation default_decl_visitor::VisitVarDecl(const clang::VarDecl *decl) {
        if (unsupported(decl)) {
            return {};
        }

        bool has_allocator = decl->getType()->isVariableArrayType();
        bool has_init = decl->getInit();

        auto array_allocator = [decl](auto &state, auto loc) {
            if (auto type = clang::dyn_cast< clang::VariableArrayType >(decl->getType())) {
                VAST_UNIMPLEMENTED; // emit(type->getSizeExpr(), state, loc);
            }
        };

        auto set_storage_classes = [&] (auto var) {
            if (auto sc = storage_class(decl); sc != hl::StorageClass::sc_none) {
                var.setStorageClass(sc);
            }

            if (auto tsc = thread_storage_class(decl); tsc != hl::TSClass::tsc_none) {
                var.setThreadStorageClass(tsc);
            }

            return var;
        };

        auto var = maybe_declare([&] {
            return bld.compose< hl::VarDeclOp >()
                .bind(self.location(decl))
                .bind(self.visit_as_lvalue_type(decl->getType()))
                .bind(self.symbol(decl))
                // The initializer region is filled later as it might
                // have references to the VarDecl we are currently
                // visiting - int *x = malloc(sizeof(*x))
                .bind_if(has_init, [](auto, auto){})
                .bind_if(has_allocator, std::move(array_allocator))
                .freeze_as_maybe() // construct global
                .transform(set_storage_classes)
                .take();
        });

        if (decl->hasInit()) {
            fill_init(decl->getInit(), var);
        }

        return var;
    }

    void default_decl_visitor::fill_init(const clang_expr *init, hl::VarDeclOp var) {
        auto &region = var.getInitializer();
        VAST_ASSERT(region.hasOneBlock());
        auto _ = bld.scoped_insertion_at_start(&region);

        bld.compose< hl::ValueYieldOp >()
            .bind(self.location(init))
            .bind_transform(self.visit(init), first_result)
            .freeze();
    }

    operation default_decl_visitor::VisitParmVarDecl(const clang::ParmVarDecl *decl) {
        if (auto name = self.symbol(decl)) {
            if (auto var = self.scope.lookup_var(name.value())) {
                return var.getDefiningOp();
            }
        }

        return {};
    }

    operation default_decl_visitor::VisitImplicitParamDecl(const clang::ImplicitParamDecl */* decl */) {
        return {};
    }

    operation default_decl_visitor::VisitLinkageSpecDecl(const clang::LinkageSpecDecl */* decl */) {
        return {};
    }

    operation default_decl_visitor::VisitFunctionDecl(const clang::FunctionDecl *decl) {
        auto gen = mk_scoped_generator< function_generator >(self.scope, bld, self);
        return gen.emit(decl);
    }

    operation default_decl_visitor::VisitTranslationUnitDecl(const clang::TranslationUnitDecl *decl) {
        return {};
    }

    operation default_decl_visitor::VisitTypedefNameDecl(const clang::TypedefNameDecl *decl) {
        return {};
    }

    operation default_decl_visitor::VisitTypedefDecl(const clang::TypedefDecl *decl) {
        return maybe_declare([&] {
            return bld.compose< hl::TypeDefOp >()
                .bind(self.location(decl))
                .bind(self.symbol(decl))
                .bind(self.visit(decl->getUnderlyingType()))
                .freeze();
        });
    }

    operation default_decl_visitor::VisitTypeAliasDecl(const clang::TypeAliasDecl *decl) {
        return {};
    }

    operation default_decl_visitor::VisitLabelDecl(const clang::LabelDecl *decl) {
        return maybe_declare([&] {
            return bld.compose< hl::LabelDeclOp >()
                .bind(self.location(decl))
                .bind(self.symbol(decl))
                .freeze();
        });
    }

    operation default_decl_visitor::VisitEmptyDecl(const clang::EmptyDecl *decl) {
        return bld.compose< hl::EmptyDeclOp >().bind(self.location(decl)).freeze();
    }

    operation default_decl_visitor::VisitEnumDecl(const clang::EnumDecl *decl) {
        // TODO deal with incomplete decls
        return maybe_declare([&] {
            if (!decl->isComplete()) {
                return bld.compose< hl::EnumDeclOp >()
                    .bind(self.location(decl))
                    .bind(self.symbol(decl))
                    .freeze();
            }

            auto constants = [&] (auto &bld, auto loc) {
                for (auto con : decl->enumerators()) {
                    visit(con);
                }
            };

            return bld.compose< hl::EnumDeclOp >()
                .bind(self.location(decl))
                .bind(self.symbol(decl))
                .bind(self.visit(decl->getIntegerType()))
                .bind(constants)
                .freeze();
        });
    }

    operation default_decl_visitor::VisitEnumConstantDecl(const clang::EnumConstantDecl *decl) {
        return maybe_declare([&] {
            auto initializer = [&] (auto & /* bld */, auto loc) {
                bld.compose< hl::ValueYieldOp >()
                    .bind(loc)
                    .bind_transform(self.visit(decl->getInitExpr()), first_result)
                    .freeze();
            };

            return bld.compose< hl::EnumConstantOp >()
                .bind(self.location(decl))
                .bind(self.symbol(decl))
                .bind(self.visit(decl->getType()))
                .bind(decl->getInitVal())
                .bind_if(decl->getInitExpr(), std::move(initializer))
                .freeze();
        });
    }

    operation default_decl_visitor::VisitRecordDecl(const clang::RecordDecl *decl) {
        if (!decl->isCompleteDefinition()) {
            return mk_incomplete_decl(decl);
        }

        if (decl->isUnion()) {
            return mk_record_decl< hl::UnionDeclOp >(decl);
        } else {
            return mk_record_decl< hl::StructDeclOp >(decl);
        }
    }

    operation default_decl_visitor::VisitCXXRecordDecl(const clang::CXXRecordDecl *decl) {
        return {};
    }

    operation default_decl_visitor::VisitAccessSpecDecl(const clang::AccessSpecDecl *decl) {
        return {};
    }

    operation default_decl_visitor::VisitFieldDecl(const clang::FieldDecl *decl)  {
        // define field type if the field defines a new nested type
        if (auto tag = decl->getType()->getAsTagDecl()) {
            if (tag->isThisDeclarationADefinition()) {
                if (auto symbol = self.symbol(tag)) {
                    if (!is_declared_type(self.symbol(tag).value())) {
                        visit(tag);
                    }
                }
            }
        }

        auto visit_bitfield = [&] {
            auto &actx = decl->getASTContext();
            return decl->getBitWidth() ? bld.u32(decl->getBitWidthValue(actx)) : nullptr;
        };

        return maybe_declare([&] {
            return bld.compose< hl::FieldDeclOp >()
                .bind(self.location(decl))
                .bind(self.symbol(decl))
                .bind(self.visit(decl->getType()))
                .bind(visit_bitfield())
                .freeze();
        });
    }

    operation default_decl_visitor::mk_incomplete_decl(const clang::RecordDecl *decl) {
        return maybe_declare([&] {
            return bld.compose< hl::TypeDeclOp >()
                .bind(self.location(decl))
                .bind(self.symbol(decl))
                .freeze();
        });
    }

} // namespace vast::hl
