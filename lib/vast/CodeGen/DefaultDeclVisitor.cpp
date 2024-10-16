// Copyright (c) 2024, Trail of Bits, Inc.

#include "vast/CodeGen/DefaultDeclVisitor.hpp"
#include "vast/CodeGen/CodeGenFunction.hpp"
#include "vast/CodeGen/DefaultTypeVisitor.hpp"
#include "vast/Dialect/Core/Linkage.hpp"
#include "vast/Util/Maybe.hpp"
#include <utility>

namespace vast::cg {
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

        if (lang.OpenCL || lang.OpenMPIsTargetDevice || lang.CUDA || lang.CUDAIsDevice
            || lang.OpenMP)
        {
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

    operation default_decl_visitor::visit_prototype(const clang_function *decl) {
        if (unsupported(decl)) {
            return {};
        }

        return bld.compose< vast_function >()
            .bind(self.location(decl))
            .bind(self.symbol(decl))
            .bind_dyn_cast< vast_function_type >(
                visit_function_type(self, mctx, decl->getFunctionType(), decl->isVariadic())
            )
            .bind_always(core::get_function_linkage(decl))
            .freeze_as_maybe() // construct vast_function
            .transform([&](auto fn) { return set_visibility(decl, fn); })
            .take();
    }

    //
    // Variable Declaration
    //

    core::StorageClass storage_class(const clang_var_decl *decl) {
        switch (decl->getStorageClass()) {
            case clang::SC_None:
                return core::StorageClass::sc_none;
            case clang::SC_Auto:
                return core::StorageClass::sc_auto;
            case clang::SC_Static:
                return core::StorageClass::sc_static;
            case clang::SC_Extern:
                return core::StorageClass::sc_extern;
            case clang::SC_PrivateExtern:
                return core::StorageClass::sc_private_extern;
            case clang::SC_Register:
                return core::StorageClass::sc_register;
        }
        VAST_UNIMPLEMENTED_MSG("unknown storage class");
    }

    core::TSClass thread_storage_class(const clang_var_decl *decl) {
        switch (decl->getTSCSpec()) {
            case clang::TSCS_unspecified:
                return core::TSClass::tsc_none;
            case clang::TSCS___thread:
                return core::TSClass::tsc_gnu_thread;
            case clang::TSCS_thread_local:
                return core::TSClass::tsc_cxx_thread;
            case clang::TSCS__Thread_local:
                return core::TSClass::tsc_c_thread;
        }
        VAST_UNIMPLEMENTED_MSG("unknown thread storage class");
    }

    bool unsupported(const clang_var_decl *decl) {
        auto &actx = decl->getASTContext();
        auto lang  = actx.getLangOpts();

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
        bool has_init      = decl->getInit();
        bool is_global     = !decl->isLocalVarDeclOrParm();
        bool emit_init     = has_init && !(is_global && policy->skip_global_initializer(decl));

        auto array_allocator = [this, decl](auto &state, auto loc) {
            if (auto type = clang::dyn_cast< clang::VariableArrayType >(decl->getType())) {
                mk_value_builder(type->getSizeExpr())(bld, loc);
            }
        };

        auto initializer_builder = [this, decl](auto &state, auto loc) {
            bld.compose< hl::ValueYieldOp >()
                .bind_always(loc)
                .bind_transform(self.visit(decl->getInit()), first_result)
                .freeze();
        };

        auto linkage_builder = [&](const clang::VarDecl *decl) {
            auto gva_linkage = decl->getASTContext().GetGVALinkageForVariable(decl);
            return core::get_declarator_linkage(
                decl,
                gva_linkage,
                decl->getType().isConstQualified(),
                this->policy->get_no_common()
            );
        };

        auto var = maybe_declare(decl, [&] {
            return bld.compose< hl::VarDeclOp >()
                .bind(self.location(decl))
                .bind(visit_as_lvalue_type(self, mctx, decl->getType()))
                .bind(self.symbol(decl))
                .bind_always(storage_class(decl))
                .bind_always(thread_storage_class(decl))
                .bind_always(decl->getType().isConstQualified())
                .bind_choose(is_global, std::optional(linkage_builder(decl)), std::nullopt)
                // FIXME: The initializer region is filled later as it might
                // have references to the VarDecl we are currently
                // visiting - int *x = malloc(sizeof(*x))
                .bind_choose(emit_init, std::move(initializer_builder), std::nullopt)
                .bind_choose(has_allocator, std::move(array_allocator), std::nullopt)
                .freeze();
        });

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
        auto blk = bld.getInsertionBlock();
        if (auto fn = mlir::dyn_cast< core::function_op_interface >(blk->getParentOp())) {
            auto param_index = decl->getFunctionScopeIndex();
            return bld.compose< hl::ParmVarDeclOp >()
                .bind(self.location(decl))
                .bind(self.symbol(decl))
                .bind(fn.getArgument(param_index))
                .freeze();
        }

        return {};
    }

    operation default_decl_visitor::VisitImplicitParamDecl(
        const clang::ImplicitParamDecl * /* decl */
    ) {
        return {};
    }

    operation default_decl_visitor::VisitLinkageSpecDecl(
        const clang::LinkageSpecDecl * /* decl */
    ) {
        return {};
    }

    operation default_decl_visitor::VisitFunctionDecl(const clang::FunctionDecl *decl) {
        auto gen   = mk_scoped_generator< function_generator >(self.scope, bld, self);
        gen.policy = policy;
        return gen.emit(decl);
    }

    operation
    default_decl_visitor::VisitTranslationUnitDecl(const clang::TranslationUnitDecl *decl) {
        return bld.compose< hl::TranslationUnitOp >()
            .bind(self.location(decl))
            .bind_always(mk_decl_context_builder(decl))
            .freeze();
    }

    operation default_decl_visitor::VisitTypedefNameDecl(const clang::TypedefNameDecl *decl) {
        if (auto ty = clang::dyn_cast< clang::TypedefDecl >(decl)) {
            return VisitTypedefDecl(ty);
        }

        if (auto ty = clang::dyn_cast< clang::TypeAliasDecl >(decl)) {
            return VisitTypeAliasDecl(ty);
        }

        return {};
    }

    operation default_decl_visitor::VisitTypedefDecl(const clang::TypedefDecl *decl) {
        return maybe_declare(decl, [&] {
            return bld.compose< hl::TypeDefOp >()
                .bind(self.location(decl))
                .bind(self.symbol(decl))
                .bind(self.visit(decl->getUnderlyingType()))
                .freeze();
        });
    }

    operation default_decl_visitor::VisitTypeAliasDecl(const clang::TypeAliasDecl *decl) {
        return maybe_declare(decl, [&] {
            return bld.compose< hl::TypeAliasOp >()
                .bind(self.location(decl))
                .bind(self.symbol(decl))
                .bind(self.visit(decl->getUnderlyingType()))
                .freeze();
        });
        return {};
    }

    operation default_decl_visitor::VisitLabelDecl(const clang::LabelDecl *decl) {
        if (auto symbol = self.symbol(decl)) {
            if (auto label = self.scope.lookup_label(decl)) {
                return label;
            }
        }

        return maybe_declare(decl, [&] {
            return bld.compose< hl::LabelDeclOp >()
                .bind(self.location(decl))
                .bind(self.symbol(decl))
                .freeze();
        });
    }

    operation default_decl_visitor::VisitEmptyDecl(const clang::EmptyDecl *decl) {
        return bld.compose< hl::EmptyDeclOp >().bind(self.location(decl)).freeze();
    }

    void default_decl_visitor::fill_enum_constants(const clang::EnumDecl *decl) {
        for (auto con : decl->enumerators()) {
            self.visit(con);
        }
    }

    operation default_decl_visitor::VisitEnumDecl(const clang::EnumDecl *decl) {
        if (auto symbol = self.symbol(decl)) {
            if (auto op = self.scope.lookup_type(decl)) {
                auto enum_decl = mlir::cast< hl::EnumDeclOp >(op);
                // Fill in the enum constants if the enum was predeclared
                if (decl->isComplete() && !enum_decl.isComplete()) {
                    auto _ = bld.scoped_insertion_at_start(&enum_decl.getConstantsBlock());
                    enum_decl.setType(self.visit(decl->getIntegerType()));
                    fill_enum_constants(decl);
                }

                return enum_decl;
            }
        }

        return maybe_declare(decl, [&] {
            if (!decl->isComplete()) {
                return bld.compose< hl::EnumDeclOp >()
                    .bind(self.location(decl))
                    .bind(self.symbol(decl))
                    .freeze();
            }

            auto constants = [&](auto &bld, auto loc) { fill_enum_constants(decl); };

            return bld.compose< hl::EnumDeclOp >()
                .bind(self.location(decl))
                .bind(self.symbol(decl))
                .bind(self.visit(decl->getIntegerType()))
                .bind_always(constants)
                .freeze();
        });
    }

    operation default_decl_visitor::VisitEnumConstantDecl(const clang::EnumConstantDecl *decl) {
        return maybe_declare(decl, [&] {
            auto initializer = [&](auto & /* bld */, auto loc) {
                bld.compose< hl::ValueYieldOp >()
                    .bind_always(loc)
                    .bind_transform(self.visit(decl->getInitExpr()), first_result)
                    .freeze();
            };

            return bld.compose< hl::EnumConstantOp >()
                .bind(self.location(decl))
                .bind(self.symbol(decl))
                .bind(self.visit(decl->getType()))
                .bind_always(decl->getInitVal())
                .bind_if(decl->getInitExpr(), std::move(initializer))
                .freeze();
        });
    }

    void default_decl_visitor::fill_decl_members(const clang::RecordDecl *decl) {
        members_scope members_scope(&self.scope);
        // TODO deduplicate lookup mechanism
        for (auto *decl : decl->decls()) {
            // FIXME: Handle IndirectFieldDecl.
            if (clang::isa< clang::IndirectFieldDecl >(decl)) {
                continue;
            }
            self.visit(decl);
        }
    }

    operation default_decl_visitor::VisitRecordDecl(const clang::RecordDecl *decl) {
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

    operation default_decl_visitor::VisitFieldDecl(const clang::FieldDecl *decl) {
        // define field type if the field defines a new nested type
        if (auto tag = decl->getType()->getAsTagDecl()) {
            if (tag->isThisDeclarationADefinition()) {
                if (auto symbol = self.symbol(tag)) {
                    if (!is_declared_type(decl)) {
                        visit(tag);
                    }
                }
            }
        }

        auto visit_bitfield = [&] {
            auto &actx = decl->getASTContext();
            return decl->getBitWidth() ? bld.u32(decl->getBitWidthValue(actx)) : nullptr;
        };

        return maybe_declare(decl, [&] {
            return bld.compose< hl::FieldDeclOp >()
                .bind(self.location(decl))
                .bind(self.symbol(decl))
                .bind(self.visit(decl->getType()))
                .bind_always(visit_bitfield())
                .freeze();
        });
    }

    operation default_decl_visitor::VisitIndirectFieldDecl(const clang::IndirectFieldDecl *decl
    ) {
        return {};
    }

    operation default_decl_visitor::VisitStaticAssertDecl(const clang::StaticAssertDecl *decl) {
        return bld.compose< hl::StaticAssertDecl >()
            .bind(self.location(decl))
            .bind_always(decl->isFailed())
            .bind_always(mk_value_builder(decl->getAssertExpr()))
            .bind_if(decl->getMessage(), mk_value_builder(decl->getMessage()))
            .freeze();
    }

    operation default_decl_visitor::VisitFileScopeAsmDecl(const clang::FileScopeAsmDecl *decl) {
        return bld.compose< hl::FileScopeAsmOp >()
            .bind(self.location(decl))
            .bind(decl->getAsmString()->getString())
            .freeze();
    }

} // namespace vast::cg
