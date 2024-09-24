// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/CodeGenPolicy.hpp"
#include "vast/Util/Warnings.hpp"
#include <memory>

VAST_RELAX_WARNINGS
#include <clang/AST/DeclVisitor.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/ClangVisitorBase.hpp"

#include "vast/CodeGen/CodeGenMetaGenerator.hpp"
#include "vast/CodeGen/SymbolGenerator.hpp"

namespace vast::cg {

    struct default_decl_visitor : decl_visitor_base< default_decl_visitor >
    {
        using base = decl_visitor_base< default_decl_visitor >;
        using base::base;

        using decl_visitor_base< default_decl_visitor >::Visit;

        operation visit(const clang_decl *decl) { return Visit(decl); }
        operation visit_prototype(const clang_function *decl);

        void fill_init(const clang_expr *init, hl::VarDeclOp var);

        operation VisitVarDecl(const clang_var_decl *decl);
        operation VisitParmVarDecl(const clang::ParmVarDecl *decl);
        operation VisitImplicitParamDecl(const clang::ImplicitParamDecl *decl);
        operation VisitLinkageSpecDecl(const clang::LinkageSpecDecl *decl);
        operation VisitFunctionDecl(const clang::FunctionDecl *decl);
        operation VisitTranslationUnitDecl(const clang::TranslationUnitDecl *decl);
        operation VisitTypedefNameDecl(const clang::TypedefNameDecl *decl);
        operation VisitTypedefDecl(const clang::TypedefDecl *decl);
        operation VisitTypeAliasDecl(const clang::TypeAliasDecl *decl);
        operation VisitLabelDecl(const clang::LabelDecl *decl);
        operation VisitEmptyDecl(const clang::EmptyDecl *decl);
        operation VisitEnumDecl(const clang::EnumDecl *decl);
        operation VisitEnumConstantDecl(const clang::EnumConstantDecl *decl);
        operation VisitRecordDecl(const clang::RecordDecl *decl);
        operation VisitCXXRecordDecl(const clang::CXXRecordDecl *decl);
        operation VisitAccessSpecDecl(const clang::AccessSpecDecl *decl);
        operation VisitFieldDecl(const clang::FieldDecl *decl);
        operation VisitIndirectFieldDecl(const clang::IndirectFieldDecl *decl);
        operation VisitStaticAssertDecl(const clang::StaticAssertDecl *decl);
        operation VisitFileScopeAsmDecl(const clang::FileScopeAsmDecl *decl);

        void fill_enum_constants(const clang::EnumDecl *decl);

        void fill_decl_members(const clang::RecordDecl *decl);

        template< typename RecordDeclOp >
        operation mk_record_decl(const clang::RecordDecl *decl);

        std::shared_ptr< codegen_policy > policy;
    };

    template< typename RecordDeclOp >
    operation default_decl_visitor::mk_record_decl(const clang::RecordDecl *decl) {
        if (auto name = self.symbol(decl)) {
            if (auto op = self.scope.lookup_type(decl)) {
                auto record_decl = mlir::cast< RecordDeclOp >(op);
                // Fill in the record members if the record was predeclared
                if (decl->isCompleteDefinition() && !record_decl.isCompleteDefinition()) {
                    auto _ = bld.scoped_insertion_at_start(&record_decl.getFieldsBlock());
                    fill_decl_members(decl);
                }
                return record_decl;
            }
        }

        auto field_builder = [&] (auto &/* bld */, auto /* loc */) {
            fill_decl_members(decl);
        };

        return maybe_declare(decl, [&] {
            return bld.compose< RecordDeclOp >()
                .bind(self.location(decl))
                .bind(self.symbol(decl))
                .bind_always(field_builder)
                .freeze();
        });
    }

} // namespace vast::cg
