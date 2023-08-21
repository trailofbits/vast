// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"
#include "vast/Util/Warnings.hpp"

#include "vast/Dialect/Unsupported/UnsupportedDialect.hpp"
#include "vast/Dialect/Unsupported/UnsupportedOps.hpp"
#include "vast/Dialect/Unsupported/UnsupportedTypes.hpp"

#include "vast/CodeGen/CodeGen.hpp"
#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenContext.hpp"
#include "vast/CodeGen/CodeGenFallBackVisitor.hpp"
#include "vast/CodeGen/CodeGenVisitor.hpp"

namespace vast::cg {

#define MAKE_DECL(type) \
    operation Visit##type(const clang::type *decl) { \
        return make_decl< unsup::UnsupportedDecl >(decl); \
    }

#define MAKE_DECL_WITH_BODY(type, body) \
    operation Visit##type(const clang::type *decl) { \
        return make_decl< unsup::UnsupportedDecl >(decl, decl->get##body()); \
    }

#define MAKE_EXPR(type) \
    operation Visit##type(const clang::type *expr) { \
        return make_expr_default< unsup::UnsupportedExpr >(expr); \
    }

#define MAKE_OPERATION(type) \
    operation Visit##type(const clang::type *expr) { \
        return make_operation< unsup::UnsupportedOp >(expr); \
    }

#define MAKE_TYPE(type) \
    Type Visit##type(const clang::type *ty) { \
        if (ty->isSugared()) \
            return make< unsup::UnsupportedType >(visit(ty->desugar())); \
        return make< unsup::UnsupportedType >(Type()); \
    }

#define MAKE_TYPE_WITH_ELEMENT(type, element) \
    Type Visit##type(const clang::type *ty) { \
        return make< unsup::UnsupportedType >(visit(ty->get##element())); \
    }

    template< typename Derived >
    struct UnsupportedDeclVisitor
        : clang::ConstDeclVisitor< UnsupportedDeclVisitor< Derived >, vast::Operation * >
        , vast::cg::CodeGenVisitorLens< UnsupportedDeclVisitor< Derived >, Derived >
        , vast::cg::CodeGenBuilder< UnsupportedDeclVisitor< Derived >, Derived > {
        using LensType =
            vast::cg::CodeGenVisitorLens< UnsupportedDeclVisitor< Derived >, Derived >;

        using LensType::acontext;
        using LensType::context;
        using LensType::derived;
        using LensType::mcontext;
        using LensType::meta_location;
        using LensType::visit;

        template< typename Op, typename... Args >
        auto make(Args &&...args) {
            return this->template create< Op >(std::forward< Args >(args)...);
        }

        template< typename Decl >
        std::string name(const Decl *decl) {
            std::stringstream ss;
            if (auto decl_ = dyn_cast< clang::Decl >(decl)) {
                ss << decl_->getDeclKindName();
            }
            if (auto named_decl = dyn_cast< clang::NamedDecl >(decl)) {
                std::string decl_name = context().decl_name(named_decl).str();
                ss << "::" << decl_name;
            }
            return ss.str();
        }

        template< typename Op, typename Decl >
        operation make_decl(const Decl *decl) {
            auto loc = meta_location(decl);
            if (auto context = dyn_cast< const clang::DeclContext >(decl)) {
                auto fields = [&](auto &bld, auto loc) {
                    for (auto child : context->decls()) {
                        visit(child);
                    }
                };
                return make< Op >(loc, name(decl), fields);
            }

            auto fields = [&](auto &bld, auto loc) { ; };
            return make< Op >(loc, name(decl), fields);
        }

        template< typename Op, typename Decl, typename DeclOrExpr >
        operation make_decl(const Decl *decl, const DeclOrExpr *child) {
            auto loc    = meta_location(decl);
            auto fields = [&](auto &bld, auto loc) { visit(child); };
            return make< Op >(loc, name(decl), fields);
        }

        MAKE_DECL_WITH_BODY(StaticAssertDecl, AssertExpr)
        MAKE_DECL_WITH_BODY(BlockDecl, Body)
        MAKE_DECL_WITH_BODY(BindingDecl, Binding)
        MAKE_DECL_WITH_BODY(CapturedDecl, Body)
        MAKE_DECL_WITH_BODY(NamespaceAliasDecl, Namespace)
        MAKE_DECL_WITH_BODY(UsingDecl, UnderlyingDecl)
        MAKE_DECL_WITH_BODY(UsingShadowDecl, TargetDecl)

        MAKE_DECL(LinkageSpecDecl)
        MAKE_DECL(NamespaceDecl)
        MAKE_DECL(AccessSpecDecl)
        MAKE_DECL(BuiltinTemplateDecl)
        MAKE_DECL(CXXConstructorDecl)
        MAKE_DECL(CXXConversionDecl)
        MAKE_DECL(CXXDeductionGuideDecl)
        MAKE_DECL(CXXDestructorDecl)
        MAKE_DECL(CXXMethodDecl)
        MAKE_DECL(CXXRecordDecl)
        MAKE_DECL(ClassScopeFunctionSpecializationDecl)
        MAKE_DECL(ClassTemplateDecl)
        MAKE_DECL(ClassTemplatePartialSpecializationDecl)
        MAKE_DECL(ClassTemplateSpecializationDecl)
        MAKE_DECL(ConceptDecl)
        MAKE_DECL(ConstructorUsingShadowDecl)
        MAKE_DECL(DecompositionDecl)
        MAKE_DECL(EmptyDecl)
        MAKE_DECL(EnumDecl)
        MAKE_DECL(EnumConstantDecl)
        MAKE_DECL(ExportDecl)
        MAKE_DECL(ExternCContextDecl)
        MAKE_DECL(FieldDecl)
        MAKE_DECL(FileScopeAsmDecl)
        MAKE_DECL(FriendDecl)
        MAKE_DECL(FriendTemplateDecl)
        MAKE_DECL(FunctionDecl)
        MAKE_DECL(FunctionTemplateDecl)
        MAKE_DECL(HLSLBufferDecl)
        MAKE_DECL(ImplicitConceptSpecializationDecl)
        MAKE_DECL(ImplicitParamDecl)
        MAKE_DECL(ImportDecl)
        MAKE_DECL(IndirectFieldDecl)
        MAKE_DECL(LabelDecl)
        MAKE_DECL(LifetimeExtendedTemporaryDecl)
        MAKE_DECL(MSGuidDecl)
        MAKE_DECL(MSPropertyDecl)
        MAKE_DECL(NonTypeTemplateParmDecl)
        MAKE_DECL(ParmVarDecl)
        MAKE_DECL(PragmaCommentDecl)
        MAKE_DECL(PragmaDetectMismatchDecl)
        MAKE_DECL(RecordDecl)
        MAKE_DECL(RequiresExprBodyDecl)
        MAKE_DECL(TemplateParamObjectDecl)
        MAKE_DECL(TemplateTemplateParmDecl)
        MAKE_DECL(TemplateTypeParmDecl)
        MAKE_DECL(TopLevelStmtDecl)
        MAKE_DECL(CodeGenUnitDecl)
        MAKE_DECL(TypeAliasDecl)
        MAKE_DECL(TypeAliasTemplateDecl)
        MAKE_DECL(TypedefDecl)
        MAKE_DECL(UnnamedGlobalConstantDecl)
        MAKE_DECL(UnresolvedUsingIfExistsDecl)
        MAKE_DECL(UnresolvedUsingTypenameDecl)
        MAKE_DECL(UnresolvedUsingValueDecl)
        MAKE_DECL(UsingDirectiveDecl)
        MAKE_DECL(UsingEnumDecl)
        MAKE_DECL(UsingPackDecl)
        MAKE_DECL(VarDecl)
        MAKE_DECL(VarTemplateDecl)
        MAKE_DECL(VarTemplatePartialSpecializationDecl)
        MAKE_DECL(VarTemplateSpecializationDecl)
    };

    template< typename Derived >
    struct UnsupportedStmtVisitor
        : clang::ConstStmtVisitor< UnsupportedStmtVisitor< Derived >, vast::Operation * >
        , vast::cg::CodeGenVisitorLens< UnsupportedStmtVisitor< Derived >, Derived >
        , vast::cg::CodeGenBuilder< UnsupportedStmtVisitor< Derived >, Derived > {
        using LensType =
            vast::cg::CodeGenVisitorLens< UnsupportedStmtVisitor< Derived >, Derived >;

        using LensType::acontext;
        using LensType::context;
        using LensType::derived;
        using LensType::mcontext;

        using LensType::meta_location;
        using LensType::visit;

        using Builder =
            vast::cg::CodeGenBuilder< UnsupportedStmtVisitor< Derived >, Derived >;
        using RegionAndType = std::pair< std::unique_ptr< Region >, Type >;

        using Builder::builder;
        using Builder::insertion_guard;
        using Builder::set_insertion_point_to_end;
        using Builder::set_insertion_point_to_start;

        template< typename Op, typename... Args >
        auto create(Args &&...args) {
            return builder().template create< Op >(std::forward< Args >(args)...);
        }

        template< typename Op, typename... Args >
        auto make(Args &&...args) {
            return this->template create< Op >(std::forward< Args >(args)...);
        }

        template< typename StmtType >
        std::unique_ptr< Region > make_region_from_child(const StmtType *stmt) {
            auto guard = insertion_guard();

            auto reg = std::make_unique< Region >();
            set_insertion_point_to_start(&reg->emplaceBlock());
            for (auto it = stmt->child_begin(); it != stmt->child_end(); ++it) {
                visit(*it);
            }
            return reg;
        }

        template< typename StmtType >
        RegionAndType yield_region(const StmtType *stmt) {
            auto guard = insertion_guard();
            auto reg   = make_region_from_child(stmt);

            auto &block = reg->back();
            auto type   = Type();
            set_insertion_point_to_end(&block);
            if (block.back().getNumResults() > 0) {
                type = block.back().getResult(0).getType();
                create< hl::ValueYieldOp >(meta_location(stmt), block.back().getResult(0));
            }
            return { std::move(reg), type };
        }

        template< typename Op, typename StmtType >
        operation make_expr_default(const StmtType *stmt) {
            auto loc            = meta_location(stmt);
            auto [region, type] = yield_region(stmt);
            auto node           = stmt->getStmtClassName();
            return make< Op >(loc, node, type, std::move(region));
        }

        template< typename Op, typename Expr >
        operation make_operation(const Expr *expr) {
            auto loc   = meta_location(expr);
            auto rtype = visit(expr->getType());
            auto node  = expr->getStmtClassName();

            llvm::SmallVector< vast::Value > elements;
            for (auto it = expr->child_begin(); it != expr->child_end(); ++it) {
                elements.push_back(visit(*it)->getResult(0));
            }
            return make< Op >(loc, rtype, node, elements);
        }

        template< typename Op, typename Expr, typename Region >
        operation make_operation(const Expr *expr, const Region &region) {
            auto loc   = meta_location(expr);
            auto rtype = visit(expr->getType());
            auto node  = expr->getStmtClassName();
            return make< Op >(loc, rtype, node, region);
        }

        operation VisitGCCAsmStmt(const clang::GCCAsmStmt *expr) {
            auto loc   = meta_location(expr);
            auto rtype = Type();

            llvm::SmallVector< vast::Value > elements;
            for (auto i = 0u; i < expr->getNumInputs(); i++) {
                auto in = expr->getInputExpr(i);
                elements.push_back(visit(in)->getResult(0));
            }

            for (auto i = 0u; i < expr->getNumOutputs(); i++) {
                auto out = expr->getOutputExpr(i);
                elements.push_back(visit(out)->getResult(0));
            }

            return make< unsup::UnsupportedOp >(loc, rtype, expr->getStmtClassName(), elements);
        }

        operation VisitOpaqueValueExpr(const clang::OpaqueValueExpr *expr) {
            llvm::SmallVector< vast::Value > elements;
            elements.push_back(visit(expr->getSourceExpr())->getResult(0));
            return make_operation< unsup::UnsupportedOp >(expr, elements);
        }

        MAKE_OPERATION(AtomicExpr)
        MAKE_OPERATION(BinaryOperator)
        MAKE_OPERATION(ConditionalOperator)
        MAKE_OPERATION(CXXConstructExpr)
        MAKE_OPERATION(CXXDefaultInitExpr)
        MAKE_OPERATION(GenericSelectionExpr)
        MAKE_OPERATION(ImplicitValueInitExpr)
        MAKE_OPERATION(OffsetOfExpr)
        MAKE_OPERATION(PredefinedExpr)
        MAKE_OPERATION(UnaryOperator)

        MAKE_EXPR(AddrLabelExpr)
        MAKE_EXPR(ArrayInitIndexExpr)
        MAKE_EXPR(ArrayInitLoopExpr)
        MAKE_EXPR(ArraySubscriptExpr)
        MAKE_EXPR(ArrayTypeTraitExpr)
        MAKE_EXPR(AsTypeExpr)
        MAKE_EXPR(AttributedStmt)
        MAKE_EXPR(BlockExpr)
        MAKE_EXPR(BreakStmt)
        MAKE_EXPR(BuiltinBitCastExpr)
        MAKE_EXPR(CStyleCastExpr)
        MAKE_EXPR(CUDAKernelCallExpr)
        MAKE_EXPR(CXXAddrspaceCastExpr)
        MAKE_EXPR(CXXBindTemporaryExpr)
        MAKE_EXPR(CXXBoolLiteralExpr)
        MAKE_EXPR(CXXCatchStmt)
        MAKE_EXPR(CXXConstCastExpr)

        MAKE_EXPR(CXXDefaultArgExpr)
        MAKE_EXPR(CXXDeleteExpr)
        MAKE_EXPR(CXXDependentScopeMemberExpr)
        MAKE_EXPR(CXXDynamicCastExpr)
        MAKE_EXPR(CXXFoldExpr)
        MAKE_EXPR(CXXForRangeStmt)
        MAKE_EXPR(CXXFunctionalCastExpr)
        MAKE_EXPR(CXXInheritedCtorInitExpr)
        MAKE_EXPR(CXXMemberCallExpr)
        MAKE_EXPR(CXXNewExpr)
        MAKE_EXPR(CXXNoexceptExpr)
        MAKE_EXPR(CXXNullPtrLiteralExpr)
        MAKE_EXPR(CXXOperatorCallExpr)
        MAKE_EXPR(CXXParenListInitExpr)
        MAKE_EXPR(CXXPseudoDestructorExpr)
        MAKE_EXPR(CXXReinterpretCastExpr)
        MAKE_EXPR(CXXRewrittenBinaryOperator)
        MAKE_EXPR(CXXScalarValueInitExpr)
        MAKE_EXPR(CXXStaticCastExpr)
        MAKE_EXPR(CXXStdInitializerListExpr)
        MAKE_EXPR(CXXTemporaryObjectExpr)
        MAKE_EXPR(CXXThisExpr)
        MAKE_EXPR(CXXThrowExpr)
        MAKE_EXPR(CXXTryStmt)
        MAKE_EXPR(CXXTypeidExpr)
        MAKE_EXPR(CXXUnresolvedConstructExpr)
        MAKE_EXPR(CXXUuidofExpr)
        MAKE_EXPR(CallExpr)
        MAKE_EXPR(CapturedStmt)
        MAKE_EXPR(CaseStmt)
        MAKE_EXPR(CharacterLiteral)
        MAKE_EXPR(ChooseExpr)
        MAKE_EXPR(CoawaitExpr)
        MAKE_EXPR(CompoundAssignOperator)
        MAKE_EXPR(CompoundLiteralExpr)
        MAKE_EXPR(CompoundStmt)
        MAKE_EXPR(ConceptSpecializationExpr)

        MAKE_EXPR(ConstantExpr)
        MAKE_EXPR(ContinueStmt)
        MAKE_EXPR(ConvertVectorExpr)
        MAKE_EXPR(CoreturnStmt)
        MAKE_EXPR(CoroutineBodyStmt)
        MAKE_EXPR(CoyieldExpr)
        MAKE_EXPR(DeclRefExpr)
        MAKE_EXPR(DeclStmt)
        MAKE_EXPR(DefaultStmt)
        MAKE_EXPR(DependentCoawaitExpr)
        MAKE_EXPR(DependentScopeDeclRefExpr)
        MAKE_EXPR(DesignatedInitExpr)
        MAKE_EXPR(DesignatedInitUpdateExpr)
        MAKE_EXPR(DoStmt)
        MAKE_EXPR(ExprWithCleanups)
        MAKE_EXPR(ExpressionTraitExpr)
        MAKE_EXPR(ExtVectorElementExpr)
        MAKE_EXPR(FixedPointLiteral)
        MAKE_EXPR(FloatingLiteral)
        MAKE_EXPR(ForStmt)
        MAKE_EXPR(FunctionParmPackExpr)
        MAKE_EXPR(GNUNullExpr)

        MAKE_EXPR(GotoStmt)
        MAKE_EXPR(IfStmt)
        MAKE_EXPR(ImaginaryLiteral)
        MAKE_EXPR(ImplicitCastExpr)
        MAKE_EXPR(IndirectGotoStmt)
        MAKE_EXPR(InitListExpr)
        MAKE_EXPR(IntegerLiteral)
        MAKE_EXPR(LabelStmt)
        MAKE_EXPR(LambdaExpr)
        MAKE_EXPR(MSAsmStmt)
        MAKE_EXPR(MSDependentExistsStmt)
        MAKE_EXPR(MSPropertyRefExpr)
        MAKE_EXPR(MSPropertySubscriptExpr)
        MAKE_EXPR(MaterializeTemporaryExpr)
        MAKE_EXPR(MatrixSubscriptExpr)
        MAKE_EXPR(MemberExpr)
        MAKE_EXPR(NoInitExpr)
        MAKE_EXPR(NullStmt)
        MAKE_EXPR(PackExpansionExpr)
        MAKE_EXPR(ParenExpr)
        MAKE_EXPR(ParenListExpr)

        MAKE_EXPR(PseudoObjectExpr)
        MAKE_EXPR(RecoveryExpr)
        MAKE_EXPR(RequiresExpr)
        MAKE_EXPR(ReturnStmt)
        MAKE_EXPR(SEHExceptStmt)
        MAKE_EXPR(SEHFinallyStmt)
        MAKE_EXPR(SEHLeaveStmt)
        MAKE_EXPR(SEHTryStmt)
        MAKE_EXPR(SYCLUniqueStableNameExpr)
        MAKE_EXPR(ShuffleVectorExpr)
        MAKE_EXPR(SizeOfPackExpr)
        MAKE_EXPR(SourceLocExpr)
        MAKE_EXPR(StmtExpr)
        MAKE_EXPR(StringLiteral)
        MAKE_EXPR(SubstNonTypeTemplateParmExpr)
        MAKE_EXPR(SubstNonTypeTemplateParmPackExpr)
        MAKE_EXPR(SwitchStmt)
        MAKE_EXPR(TypeTraitExpr)
        MAKE_EXPR(TypoExpr)
        MAKE_EXPR(UnaryExprOrTypeTraitExpr)

        MAKE_EXPR(UnresolvedLookupExpr)
        MAKE_EXPR(UnresolvedMemberExpr)
        MAKE_EXPR(UserDefinedLiteral)
        MAKE_EXPR(VAArgExpr)
        MAKE_EXPR(WhileStmt)
    };

    template< typename Derived >
    struct UnsupportedTypeVisitor
        : clang::TypeVisitor< UnsupportedTypeVisitor< Derived >, vast::hl::Type >
        , vast::cg::CodeGenVisitorLens< UnsupportedTypeVisitor< Derived >, Derived >
        , vast::cg::CodeGenBuilder< UnsupportedTypeVisitor< Derived >, Derived > {
        using LensType =
            vast::cg::CodeGenVisitorLens< UnsupportedTypeVisitor< Derived >, Derived >;

        using LensType::acontext;
        using LensType::context;
        using LensType::derived;
        using LensType::mcontext;
        using LensType::visit;

        template< typename high_level_type >
        auto type_builder() {
            return this->template make_type< high_level_type >().bind(&mcontext());
        }

        template< typename Op, typename Type >
        auto make(Type desugar) {
            return type_builder< Op >().bind(desugar).freeze();
        }

        MAKE_TYPE_WITH_ELEMENT(ArrayType, ElementType)
        MAKE_TYPE_WITH_ELEMENT(PackExpansionType, Pattern)
        MAKE_TYPE_WITH_ELEMENT(PipeType, ElementType)
        MAKE_TYPE_WITH_ELEMENT(PointerType, PointeeType)
        MAKE_TYPE_WITH_ELEMENT(ReferenceType, PointeeTypeAsWritten)
        MAKE_TYPE_WITH_ELEMENT(VectorType, ElementType)

        MAKE_TYPE(AdjustedType)
        MAKE_TYPE(AtomicType)
        MAKE_TYPE(AttributedType)
        MAKE_TYPE(BTFTagAttributedType)
        MAKE_TYPE(BitIntType)
        MAKE_TYPE(BlockPointerType)
        MAKE_TYPE(ComplexType)
        MAKE_TYPE(DecltypeType)
        MAKE_TYPE(DeducedType)
        MAKE_TYPE(DependentAddressSpaceType)
        MAKE_TYPE(DependentBitIntType)
        MAKE_TYPE(DependentSizedExtVectorType)
        MAKE_TYPE(DependentVectorType)
        MAKE_TYPE(InjectedClassNameType)
        MAKE_TYPE(MacroQualifiedType)
        MAKE_TYPE(MatrixType)
        MAKE_TYPE(MemberPointerType)
        MAKE_TYPE(ObjCObjectPointerType)
        MAKE_TYPE(ObjCObjectType)
        MAKE_TYPE(ObjCTypeParamType)
        MAKE_TYPE(ParenType)
        MAKE_TYPE(SubstTemplateTypeParmPackType)
        MAKE_TYPE(SubstTemplateTypeParmType)
        // TagType
        MAKE_TYPE(TemplateSpecializationType)
        MAKE_TYPE(TemplateTypeParmType)
        MAKE_TYPE(TypeOfExprType)
        MAKE_TYPE(TypeOfType)
        // TypeWithKeyword
        MAKE_TYPE(TypedefType)
        MAKE_TYPE(UnaryTransformType)
        MAKE_TYPE(UnresolvedUsingType)
        MAKE_TYPE(UsingType)
    };

#undef MAKE_DECL
#undef MAKE_EXPR
#undef MAKE_OPERATION
#undef MAKE_TYPE
#undef MAKE_TYPE_WITH_ELEMENT

    template< typename Derived, template< typename > typename FallBack >
    struct UnsupportedFallBackVisitor
        : UnsupportedDeclVisitor< Derived >
        , UnsupportedStmtVisitor< Derived >
        , UnsupportedTypeVisitor< Derived >
        , FallBack< Derived >
    {
        using DeclVisitor = UnsupportedDeclVisitor< Derived >;
        using StmtVisitor = UnsupportedStmtVisitor< Derived >;
        using TypeVisitor = UnsupportedTypeVisitor< Derived >;

        using FallBackVisitor = FallBack< Derived >;

        operation Visit(const clang::Stmt *stmt) {
            if (auto result = StmtVisitor::Visit(stmt)) {
                return result;
            }

            return FallBackVisitor::Visit(stmt);
        }

        operation Visit(const clang::Decl *decl) {
            if (auto result = DeclVisitor::Visit(decl)) {
                return result;
            }

            return FallBackVisitor::Visit(decl);
        }

        Type Visit(const clang::Type *type) {
            if (auto result = TypeVisitor::Visit(type)) {
                return result;
            }

            return FallBackVisitor::Visit(type);
        }

        Type Visit(clang::QualType type) {
            if (!type.isNull()) {
                if (auto result = TypeVisitor::Visit(type.getTypePtr())) {
                    return result;
                }
            }

            return FallBackVisitor::Visit(type);
        }
    };

} // namespace vast::cg
