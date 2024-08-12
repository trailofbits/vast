// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/Analysis/Iterators.hpp"
#include "vast/Analysis/Clang/CFG.hpp"
#include "vast/Analysis/Clang/FlowSensitive/DataflowWorklist.hpp"
#include "vast/Interfaces/CFG/CFGInterface.hpp"
#include "vast/Interfaces/Analysis/AnalysisDeclContextInterface.hpp"
#include "vast/Interfaces/AST/DeclInterface.hpp"
#include "vast/Interfaces/AST/TypeInterface.hpp"
#include "vast/Interfaces/AST/StmtInterface.hpp"
#include "vast/Interfaces/AST/ExprInterface.hpp"
#include "vast/Interfaces/AST/StmtVisitor.h"

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/PackedVector.h>
#include <llvm/ADT/PointerIntPair.h>
#include <llvm/ADT/SmallVector.h>
#include <optional>

namespace vast::analysis {

    /// A use of a variable, which might be uninitialized.
    class UninitUse {
    public:
        struct Branch {
            ast::StmtInterface Terminator;
            unsigned Output;
        };

    private:
        /// The expression which uses this variable.
        ast::ExprInterface User;

        /// Is this use uninitialized whenever the function is called?
        bool UninitAfterCall = false;

        /// Is this use uninitialized whenever the variable declaration is reached?
        bool UninitAfterDecl = false;

        /// Does this use always see an uninitialized value?
        bool AlwaysUninit;

        /// This use is always uninitialized if it occurs after any of these branches
        /// is taken.
        llvm::SmallVector< Branch, 2 > UninitBranches;

    public:
        UninitUse(ast::ExprInterface User, bool AlwaysUninit)
            : User(User), AlwaysUninit(AlwaysUninit) {}

        void addUninitBranch(Branch B) {
            UninitBranches.push_back(B);
        }

        void setUninitAfterCall() { UninitAfterCall = true; }
        void setUninitAfterDecl() { UninitAfterDecl = true; }

        /// Get the expression containing the uninitialized use.
        ast::ExprInterface getUser() const { return User; }

        /// The kind of uninitialized use.
        enum class Kind {
            /// The use might be uninitialized.
            Maybe,

            /// The use is uninitialized whenever a certain branch is taken.
            Sometimes,

            /// The use is uninitialized the first time it is reached after we reach
            /// the variable's declaration.
            AfterDecl,

            /// The use is uninitialized the first time it is reached after the function
            /// is called.
            AfterCall,

            /// The use is always uninitialized.
            Always
        };

        using branch_iterator = llvm::SmallVectorImpl< Branch >::const_iterator;

        /// Branches which inevitably result in the variable being used uninitialized.
        branch_iterator branch_begin() const { return UninitBranches.begin(); }
        branch_iterator branch_end() const { return UninitBranches.end(); }
        bool branch_empty() const { return UninitBranches.empty(); }

        /// Get the kind of uninitialized use.
        Kind getKind() const {
            return AlwaysUninit ? Kind::Always :
                UninitAfterCall ? Kind::AfterCall :
                UninitAfterDecl ? Kind::AfterDecl :
                !branch_empty() ? Kind::Sometimes : Kind::Maybe;
        }
    };

    // osablonovat triedu
    class UninitVariablesHandler {
    public:
        UninitVariablesHandler() = default;
        virtual ~UninitVariablesHandler();

        /// Called when the uninitialized variable is used at the given expression.
        virtual void handleUseOfUninitVariable(ast::VarDeclInterface vd,
                                               const UninitUse &use) {}

        /// Called when the uninitialized variable is used as const refernce argument.
        virtual void handleConstRefUseOfUninitVariable(ast::VarDeclInterface vd,
                                                       const UninitUse &use) {}

        /// Called when the uninitialized variable analysis detects the
        /// idiom 'int x = x'.  All other uses of 'x' within the initializer
        /// are handled by handleUseOfUninitVariable.
        virtual void handleSelfInit(ast::VarDeclInterface vd) {}
    };

    struct UninitVariablesAnalysisStats {
        unsigned NumVariablesAnalyzed;
        unsigned NumBlockVisits;
    };

    class Capture {
        enum {
            flag_isByRef = 0x1,
            flag_isNested = 0x2
        };

        /// The variable being captured.
        llvm::PointerIntPair< ast::VarDeclInterface *, 2 > VariableAndFlags;

        /// The copy expression, expressed in terms of a DeclRef (or
        /// BlockDeclRef) to the captured variable.  Only required if the
        /// variable has a C++ class type.
        ast::ExprInterface CopyExpr;

    public:
        Capture(ast::VarDeclInterface *variable, bool byRef, bool nested, ast::ExprInterface copy)
            : VariableAndFlags(variable,
                        (byRef ? flag_isByRef : 0) | (nested ? flag_isNested : 0)),
              CopyExpr(copy) {}

        /// The variable being captured.
        ast::VarDeclInterface *getVariable() const { return VariableAndFlags.getPointer(); }

        /// Whether this is a "by ref" capture, i.e. a capture of a __block
        /// variable.
        bool isByRef() const { return VariableAndFlags.getInt() & flag_isByRef; }

        bool isEscapingByref() const {
          return getVariable()->isEscapingByref();
        }

        bool isNonEscapingByref() const {
          return getVariable()->isNonEscapingByref();
        }

        /// Whether this is a nested capture, i.e. the variable captured
        /// is not from outside the immediately enclosing function/block.
        bool isNested() const { return VariableAndFlags.getInt() & flag_isNested; }

        bool hasCopyExpr() const { return CopyExpr != nullptr; }
        ast::ExprInterface getCopyExpr() const { return CopyExpr; }
        void setCopyExpr(ast::ExprInterface e) { CopyExpr = e; }
    };

    static mlir::Operation *getDecl(ast::DeclRefExprInterface) {
        VAST_UNIMPLEMENTED;
    }

    static ast::DeclInterface getSingleDecl(ast::DeclStmtInterface) {
        VAST_UNIMPLEMENTED;
    }

    static ast::BlockDeclInterface getBlockDecl(ast::BlockExprInterface) {
        VAST_UNIMPLEMENTED;
    }

    static ast::RecordDeclInterface getAsRecordDecl(ast::QualTypeInterface) {
        VAST_UNIMPLEMENTED;
    }

    static bool isZeroSize(ast::FieldDeclInterface) {
        VAST_UNIMPLEMENTED;
    }

    static ast::DeclInterface getCalleeDecl(ast::CallExprInterface) {
        VAST_UNIMPLEMENTED;
    }

    static llvm::ArrayRef< Capture > captures(ast::BlockDeclInterface) {
        VAST_UNIMPLEMENTED;
    }

    static std::vector< ast::DeclInterface > decls(ast::DeclStmtInterface) {
        VAST_UNIMPLEMENTED;
    }

    static ast::FunctionDeclInterface getDirectCallee(ast::CallExprInterface) {
        VAST_UNIMPLEMENTED;
    }

    static bool recordIsNotEmpty(ast::RecordDeclInterface RD) {
        // We consider a record decl to be empty if it contains only unnamed bit-
        // fields, zero-width fields, and fields of empty record type.
        for (auto FD : RD.fields()) {
            if (FD.isUnnamedBitField())
                continue;

            if (isZeroSize(FD))
               continue;
            
            // The only case remaining to check is for a field declaration of record
            // type and whether that record itself is empty.
            if (auto FieldRD = getAsRecordDecl(FD.getType());
                !FieldRD || recordIsNotEmpty(FieldRD))
                return true;
        }
        return false;
    }

    static bool isTrackedVar(ast::VarDeclInterface vd, ast::DeclContextInterface dc) {
        if (vd.isLocalVarDecl() && !vd.hasGlobalStorage() &&
            !vd.isExceptionVariable() && !vd.isInitCapture() && !vd.isImplicit() &&
            vd.getDeclContext() == dc) {
            ast::QualTypeInterface ty = vd.getType();
            if (auto RD = getAsRecordDecl(ty)) {
                return recordIsNotEmpty(RD);
            }

            ast::TypeInterface T = ty.getTypePtr();
            return T.isScalarType() || T.isVectorType() || T.isRVVSizelessBuiltinType();
        }
        return false;
    }

//------------------------------------------------------------------------====//
// DeclToIndex: a mapping from Decls we track to value indices.
//====------------------------------------------------------------------------//

    namespace {
        
        class DeclToIndex {
            llvm::DenseMap< ast::VarDeclInterface, unsigned > map;

        public:
            DeclToIndex() = default;

            /// Compute the actual mapping from declarations to bits.
            void computeMap(ast::DeclContextInterface dc) {
                unsigned count = 0;
                specific_decl_interface_iterator< ast::VarDeclInterface > I(dc.decls_begin()),
                                                                          E(dc.decls_end());
                for ( ; I != E; ++I) {
                    ast::VarDeclInterface vd = *I;
                    if (isTrackedVar(vd, dc)) {
                        map[vd] = count++;
                    }
                }
            }
            
            /// Return the number of declarations in the map.
            unsigned size() const {
                return map.size();
            }

            /// Returns the bit vector index for a given declaration.
            std::optional< unsigned > getValueIndex(ast::VarDeclInterface d) const {
                llvm::DenseMap< ast::VarDeclInterface, unsigned >::const_iterator I = map.find(d);

                if (I == map.end()) {
                    return std::nullopt;
                }
                return I->second;
            }
        };
    
    } // namespace
      
//------------------------------------------------------------------------====//
// CFGBlockValues: dataflow values for CFG blocks.
//====------------------------------------------------------------------------//

    enum Value {
        Unknown =          0x0, /* 00 */
        Initialized =      0x1, /* 01 */
        Uninitialized =    0x2, /* 10 */
        MayUninitialized = 0x3  /* 11 */
    };

    static bool isUninitialized(Value v) {
        return v >= Value::Uninitialized;
    }

    static bool isAlwaysUninit(Value v) {
        return v == Value::Uninitialized;
    }

    namespace {
        
        using ValueVector = llvm::PackedVector<Value, 2, llvm::SmallBitVector>;

        class CFGBlockValues {
            cfg::CFGInterface &cfg;
            clang::SmallVector< ValueVector, 8 > vals;
            ValueVector scratch;
            DeclToIndex declToIndex;

        public:
            CFGBlockValues(cfg::CFGInterface &c) : cfg(c), vals(0) {}

            void computeSetOfDeclarations(ast::DeclContextInterface dc) {
                declToIndex.computeMap(dc);
                unsigned decls = declToIndex.size();
                scratch.resize(decls);
                unsigned n = cfg.getNumBlockIDs();
                if (!n) {
                    return;
                }
                vals.resize(n);
                for (auto &val : vals) {
                    val.resize(decls);
                }
            }

            bool hasNoDeclarations() {
                return declToIndex.size() == 0;
            }

            unsigned getNumEntries() {
                return declToIndex.size();
            }

            ValueVector &getValueVector(cfg::CFGBlockInterface block) {
                return vals[block.getBlockID()];
            }

            void resetScratch() {
                scratch.reset();
            }

            void setAllScratchValues(Value v) {
                for (unsigned I = 0, E = scratch.size(); I != E; ++I ) {
                    scratch[I] = v;
                }
            }

            void mergeIntoScratch(const ValueVector &source, bool isFirst) {
                if (isFirst) {
                    scratch = source;
                } else {
                    scratch |= source;
                }
            }

            bool updateValueVectorWithScratch(cfg::CFGBlockInterface block) {
                ValueVector &dst = getValueVector(block);
                bool changed = (dst != scratch);
                if (changed)
                    dst = scratch;
                return changed;
            }

            ValueVector::reference operator[](ast::VarDeclInterface vd) {
                return scratch[*declToIndex.getValueIndex(vd)];
            }

            Value getValue(cfg::CFGBlockInterface block, cfg::CFGBlockInterface dstBlock,
                           ast::VarDeclInterface vd) {
                std::optional< unsigned > idx = declToIndex.getValueIndex(vd);
                return getValueVector(block)[*idx];
            }
        };
    
    } // namespace
    
    //------------------------------------------------------------------------====//
    // Classification of DeclRefExprs as use or initialization.
    //====------------------------------------------------------------------------//

    namespace {
    
        class FindVarResult {
            ast::VarDeclInterface vd;
            ast::DeclRefExprInterface dr;

        public:
            FindVarResult(ast::VarDeclInterface vd, ast::DeclRefExprInterface dr) : vd(vd), dr(dr) {}

            ast::DeclRefExprInterface getDeclRefExpr() {
                return dr;
            }

            ast::VarDeclInterface getDecl() {
                return vd;
            }
        };

    } // namespace
    
    static mlir::Operation *stripCasts(ast::ASTContextInterface C, ast::ExprInterface Ex) {
        while (Ex) {
            Ex = Ex.IgnoreParenNoopCasts(C);
            if (auto CE = dyn_cast< ast::CastExprInterface >(Ex.getOperation())) {
                if (CE.getCastKind() == clang::CK_LValueBitCast) {
                    Ex = CE.getSubExpr();
                    continue;
                }
            }
            break;
        }
        return Ex.getOperation();
    }

    static ast::DeclRefExprInterface getSelfInitExpr(ast::VarDeclInterface VD) {
        if (VD.getType()->isRecordType()) {
            return {};
        }
        if (ast::ExprInterface Init = VD.getInit()) {
            auto DRE = dyn_cast< ast::DeclRefExprInterface >(stripCasts(VD.getASTContext(), Init));
            if (DRE && dyn_cast< ast::ValueDeclInterface >(getDecl(DRE)) == VD) {
                return DRE;
            }
        }
        return {};
    }

    /// If E is an expression comprising a reference to a single variable, find that
    /// variable.
    static FindVarResult findVar(ast::ExprInterface E, ast::DeclContextInterface DC) {
        auto DRE = dyn_cast< ast::DeclRefExprInterface >(stripCasts(DC.getParentASTContext(), E));
        if (DRE) {
            auto VD = dyn_cast< ast::VarDeclInterface >(getDecl(DRE));
            if (VD) {
                if (isTrackedVar(VD, DC)) {
                    return FindVarResult(VD, DRE);
                }
            }
        }
        return FindVarResult({}, {});
    }

    static bool isPointerToConst(ast::QualTypeInterface QT) {
        ast::TypeInterface T = QT.getTypePtr();
        return T.isAnyPointerType() && T.getPointeeType().isConstQualified();
    }

    static bool hasTrivialBody(ast::CallExprInterface CE) {
        if (ast::FunctionDeclInterface FD = getDirectCallee(CE)) {
            if (ast::FunctionTemplateDeclInterface FTD = FD.getPrimaryTemplate()) {
                return FTD.getTemplatedDecl().hasTrivialBody();
            }
            return FD.hasTrivialBody();
        }
        return false;
    }
    
    namespace {

        static mlir::Operation *getMemberDecl(ast::MemberExprInterface) {
            VAST_UNIMPLEMENTED;
        }
        
        /// Classify each DeclRefExpr as an initialization or a use. Any
        /// DeclRefExpr which isn't explicitly classified will be assumed to have
        /// escaped the analysis and will be treated as an initialization.
        class ClassifyRefs : public ast::StmtVisitor< ClassifyRefs > {
        using base = ast::StmtVisitor< ClassifyRefs >;
        public:
            enum class Class {
                Init,
                Use,
                SelfInit,
                ConstRefUse,
                Ignore
            };

        private:
            ast::DeclContextInterface DC;
            llvm::DenseMap< ast::DeclRefExprInterface, Class > Classification;

            bool isTrackedVar(ast::VarDeclInterface VD) const {
                return vast::analysis::isTrackedVar(VD, DC);
            }

            void classify(ast::ExprInterface E, Class C) {
                E = E.IgnoreParens();
                if (auto CO = dyn_cast< ast::ConditionalOperatorInterface >(E.getOperation())) {
                    classify(CO.getTrueExpr(), C);
                    classify(CO.getFalseExpr(), C);
                    return;
                }

                if (auto BCO = dyn_cast< ast::BinaryConditionalOperatorInterface >(E.getOperation())) {
                    classify(BCO.getFalseExpr(), C);
                    return;
                }

                if (auto OVE = dyn_cast< ast::OpaqueValueExprInterface >(E.getOperation())) {
                    classify(OVE.getSourceExpr(), C);
                    return;
                }

                if (auto ME = dyn_cast< ast::MemberExprInterface >(E.getOperation())) {
                    if (auto VD = dyn_cast< ast::VarDeclInterface >(getMemberDecl(ME))) {
                        if (!VD.isStaticDataMember()) {
                            classify(ME.getBase(), C);
                        }
                    }
                    return;
                }

                if (auto BO = dyn_cast< ast::BinaryOperatorInterface >(E.getOperation())) {
                    switch (BO.getOpcode()) {
                        case clang::BO_PtrMemD:
                        case clang::BO_PtrMemI:
                            classify(BO.getLHS(), C);
                            return;

                        case clang::BO_Comma:
                            classify(BO.getRHS(), C);
                            return;

                        default:
                            return;
                    }
                }

                FindVarResult Var = ::vast::analysis::findVar(E, DC);
                if (ast::DeclRefExprInterface DRE = Var.getDeclRefExpr()) {
                    Classification[DRE] = std::max(Classification[DRE], C);
                }
            }


        public:
            ClassifyRefs(AnalysisDeclContextInterface AC)
                : DC(cast< ast::DeclContextInterface >(AC.getDecl().getOperation())) {}

            void operator()(ast::StmtInterface S) {
                base::base::Visit(S);
            }

            Class get(ast::DeclRefExprInterface DRE) const {
                llvm::DenseMap< ast::DeclRefExprInterface, Class >::const_iterator I = Classification.find(DRE);
                if (I != Classification.end()) {
                    return I->second;
                }

                auto VD = dyn_cast< ast::VarDeclInterface >(getDecl(DRE));
                if (!VD || !isTrackedVar(VD)) {
                    return Class::Ignore;
                }

                return Class::Init;
            }

            void VisitDeclStmt(ast::DeclStmtInterface DS) {
                for (auto DI : decls(DS)) {
                    auto VD = dyn_cast< ast::VarDeclInterface >(DI.getOperation());
                    if (VD && isTrackedVar(VD)) {
                        if (ast::DeclRefExprInterface DRE = getSelfInitExpr(VD)) {
                            Classification[DRE] = Class::SelfInit;
                        }
                    }
                }
            }

            void VisitUnaryOperator(ast::UnaryOperatorInterface UO) {
                // Increment and decrement are uses despite there being no lvalue-to-rvalue
                // conversion.
                if (UO.isIncrementDecrementOp()) {
                    classify(UO.getSubExpr(), Class::Use);
                }
            }

            void VisitBinaryOperator(ast::BinaryOperatorInterface BO) {
                // Ignore the evaluation of a DeclRefExpr on the LHS of an assignment. If this
                // is not a compound-assignment, we will treat it as initializing the variable
                // when TransferFunctions visits it. A compound-assignment does not affect
                // whether a variable is uninitialized, and there's no point counting it as a
                // use.
                if (BO.isCompoundAssignmentOp()) {
                    classify(BO.getLHS(), Class::Use);
                }
                else if (BO.getOpcode() == clang::BO_Assign || BO.getOpcode() == clang::BO_Comma) {
                    classify(BO.getLHS(), Class::Ignore);
                }
            }

            void VisitCallExpr(ast::CallExprInterface CE) {
                // Classify arguments to std::move as used.
                if (CE.isCallToStdMove()) {
                    // RecordTypes are handled in SemaDeclCXX.cpp.
                    if (!CE.getArg(0).getType().getTypePtr().isRecordType()) {
                        classify(CE.getArg(0), Class::Use);
                    }
                    return;
                }
                bool isTrivialBody = hasTrivialBody(CE);
                // If a value is passed by const pointer to a function,
                // we should not assume that it is initialized by the call, and we
                // conservatively do not assume that it is used.
                // If a value is passed by const reference to a function,
                // it should already be initialized.
                for (call_expr_arg_iterator I = CE.arg_begin(), E = CE.arg_end();
                     I != E; ++I) {
                    if ((*I).isGLValue()) {
                        if ((*I).getType().isConstQualified()) {
                            classify(*I, isTrivialBody ? Class::Ignore : Class::ConstRefUse);  
                        }
                    }
                    else if (isPointerToConst((*I).getType())) {
                        auto Ex = dyn_cast< ast::ExprInterface >(stripCasts(DC.getParentASTContext(), *I));
                        auto UO = dyn_cast< ast::UnaryOperatorInterface >(Ex.getOperation());
                        if (UO && UO.getOpcode() == clang::UO_AddrOf) {
                            Ex = UO.getSubExpr();
                        }
                        classify(Ex, Class::Ignore);
                    }
                }
            }

            void VisitCastExpr(ast::CastExprInterface CE) {
                if (CE.getCastKind() == clang::CastKind::CK_LValueToRValue) {
                    classify(CE.getSubExpr(), Class::Use);
                } else if (auto CSE = dyn_cast< ast::CStyleCastExprInterface >(CE.getOperation())) {
                    if (CSE.getType()->isVoidType()) {
                        // Squelch any detected load of an uninitialized value if
                        // we cast it to void.
                        // e.g. (void) x;
                        classify(CSE.getSubExpr(), Class::Ignore);
                    }
                }
            }

            void VisitOMPExecutableDirective(ast::OMPExecutableDirectiveInterface ED) {
                VAST_UNIMPLEMENTED;
            }
        };
    } // namespace

    //------------------------------------------------------------------------====//
    // Transfer function for uninitialized values analysis.
    //====------------------------------------------------------------------------//
    
    namespace {
    
        class TransferFunctions : public ast::StmtVisitor< TransferFunctions > {
            CFGBlockValues &vals;
            cfg::CFGInterface cfg;
            cfg::CFGBlockInterface block;
            AnalysisDeclContextInterface ac;
            const ClassifyRefs &classification;
            UninitVariablesHandler &handler;

        public:
            TransferFunctions(auto &vals, auto cfg, auto block, auto ac, const auto &classification, auto &handler)
              : vals(vals), cfg(cfg), block(block), ac(ac),
                classification(classification), handler(handler) {}

            bool isTrackedVar(ast::VarDeclInterface vd) {
                return ::vast::analysis::isTrackedVar(vd, cast< ast::DeclContextInterface >(ac.getDecl().getOperation()));
            }

            FindVarResult findVar(ast::ExprInterface ex) {
                return ::vast::analysis::findVar(ex, cast< ast::DeclContextInterface >(ac.getDecl().getOperation()));
            }

            void VisitDeclRefExpr(ast::DeclRefExprInterface dr) {
                switch (classification.get(dr)) {
                    case ClassifyRefs::Class::Ignore:
                        return;
                    
                    case ClassifyRefs::Class::Use:
                        reportUse(dr, cast< ast::VarDeclInterface >(getDecl(dr)));
                        break;

                    case ClassifyRefs::Class::Init:
                        vals[cast< ast::VarDeclInterface >(getDecl(dr))] = Value::Initialized;
                        break;

                    case ClassifyRefs::Class::SelfInit:
                        handler.handleSelfInit(cast< ast::VarDeclInterface >(getDecl(dr)));
                        break;

                    case ClassifyRefs::Class::ConstRefUse:
                        reportConstRefUse(dr, cast< ast::VarDeclInterface >(getDecl(dr)));
                        break;
                }
            }

            void reportUse(ast::ExprInterface ex, ast::VarDeclInterface vd) {
                Value v = vals[vd];
                if (isUninitialized(v)) {
                    handler.handleUseOfUninitVariable(vd, getUninitUse(ex, vd, v));
                }
            }

            void reportConstRefUse(ast::ExprInterface ex, ast::VarDeclInterface vd) {
                Value v = vals[vd];
                if (isAlwaysUninit(v)) {
                    handler.handleConstRefUseOfUninitVariable(vd, getUninitUse(ex, vd, v));
                }
            }

            UninitUse getUninitUse(ast::ExprInterface ex, ast::VarDeclInterface vd, Value v) {
                UninitUse Use(ex, isAlwaysUninit(v));

                assert(isUninitialized(v));
                if (Use.getKind() == UninitUse::Kind::Always) {
                    return Use;
                }

                // If an edge which leads unconditionally to this use did not initialize
                // the variable, we can say something stronger than 'may be uninitialized':
                // we can say 'either it's used uninitialized or you have dead code'.
                //
                // We track the number of successors of a node which have been visited, and
                // visit a node once we have visited all of its successors. Only edges where
                // the variable might still be uninitialized are followed. Since a variable
                // can't transfer from being initialized to being uninitialized, this will
                // trace out the subgraph which inevitably leads to the use and does not
                // initialize the variable. We do not want to skip past loops, since their
                // non-termination might be correlated with the initialization condition.
                //
                // For example:
                //
                //         void f(bool a, bool b) {
                // block1:   int n;
                //           if (a) {
                // block2:     if (b)
                // block3:       n = 1;
                // block4:   } else if (b) {
                // block5:     while (!a) {
                // block6:       do_work(&a);
                //               n = 2;
                //             }
                //           }
                // block7:   if (a)
                // block8:     g();
                // block9:   return n;
                //         }
                //
                // Starting from the maybe-uninitialized use in block 9:
                //  * Block 7 is not visited because we have only visited one of its two
                //    successors.
                //  * Block 8 is visited because we've visited its only successor.
                // From block 8:
                //  * Block 7 is visited because we've now visited both of its successors.
                // From block 7:
                //  * Blocks 1, 2, 4, 5, and 6 are not visited because we didn't visit all
                //    of their successors (we didn't visit 4, 3, 5, 6, and 5, respectively).
                //  * Block 3 is not visited because it initializes 'n'.
                // Now the algorithm terminates, having visited blocks 7 and 8, and having
                // found the frontier is blocks 2, 4, and 5.
                //
                // 'n' is definitely uninitialized for two edges into block 7 (from blocks 2
                // and 4), so we report that any time either of those edges is taken (in
                // each case when 'b == false'), 'n' is used uninitialized.
                llvm::SmallVector< cfg::CFGBlockInterface, 32 > Queue;
                llvm::SmallVector< unsigned, 32 > SuccsVisited(cfg.getNumBlockIDs(), 0);
                Queue.push_back(block);
                // Specify that we've already visited all successors of the starting block.
                // This has the dual purpose of ensuring we never add it to the queue, and
                // of marking it as not being a candidate element of the frontier.
                SuccsVisited[block.getBlockID()] = block.succ_size();
                while (!Queue.empty()) {
                    cfg::CFGBlockInterface B = Queue.pop_back_val();

                    // If the use is always reached from the entry block, make a note of that.
                    if (B == cfg.getEntry()) {
                        Use.setUninitAfterCall();
                    }

                    for (cfg::pred_iterator I = B.pred_begin(), E = B.pred_end(); I != E; ++I) {
                        auto Pred = dyn_cast< cfg::CFGBlockInterface >(**I);
                        if (!Pred) {
                            continue;
                        }

                        Value AtPredExit = vals.getValue(Pred, B, vd);
                        if (AtPredExit == Value::Initialized) {
                            // This block initializes the variable.
                            continue;
                        }
                        if (AtPredExit == Value::MayUninitialized &&
                            vals.getValue(B, nullptr, vd) == Value::Uninitialized) {
                            // This block declares the variable (uninitialized), and is reachable
                            // from a block that initializes the variable. We can't guarantee to
                            // give an earlier location for the diagnostic (and it appears that
                            // this code is intended to be reachable) so give a diagnostic here
                            // and go no further down this path.
                            Use.setUninitAfterDecl();
                            continue; 
                        }

                        unsigned &SV = SuccsVisited[Pred.getBlockID()];
                        if (!SV) {
                            // When visiting the first successor of a block, mark all NULL
                            // successors as having been visited.
                            for (cfg::succ_iterator SI = Pred.succ_begin(), SE = Pred.succ_end(); SI != SE; ++SI) {
                                if (!**SI) {
                                    ++SV;
                                }
                            }
                        }

                        if (++SV == Pred.succ_size()) {
                            // All paths from this block lead to the use and don't initialize the variable.
                            Queue.push_back(Pred);
                        }
                    }

                    // Scan the frontier, looking for blocks where the variable was uninitialized.
                    for (auto Block : cfg) {
                        unsigned BlockID = Block.getBlockID();
                        ast::StmtInterface Term = Block.getTerminatorStmt();
                        if (SuccsVisited[BlockID] && SuccsVisited[BlockID] < Block.succ_size() && Term) {
                            // This block inevitably leads to the use. If we have an edge from here
                            // to a post-dominator block, and the variable is uninitialized on that
                            // edge, we have found a bug.
                            for (cfg::succ_iterator I = Block.succ_begin(), E = Block.succ_end(); I != E; ++I) {
                                auto Succ = dyn_cast< cfg::CFGBlockInterface >(**I);
                                if (Succ && SuccsVisited[Succ.getBlockID()] >= Succ.succ_size() &&
                                    vals.getValue(Block, Succ, vd) == Value::Uninitialized) {
                                    // Switch cases are a special case: report the label to the caller
                                    // as the 'terminator', not the switch statement itself. Suppress
                                    // situations where no label matched: we can't be sure that's possible.
                                    if (isa< ast::SwitchStmtInterface >(Term.getOperation())) {
                                        ast::StmtInterface Label = Succ.getLabel();
                                        if (!Label || !isa< ast::SwitchCaseInterface >(Label.getOperation())) {
                                            // Might not be possible.
                                            continue;
                                        }
                                        UninitUse::Branch Branch;
                                        Branch.Terminator = Label;
                                        Branch.Output = 0; // Ingored.
                                        Use.addUninitBranch(Branch);
                                    }
                                    else {
                                        UninitUse::Branch Branch;
                                        Branch.Terminator = Term;
                                        Branch.Output = I - Block.succ_begin();
                                        Use.addUninitBranch(Branch);
                                    }
                                }
                            }
                        }
                    }
                }
                return Use;
            }

            void VisitOMPExecutableDirective(ast::OMPExecutableDirectiveInterface ED) {
                VAST_UNIMPLEMENTED;
            }

            void VisitBlockExpr(ast::BlockExprInterface be) {
                ast::BlockDeclInterface bd = getBlockDecl(be);
                for (auto I : captures(bd)) {
                    ast::VarDeclInterface *vd = I.getVariable();
                    if (!isTrackedVar(*vd)) {
                        continue;
                    }
                    if (I.isByRef()) {
                        vals[*vd] = Value::Initialized;
                        continue;
                    }
                    reportUse(be, *vd);
                }
            }

            void VisitCallExpr(ast::CallExprInterface ce) {
                if (ast::DeclInterface Callee = getCalleeDecl(ce)) {
                    if (Callee.getOperation()->hasAttr("ReturnsTwiceAttr")) {
                        // After a call to a function like setjmp or vfork, any variable which is
                        // initialized anywhere within this function may now be initialized. For
                        // now, just assume such a call initializes all variables. FIXME: Only
                        // mark variables as initialized if they have an initializer which is
                        // reachable from here.
                        vals.setAllScratchValues(Value::Initialized);
                    }
                    else if (Callee.getOperation()->hasAttr("AnalyzerNoReturnAttr")) {
                        // Functions labeled like "analyzer_noreturn" are often used to denote
                        // "panic" functions that in special debug situations can still return,
                        // but for the most part should not be treated as returning.  This is a
                        // useful annotation borrowed from the static analyzer that is useful for
                        // suppressing branch-specific false positives when we call one of these
                        // functions but keep pretending the path continues (when in reality the
                        // user doesn't care).
                        vals.setAllScratchValues(Value::Unknown);
                    }
                }
            }

            void VisitBinaryOperator(ast::BinaryOperatorInterface BO) {
                if (BO.getOpcode() == clang::BO_Assign) {
                    FindVarResult Var = findVar(BO.getLHS());
                    if (ast::VarDeclInterface VD = Var.getDecl()) {
                        vals[VD] = Value::Initialized;
                    }
                }
            }

            void VisitDeclStmt(ast::DeclStmtInterface DS) {
                for (auto DI : decls(DS)) {
                    auto VD = dyn_cast< ast::VarDeclInterface >(DI.getOperation());
                    if (VD && isTrackedVar(VD)) {
                        if (getSelfInitExpr(VD)) {
                            // If the initializer consists solely of a reference to itself, we
                            // explicitly mark the variable as uninitialized. This allows code
                            // like the following:
                            //
                            //   int x = x;
                            //
                            // to deliberately leave a variable uninitialized. Different analysis
                            // clients can detect this pattern and adjust their reporting
                            // appropriately, but we need to continue to analyze subsequent uses
                            // of the variable.
                            vals[VD] = Value::Uninitialized;
                        }
                        else if (VD.getInit()) {
                            // Treat the new variable as initialized.
                            vals[VD] = Value::Initialized;
                        }
                        else {
                            // No initializer: the variable is now uninitialized. This matters
                            // for cases like:
                            //   while (...) {
                            //     int n;
                            //     use(n);
                            //     n = 0;
                            //   }
                            // FIXME: Mark the variable as uninitialized whenever its scope is
                            // left, since its scope could be re-entered by a jump over the
                            // declaration.
                            vals[VD] = Value::Uninitialized;
                        }
                    }
                }
            }

            std::vector< ast::ExprInterface > outputs(ast::GCCAsmStmtInterface) {
                VAST_UNIMPLEMENTED;
            }

            void VisitGCCAsmStmt(ast::GCCAsmStmtInterface as) {
                // An "asm goto" statement is a terminator that may initialize some variables.
                if (!as.isAsmGoto()) {
                    return;
                }

                ast::ASTContextInterface C = ac.getASTContext();
                for (ast::ExprInterface O : outputs(as)) {
                    ast::ExprInterface Ex = dyn_cast< ast::ExprInterface >(stripCasts(C, O));

                    // Strip away any unary operators. Invalid l-values are reported by other
                    // semantic analysis passes.
                    while (auto UO = dyn_cast< ast::UnaryOperatorInterface >(Ex.getOperation())) {
                        Ex = dyn_cast< ast::ExprInterface >(stripCasts(C, UO.getSubExpr()));
                    }
                    
                    // Mark the variable as potentially uninitialized for those cases where
                    // it's used on an indirect path, where it's not guaranteed to be defined.
                    if (ast::VarDeclInterface VD = findVar(Ex).getDecl()) {
                        if (vals[VD] != Value::Initialized) {
                            vals[VD] = Value::MayUninitialized;
                        }
                    }
                }
            }
        };
    
    } // namespace
    
    //------------------------------------------------------------------------====//
    // High-level "driver" logic for uninitialized values analysis.
    //====------------------------------------------------------------------------//

    /// PruneBlocksHandler is a special UninitVariablesHandler that is used
    /// to detect when a CFGBlock has any *potential* use of an uninitialized
    /// variable.  It is mainly used to prune out work during the final
    struct PruneBlocksHandler : public UninitVariablesHandler {
        /// Records if a CFGBlock had a potential use of an uninitialized variable.
        llvm::BitVector hadUse;

        /// Records if any CFGBlock had a potential use of an uninitialized variable.
        bool hadAnyUse = false;

        /// The current block to scribble use information.
        unsigned currentBlock = 0;

        PruneBlocksHandler(unsigned numBlocks) : hadUse(numBlocks, false) {}

        ~PruneBlocksHandler() override = default;

        void handleUseOfUninitVariable(ast::VarDeclInterface vd,
                                       const UninitUse &use) override {
            hadUse[currentBlock] = true;
            hadAnyUse = true;
        }

        void handleConstRefUseOfUninitVariable(ast::VarDeclInterface vd,
                                               const UninitUse &use) override {
            hadUse[currentBlock] = true;
            hadAnyUse = true;
        }

        /// Called when the uninitialized variable analysis detects the
        /// idiom 'int x = x'.  All other uses of 'x' within the initializer
        /// are handled by handleUseOfUninitVariable.
        void handleSelfInit(ast::VarDeclInterface vd) override {
            hadUse[currentBlock] = true;
            hadAnyUse = true;
        }
    };

    static bool runOnBlock(auto block, auto cfg, auto ac, auto &vals, const auto &classification,
                           auto &wasAnalyzed, auto &handler) {
        wasAnalyzed[block.getBlockID()] = true;
        vals.resetScratch();
        // Merge in values of predecessor blocks.
        bool isFirst = true;

        for (cfg::pred_iterator I = block.pred_begin(), E = block.pred_end(); I != E; ++I) {
            auto pred = dyn_cast< cfg::CFGBlockInterface >(**I);
            if (!pred) {
                continue;
            }
            if (wasAnalyzed[pred.getBlockID()]) {
                vals.mergeIntoScratch(vals.getValueVector(pred), isFirst);
                isFirst = false;
            }
        }
        
        // Apply the transfer function.
        TransferFunctions tf(vals, cfg, block, ac, classification, handler);
        for (auto I : block) {
            if (std::optional< cfg::CFGStmtInterface > cs = I. template getAs< cfg::CFGStmtInterface >()) {
                tf.Visit(cs->getStmt());
            }
        }

        cfg::CFGTerminatorInterface terminator = block.getTerminator();
        if (auto as = dyn_cast< ast::GCCAsmStmtInterface >(terminator.getStmt().getOperation())) {
            if (as.isAsmGoto()) {
                tf.Visit(as);
            }
        }
        return vals.updateValueVectorWithScratch(block);
    }


    void runUninitializedVariablesAnalysis(
            ast::DeclContextInterface dc,
            cfg::CFGInterface cfg,
            AnalysisDeclContextInterface ac,
            UninitVariablesHandler &handler, 
            UninitVariablesAnalysisStats &stats) {

        CFGBlockValues vals(cfg);
        vals.computeSetOfDeclarations(dc);

        if (vals.hasNoDeclarations())
           return;

        stats.NumVariablesAnalyzed = vals.getNumEntries();

        // Precompute which expressions are uses and which are initializations.
        ClassifyRefs classification(ac);
        cfg.VisitBlockStmts(classification);

        // Mark all variables uninitialized at the entry.
        cfg::CFGBlockInterface entry = cfg.getEntry();
        ValueVector &vec = vals.getValueVector(entry);
        const unsigned n = vals.getNumEntries();
        for (unsigned j = 0; j < n; ++j) {
            vec[j] = Value::Uninitialized;
        }

        // Proceed with the worklist.
        ForwardDataflowWorklist worklist(cfg, ac);
        llvm::BitVector previouslyVisited(cfg.getNumBlockIDs());
        worklist.enqueueSuccessors(cfg.getEntry());
        llvm::BitVector wasAnalyzed(cfg.getNumBlockIDs(), false);
        wasAnalyzed[cfg.getEntry().getBlockID()] = true;
        PruneBlocksHandler PBH(cfg.getNumBlockIDs());

        while (cfg::CFGBlockInterface block = worklist.dequeue()) {
            PBH.currentBlock = block.getBlockID();
            
            // Did the block change?
            bool changed = runOnBlock(block, cfg, ac, vals, classification, wasAnalyzed, PBH);
            ++stats.NumBlockVisits;
            
            if (changed || !previouslyVisited[block.getBlockID()]) {
                worklist.enqueueSuccessors(block);
            }

            previouslyVisited[block.getBlockID()] = true;
        }

        if (!PBH.hadAnyUse) {
            return;
        }

        // Run through the blocks one more time, and report uninitialized variables.
        for (auto block : cfg) {
            if (PBH.hadUse[block.getBlockID()]) {
                runOnBlock(block, cfg, ac, vals, classification, wasAnalyzed, handler);
                ++stats.NumBlockVisits;
            }
        }
    }

    UninitVariablesHandler::~UninitVariablesHandler() = default;
} // namespace vast::analysis
