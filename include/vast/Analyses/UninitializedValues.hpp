// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/Analyses/Iterators.hpp"
#include "vast/Analyses/Clang/CFG.hpp"
#include "vast/Analyses/Clang/FlowSensitive/DataflowWorklist.hpp"
#include "vast/Interfaces/AST/DeclInterface.hpp"
#include "vast/Interfaces/AST/TypeInterface.hpp"
#include "vast/Interfaces/AST/StmtInterface.hpp"
#include "vast/Interfaces/AST/ExprInterface.hpp"
#include "vast/Interfaces/AST/StmtVisitor.h"
#include "vast/Interfaces/Analyses/AnalysisDeclContextInterface.hpp"

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/PackedVector.h>
#include <llvm/ADT/PointerIntPair.h>
#include <llvm/ADT/SmallVector.h>
#include <optional>

namespace vast::analyses {

    /// A use of a variable, which might be uninitialized.
    class UninitUseT {
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
        UninitUseT(ast::ExprInterface User, bool AlwaysUninit)
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
            if (auto RD = getAsRecordDecl(ty))
                return recordIsNotEmpty(RD);
            return ty.isScalarType() || ty.isVectorType() || ty.isRVVSizelessBuiltinType();
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

    enum class Value {
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

        template< typename CFGT, typename CFGBlockT >
        class CFGBlockValuesT {
            CFGT &cfg;
            clang::SmallVector< ValueVector, 8 > vals;
            ValueVector scratch;
            DeclToIndex declToIndex;

        public:
            CFGBlockValuesT(CFGT &c) : cfg(c), vals(0) {}

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

            ValueVector &getValueVector(const CFGBlockT *block) {
                return vals[block->getBlockID()];
            }

            void resetScratch() {
                scratch.reset();
            }

            void mergeIntoScratch(const ValueVector &source, bool isFirst) {
                if (isFirst) {
                    scratch = source;
                } else {
                    scratch |= source;
                }
            }

            bool updateValueVectorWithScratch(const CFGBlockT *block) {
                ValueVector &dst = getValueVector(block);
                bool changed = (dst != scratch);
                if (changed)
                    dst = scratch;
                return changed;
            }

            ValueVector::reference operator[](ast::VarDeclInterface vd) {
                return scratch[*declToIndex.getValueIndex(vd)];
            }
        };
    
    } // namespace
    
    //------------------------------------------------------------------------====//
    // Classification of DeclRefExprs as use or initialization.
    //====------------------------------------------------------------------------//

    namespace {
    
        class FindVarResultT {
            ast::VarDeclInterface vd;
            ast::DeclRefExprInterface dr;

        public:
            FindVarResultT(ast::VarDeclInterface vd, ast::DeclRefExprInterface dr) : vd(vd), dr(dr) {}

            ast::DeclRefExprInterface getDeclRefExpr() {
                return dr;
            }

            ast::VarDeclInterface getDecl() {
                return vd;
            }
        };

    } // namespace

    static mlir::Operation *getDecl(ast::DeclRefExprInterface) {
        VAST_UNIMPLEMENTED;
    }

    static bool compare(ast::ValueDeclInterface, ast::VarDeclInterface) {
        VAST_UNIMPLEMENTED;
    }

    static mlir::Operation *stripCasts(ast::ASTContextInterface AC, ast::ExprInterface) {
        VAST_UNIMPLEMENTED;
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
    template< typename FindVarResultT >
    static FindVarResultT findVar(ast::ExprInterface E, ast::DeclContextInterface DC) {
        auto DRE = dyn_cast< ast::DeclRefExprInterface >(stripCasts(DC.getParentASTContext(), E));
        if (DRE) {
            auto VD = dyn_cast< ast::VarDeclInterface >(getDecl(DRE));
            if (VD) {
                if (isTrackedVar(VD, DC)) {
                    return FindVarResultT(VD, DRE);
                }
            }
        }
        return FindVarResultT({}, {});
    }
    
    namespace {
        
        /// Classify each DeclRefExpr as an initialization or a use. Any
        /// DeclRefExpr which isn't explicitly classified will be assumed to have
        /// escaped the analysis and will be treated as an initialization.
        class ClassifyRefsT : public ast::StmtVisitor< ClassifyRefsT > {
        using base = ast::StmtVisitor< ClassifyRefsT >;
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
                return vast::analyses::isTrackedVar(VD, DC);
            }

            void classify(ast::ExprInterface E, Class C) {
                VAST_UNIMPLEMENTED; 
            }

            std::vector< ast::DeclInterface > decls(ast::DeclStmtInterface DS) {
                VAST_UNIMPLEMENTED;
            }

        public:
            ClassifyRefsT(AnalysisDeclContextInterface AC)
                : DC(cast< ast::DeclContextInterface >(AC.getDecl().getOperation())) {}

            void operator()(ast::StmtInterface S) {
                base::base::Visit(S);
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
        };
    } // namespace

    //------------------------------------------------------------------------====//
    // Transfer function for uninitialized values analysis.
    //====------------------------------------------------------------------------//
    
    namespace {
    
        template< typename CFGBlockValuesT, typename CFGT, typename AnalysisDeclContextT,
                  typename ClassifyRefsT, typename ObjCNoReturnT, typename UninitVariablesHandlerT,
                  typename CFGBlockT >
        class TransferFunctionsT : public ast::StmtVisitor<
                                   TransferFunctionsT< CFGBlockValuesT, CFGT, AnalysisDeclContextT,
                                                       ClassifyRefsT, ObjCNoReturnT, UninitVariablesHandlerT,
                                                       CFGBlockT >
                                   > {
            CFGBlockValuesT &vals;
            const CFGT &cfg;
            const CFGBlockT *block;
            AnalysisDeclContextT &ac;
            const ClassifyRefsT &classification;
            ObjCNoReturnT objCNoReturn;
            UninitVariablesHandlerT &handler;

        public:
            TransferFunctionsT(CFGBlockValuesT &vals, const CFGT &cfg,
                    const CFGBlockT *block, AnalysisDeclContextT &ac,
                    const ClassifyRefsT &classification,
                    UninitVariablesHandlerT &handler)
              : vals(vals), cfg(cfg), block(block), ac(ac),
                classification(classification), objCNoReturn(ac.getASTContext()),
                handler(handler) {}

            void VisitDeclRefExpr(ast::DeclRefExprInterface dr) {
                VAST_UNIMPLEMENTED;
                /*
                switch (&classification.get(dr)) {
                    case ClassifyRefsT::Class::Ignore:
                        return;
                    
                    case ClassifyRefsT::Class::Use:
                        return;

                    case ClassifyRefsT::Class::Init:
                        return;

                    case ClassifyRefsT::Class::SelfInit:
                        return;

                    case ClassifyRefsT::Class::ConstRefUse:
                        return;
                }
                */
            }

            void reportUse(ast::ExprInterface ex, ast::VarDeclInterface vd) {
                VAST_UNIMPLEMENTED;
                /*
                Value v = vals[vd];
                if (isUninitialized(v)) {
                // handler.handleUseOfUninitVariable(vd, getUninitUse(ex, vd, v));
                }
                */
            }
        };
    
    } // namespace
    
    //------------------------------------------------------------------------====//
    // High-level "driver" logic for uninitialized values analysis.
    //====------------------------------------------------------------------------//
    
    static bool runOnBlock() {
        return false; 
    }

    // osablonovat triedu
    class UninitVariablesHandler {
    public:
        UninitVariablesHandler() = default;
        virtual ~UninitVariablesHandler();

        /// Called when the uninitialized variable is used at the given expression.
        virtual void handleUseOfUninitVariable(ast::VarDeclInterface vd,
                                               const UninitUseT &use) {}

        /// Called when the uninitialized variable is used as const refernce argument.
        virtual void handleConstRefUseOfUninitVariable(ast::VarDeclInterface vd,
                                                       const UninitUseT &use) {}

        /// Called when the uninitialized variable analysis detects the
        /// idiom 'int x = x'.  All other uses of 'x' within the initializer
        /// are handled by handleUseOfUninitVariable.
        virtual void handleSelfInit(ast::VarDeclInterface vd) {}
    };


    struct UninitVariablesAnalysisStatsT {
        unsigned NumVariablesAnalyzed;
        unsigned NumBlockVisits;
    };

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
                                       const UninitUseT &use) override {
            hadUse[currentBlock] = true;
            hadAnyUse = true;
        }

        void handleConstRefUseOfUninitVariable(ast::VarDeclInterface vd,
                                               const UninitUseT &use) override {
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

    template< typename CFGT, typename CFGBlockT, typename CFGBlockValuesT >
    void runUninitializedVariablesAnalysis(
            ast::DeclContextInterface dc,
            const CFGT &cfg,
            AnalysisDeclContextInterface &ac,
            UninitVariablesHandler &handler, 
            UninitVariablesAnalysisStatsT &stats) {

        CFGBlockValuesT vals(cfg);
        vals.computeSetOfDeclarations(dc);

        if (vals.hasNoDeclarations())
           return;

        stats.NumVariablesAnalyzed = vals.getNumEntries();

        // Precompute which expressions are uses and which are initializations.
        ClassifyRefsT classification(ac);
        cfg.VisitBlockStmts(classification);

        // Mark all variables uninitialized at the entry.
        const CFGBlockT &entry = cfg.getEntry();
        ValueVector &vec = vals.getValueVector(&entry);
        const unsigned n = vals.getNumEntries();
        for (unsigned j = 0; j < n; ++j) {
            vec[j] = Value::Uninitialized;
        }

        // Proceed with the worklist.
        ForwardDataflowWorklist worklist(cfg, ac);
        llvm::BitVector previouslyVisited(cfg.getNumBlockIDs());
        worklist.enqueueSuccessors(&cfg.getEntry());
        llvm::BitVector wasAnalyzed(cfg.getNumBlockIDs(), false);
        wasAnalyzed[cfg.getEntry().getBlockID()] = true;
        PruneBlocksHandler PBH(cfg.getNumBlockIDs());

        while (const CFGBlockT *block = worklist.dequeue()) {
            PBH.currentBlock = block->getBlockID();
            
            ++stats.NumBlockVisits;
        }

        if (!PBH.hadAnyUse) {
            return;
        }

        // Run through the blocks one more time, and report uninitialized variables.
        for (const auto *block : cfg) {
            if (PBH.hadUse[block->getBlockID()]) {
                
                ++stats.NumBlockVisits;
            }
        }
    }

    UninitVariablesHandler::~UninitVariablesHandler() = default;
} // namespace vast::analyses
