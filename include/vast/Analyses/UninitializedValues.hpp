// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/Interfaces/AST/DeclInterface.hpp"
#include "vast/Interfaces/AST/TypeInterface.hpp"
#include "vast/Interfaces/AST/StmtInterface.hpp"
#include "vast/Interfaces/AST/ExprInterface.hpp"

#include "vast/Analyses/CFG.hpp"
#include "clang/AST/StmtVisitor.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PackedVector.h"
#include "llvm/ADT/SmallVector.h"
#include "vast/Analyses/FlowSensitive/DataflowWorklist.hpp"

#include <optional>

namespace vast::analyses {

    // TODO
    static ast::RecordDeclInterface getAsRecordDecl(ast::QualTypeInterface) {
        return {};
    }

    // TODO
    static bool isZeroSize(ast::FieldDeclInterface FD) {
        return false;
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
                /*
                    unsigned count = 0;
                    typename ast::DeclContextInterface:: template specific_decl_iterator
                        < ast::VarDeclInterface > I(dc.decls_begin()),
                                                  E(dc.decls_end());
                    for ( ; I != E; ++I) {
                        ast::VarDeclInterface vd = I;
                        if (isTrackedVar(vd, &dc)) {
                            map[vd] = count++;
                        }
                    }
                */
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

        template< typename CFG_T, typename CFGBlock_T >
        class CFGBlockValues_T {
            CFG_T &cfg;
            clang::SmallVector< ValueVector, 8 > vals;
            ValueVector scratch;
            DeclToIndex declToIndex;

        public:
            CFGBlockValues_T(CFG_T &c) : cfg(c), vals(0) {}

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

            ValueVector &getValueVector(const CFGBlock_T *block) {
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

            bool updateValueVectorWithScratch(const CFGBlock_T *block) {
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
    
        class FindVarResult_T {
            ast::VarDeclInterface vd;
            ast::DeclRefExprInterface dr;

        public:
            FindVarResult_T(ast::VarDeclInterface vd, ast::DeclRefExprInterface dr) : vd(vd), dr(dr) {}

            ast::DeclRefExprInterface getDeclRefExpr() {
                return dr;
            }

            ast::VarDeclInterface getDecl() {
                return vd;
            }
        };

    } // namespace

    // TODO
    static mlir::Operation *getDecl(ast::DeclRefExprInterface) {
        return {};
    }

    // TODO
    static bool compare(ast::ValueDeclInterface, ast::VarDeclInterface) {
        return false;
    }

    // TODO
    static mlir::Operation *stripCasts(ast::ASTContextInterface AC, ast::ExprInterface) {
        return nullptr;
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
        template< typename AnalysisDeclContext_T >
        class ClassifyRefs_T : public clang::StmtVisitor<
                               ClassifyRefs_T< AnalysisDeclContext_T > > {
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
                return isTrackedVar(VD, DC);
            }

            void classify(ast::ExprInterface E, Class C) {
            
            }

            std::vector< ast::DeclInterface > decls(ast::DeclStmtInterface DS) {
                return {};
            }

        public:
            ClassifyRefs_T(AnalysisDeclContext_T &AC) : DC(cast< ast::DeclContextInterface >(AC.getDecl())) {}

            void operator()(ast::StmtInterface S) {
                // TODO: Najst, co za Visit to vobec je.
                // Visit(S);
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
        class TransferFunctionsT : public clang::StmtVisitor<
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

            // TODO
            void VisitDeclRefExpr(ast::DeclRefExprInterface dr) {
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
            }

            void reportUse(ast::ExprInterface ex, ast::VarDeclInterface vd) {
                Value v = vals[vd];
                if (isUninitialized(v)) {
                // handler.handleUseOfUninitVariable(vd, getUninitUse(ex, vd, v));
                }
            }
        };
    
    } // namespace
    
    //------------------------------------------------------------------------====//
    // High-level "driver" logic for uninitialized values analysis.
    //====------------------------------------------------------------------------//
    
    static bool runOnBlock() {
        return false; 
    }

    class UninitUseT {};

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


    struct UninitVariablesAnalysisStats_T {
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

    template< typename CFG_T, typename AnalysisDeclContext_T,
              typename CFGBlock_T, typename CFGBlockValues_T >
    void runUninitializedVariablesAnalysis(
            ast::DeclContextInterface dc,
            const CFG_T &cfg,
            AnalysisDeclContext_T &ac,
            UninitVariablesHandler &handler, 
            UninitVariablesAnalysisStats_T &stats) {

        CFGBlockValues_T vals(cfg);
        vals.computeSetOfDeclarations(dc);

        if (vals.hasNoDeclarations())
           return;

        stats.NumVariablesAnalyzed = vals.getNumEntries();

        // Precompute which expressions are uses and which are initializations.
        ClassifyRefs_T classification(ac);
        cfg.VisitBlockStmts(classification);

        // Mark all variables uninitialized at the entry.
        const CFGBlock_T &entry = cfg.getEntry();
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

        while (const CFGBlock_T *block = worklist.dequeue()) {
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
