#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/Analysis/Support/BumpVector.h>
#include <llvm/ADT/iterator.h>
#include <llvm/ADT/iterator_range.h>
#include <mlir/IR/Operation.h>
VAST_RELAX_WARNINGS

#include <iterator>

namespace vast::ast {

    class DeclInterface;
    class ExprInterface;
    class StmtInterface;
}

namespace vast::analyses {

    class decl_interface_iterator {
        mlir::Operation *Current = nullptr;

    public:
        decl_interface_iterator() = default;
        explicit decl_interface_iterator(mlir::Operation *C) : Current(C) {}

        mlir::Operation *operator*() const; 
        mlir::Operation *operator->() const;
        decl_interface_iterator &operator++();
        decl_interface_iterator  operator++(int);

        friend bool operator==(decl_interface_iterator, decl_interface_iterator);
        friend bool operator!=(decl_interface_iterator, decl_interface_iterator);
    };

    template< typename SpecificDecl >
    class specific_decl_interface_iterator {
        using decl_interface_iterator = vast::analyses::decl_interface_iterator;
        decl_interface_iterator Current;

        void SkipToNextDecl() {
            while (*Current && !isa< SpecificDecl >(*Current)) {
                ++Current;
            }
        }

    public:
        specific_decl_interface_iterator() = default;
        explicit specific_decl_interface_iterator(decl_interface_iterator C) : Current(C) {
            SkipToNextDecl();
        }

        SpecificDecl operator*() const { return dyn_cast< SpecificDecl >(*Current); }
        SpecificDecl operator->() const { return *this; }

        specific_decl_interface_iterator &operator++() {
            ++Current;
            SkipToNextDecl();
            return *this;
        }

        specific_decl_interface_iterator operator++(int) {
            specific_decl_interface_iterator tmp(*this);
            ++(*this);
            return tmp;
        }
        
        friend bool operator==(const specific_decl_interface_iterator &x,
                               const specific_decl_interface_iterator &y) {
            return x.Current == y.Current;
        }

        friend bool operator!=(const specific_decl_interface_iterator &x,
                               const specific_decl_interface_iterator &y) {
            return x.Current != y.Current;
        }
    };

    template< typename T, typename TPtr = T *, typename StmtPtr = ast::StmtInterface * >
    struct CastIterator
        : llvm::iterator_adaptor_base< CastIterator< T, TPtr, StmtPtr >, StmtPtr *,
                                       std::random_access_iterator_tag, T > {
        using Base = typename CastIterator::iterator_adaptor_base;

        CastIterator() : Base(nullptr) {}
        CastIterator(StmtPtr *I) : Base(I) {}

        typename Base::value_type operator*() const {
            return cast_or_null< T >((*this->I)->getOperation());
        }
    };

    using ExprInterfaceIterator = CastIterator< ast::ExprInterface >;
    using call_expr_arg_iterator = ExprInterfaceIterator;
} // namespace vast::analyses

namespace vast::cfg {

    class CFGBlockInterface;
    class CFGElementInterface;
} 

namespace vast::cfg {

    class AdjacentBlock {
        enum Kind {
            AB_Normal,
            AB_Unreachable,
            AB_Alternate
        };

        mlir::Operation *ReachableBlock;
        llvm::PointerIntPair< mlir::Operation *, 2 > UnreachableBlock;

    public:
        /// Construct an AdjacentBlock with a possibly unreachable block.
        AdjacentBlock(cfg::CFGBlockInterface *B, bool isReachable);

        /// Construct an AdjacentBlock with a reachable block and an alternate
        /// unreachable block.
        AdjacentBlock(cfg::CFGBlockInterface *B, cfg::CFGBlockInterface *AlternateBlock);

        /// Get the reachable block, if one exists.
        mlir::Operation *getReachableBlock() const {
            return ReachableBlock;
        }

        /// Get the potentially unreachable block.
        mlir::Operation *getPossiblyUnreachableBlock() const {
            return UnreachableBlock.getPointer();
        }

        /// Provide an implicit conversion to cfg::CFGBlockInterface so that
        /// AdjacentBlock can be substituted for cfg::CFGBlockInterface.
        /*
        operator mlir::Operation*() const {
            return getReachableBlock();
        }
        */

        mlir::Operation *operator*() const {
            return getReachableBlock();
        }

        mlir::Operation *operator->() const {
            return getReachableBlock();
        }

        bool isReachable() const {
            Kind K = (Kind) UnreachableBlock.getInt();
            return K == Kind::AB_Normal || K == Kind::AB_Alternate;
        }
    };

    using AdjacentBlocks = clang::BumpVector< AdjacentBlock >;
    using pred_iterator = AdjacentBlocks::iterator;

    using CFGBlockListTy = clang::BumpVector< CFGBlockInterface >;
    using CFGIterator = CFGBlockListTy::iterator;

    using succ_iterator = AdjacentBlocks::iterator;
    using succ_range = llvm::iterator_range< succ_iterator >;

    using CFGBlockIterator = std::reverse_iterator< clang::BumpVector< CFGElementInterface >::iterator >;
} // namespace vast::cfg
