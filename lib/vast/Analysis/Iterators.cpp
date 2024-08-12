#include "vast/Analysis/Iterators.hpp"
#include "vast/Interfaces/AST/DeclInterface.hpp"
#include "vast/Interfaces/CFG/CFGInterface.hpp"

namespace vast::analysis {

    mlir::Operation *decl_interface_iterator::operator*()  const { return Current; }
    mlir::Operation *decl_interface_iterator::operator->() const { return Current; }

    decl_interface_iterator &decl_interface_iterator::operator++() {
        auto dc = dyn_cast< ast::DeclInterface >( Current );
        if ( dc )
            Current = dc.getNextDeclInContext().getOperation();
        return *this;
    }

    decl_interface_iterator decl_interface_iterator::operator++(int) {
        decl_interface_iterator tmp(*this);
        ++(*this);
        return tmp;
    }

    bool operator==(decl_interface_iterator x, decl_interface_iterator y) {
        return x.Current == y.Current;
    }

    bool operator!=(decl_interface_iterator x, decl_interface_iterator y) {
        return x.Current != y.Current;
    }
} // namespace vast::analysis

namespace vast::cfg {

    AdjacentBlock::AdjacentBlock(cfg::CFGBlockInterface *B, bool isReachable)
        : ReachableBlock(isReachable ? B->getOperation() : nullptr),
          UnreachableBlock(!isReachable ? B->getOperation() : nullptr,
                           B && isReachable ? Kind::AB_Normal : Kind::AB_Unreachable) {}

    AdjacentBlock::AdjacentBlock(cfg::CFGBlockInterface *B, cfg::CFGBlockInterface *AlternateBlock)
        : ReachableBlock(B->getOperation()),
          UnreachableBlock(B == AlternateBlock ? nullptr : AlternateBlock->getOperation(),
                           B == AlternateBlock ? Kind::AB_Alternate : Kind::AB_Normal) {}

} // namespace vast::cfg
