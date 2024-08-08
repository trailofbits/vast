#include "vast/Analyses/Iterators.hpp"
#include "vast/Interfaces/AST/DeclInterface.hpp"
#include "vast/Interfaces/CFG/CFGInterface.hpp"

namespace vast::analyses {

    ast::DeclInterface &decl_interface_iterator::operator*() const { return *Current; }
    ast::DeclInterface *decl_interface_iterator::operator->() const { return Current; }

    decl_interface_iterator &decl_interface_iterator::operator++() {
        Current = Current->getNextDeclInContext();
        return *this;
    }

    decl_interface_iterator decl_interface_iterator::operator++(int) {
        decl_interface_iterator tmp(*this);
        ++(*this);
        return tmp;
    }

    mlir::Operation *decl_interface_iterator::get_current_op() const {
        return Current->getOperation();
    }

    bool operator==(const decl_interface_iterator &x, const decl_interface_iterator &y) {
        return x.Current == y.Current;
    }

    bool operator!=(const decl_interface_iterator &x, const decl_interface_iterator &y) {
        return x.Current != y.Current;
    }
} // namespace vast::analyses

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
