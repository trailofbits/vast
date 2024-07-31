#include "vast/Analyses/Iterators.hpp"

namespace vast::analyses {

    ast::DeclInterface decl_interface_iterator::operator*() const { return *Current; }
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

    friend bool decl_interface_iterator::operator==(decl_interface_iterator x, decl_interface_iterator y) {
        return x.Current == y.Current;
    }

    friend bool decl_interface_iterator::operator!=(decl_interface_iterator x, decl_interface_iterator y) {
        return x.Current != y.Current;
    }

    template< typename SpecificDecl >
    void specific_decl_interface_iterator< SpecificDecl >::SkipToNextDecl() {
        while (*Current && !isa< SpecificDecl >(*Current)) {
            ++Current;
        }
    }

    SpecificDecl operator*() const { return cast< SpecificDelc >(*Current); }
} // namespace vast::analyses
