#pragma once

namespace vast::ast {

    class DeclInterface;
}

namespace vast::analyses {

    class decl_interface_iterator {
        ast::DeclInterface *Current = nullptr;

    public:
        decl_interface_iterator() = default;
        explicit decl_interface_iterator(ast::DeclInterface *C) : Current(C) {}

        ast::DeclInterface &operator*() const;
        ast::DeclInterface *operator->() const;
        decl_interface_iterator &operator++();
        decl_interface_iterator operator++(int);
        friend bool operator==(decl_interface_iterator, decl_interface_iterator);
        friend bool operator!=(decl_interface_iterator, decl_interface_iterator);
        mlir::Operation *get_current_op() const;
    };

    template< typename SpecificDecl >
    class specific_decl_interface_iterator {
        using decl_interface_iterator = vast::analyses::decl_interface_iterator;
        decl_interface_iterator Current;

        void SkipToNextDecl() {
            while (*Current && !isa< SpecificDecl >(Current.get_current_op())) {
                ++Current;
            }
        }

    public:
        specific_decl_interface_iterator() = default;
        explicit specific_decl_interface_iterator(decl_interface_iterator C) : Current(C) {
            SkipToNextDecl();
        }

        SpecificDecl operator*() const { return dyn_cast< SpecificDecl >(Current.get_current_op()); }
        SpecificDecl operator->() const { return **this; }

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
} // namespace vast::analyses
