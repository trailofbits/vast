// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s -check-prefix VAR
// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s -check-prefix TYPEOF

static __attribute__((always_inline)) void usb_fill_int_urb(int interval) {
    typeof((typeof(interval)) ({
    // TYPEOF-COUNT-5: hl.typeof.expr "
    // TYPEOF-NOT:     hl.typeof.expr "
    // VAR:         hl.var "_v"
    // VAR-NEXT:    hl.expr
    // VAR-NEXT:    hl.ref
    // VAR-NEXT:    hl.value.yield
    // VAR-NEXT:    }
    // VAR-NEXT:    hl.implicit_cast
    // VAR-NEXT:    hl.value.yield
    // VAR-NEXT:    }
        typeof(interval) _v = (interval);
        typeof(1) _w        = (1);
        _v > _w ? _v : _w;
    })) _x        = 5;
    typeof(16) _y = (16);
    _x < _y ? _x : _y;
}
