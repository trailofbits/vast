// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-lower-types --vast-hl-structs-to-tuples --vast-hl-to-scf | FileCheck %s

int fn()
{
    int x = 5;

    // CHECK:  scf.while : () -> () {
    // CHECK:   scf.condition(%6)
    // CHECK: } do {
    while (x != 0)
    {
        // CHECK:   scf.if %6 {
        if (x == 4)
        {
            // CHECK:     hl.scope {
            // CHECK:       hl.return %8 : i32
            // CHECK:     }
            return x;
        // CHECK:   } else {
        } else {
            // CHECK:     hl.scope {
            // CHECK:       hl.return %7 : i32
            // CHECK:     }
            // CHECK:   }
            return 5;
        }
    // CHECK:   scf.yield
    // CHECK: }
    }
    return 0;
}
