// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-dce --vast-hl-lower-types --vast-hl-to-ll-cf | FileCheck %s

void fn()
{
    // CHECK: "ll.scope" ({
    // CHECK-NEXT: ll.br ^bb1

    // CHECK: ^bb1:  // pred: ^bb0
    // CHECK: "ll.cond_scope_ret"(%5)[^bb2] : (i1) -> ()

    // CHECK: ^bb2:  // pred: bb1
    // CHECK: "ll.cond_br"(%10)[^bb3, ^bb4] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (i1) -> ()

    // CHECK: ^bb3:  // pred ^b2
    // CHECK: "ll.scope_ret"() : () -> ()

    // CHECK: ^bb4:  // pred ^bb2
    // CHECK: "ll.scope_recurse"() : () -> ()

    // CHECK: }) : () -> ()
    for (int i = 0; i < 15; ++i)
    {
        ++i;
        break;
    }
}
