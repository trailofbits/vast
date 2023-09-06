// RUN: %vast-cc --ccopts -xc --from-source %s | %vast-opt --vast-hl-dce | FileCheck %s
// REQUIRES: return

void fn()
{
    int x = 12;
    // CHECK:     hl.cond.yield [[V2:%[0-9]+]] : !hl.int
    // CHECK-NEXT:   } then {
    // CHECK-NEXT:     hl.return
    // CHECK-NEXT:   } else {
    // CHECK-NEXT:     hl.return
    // CHECK-NEXT:   }
    // CHECK-NEXT:   hl.return
    // CHECK-NEXT: }
    if (x)
    {
        return;
        ++x;
    } else {
        return;
        int z = 5;
    }
}
