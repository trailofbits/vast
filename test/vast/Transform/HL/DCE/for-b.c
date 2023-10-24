// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-dce | %file-check %s

void fn()
{
    for (int i = 0; i < 15; ++i)
    {
        // CHECK:   } then {
        // CHECK-NEXT:     hl.break
        // CHECK-NEXT:   }
        // CHECK-NEXT:   hl.continue
        // CHECK-NEXT: }
        // CHECK: hl.const
        // CHECK-NEXT: core.implicit.return
        if (i == 5)
        {
            break;
            return;
        }

        continue;
        return;
    }
}
