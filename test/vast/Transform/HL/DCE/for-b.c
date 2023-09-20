// RUN: %vast-cc --ccopts -xc --from-source %s | %vast-opt --vast-hl-dce | %file-check %s

void fn()
{
    for (int i = 0; i < 15; ++i)
    {
        // CHECK:   } then {
        // CHECK-NEXT:     hl.break
        // CHECK-NEXT:   }
        // CHECK-NEXT:   hl.continue
        // CHECK-NEXT: }
        // CHECK-NEXT: hl.const
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
