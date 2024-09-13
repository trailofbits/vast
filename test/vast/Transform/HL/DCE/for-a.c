// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-dce | %file-check %s

void fn()
{
    for (int i = 0; i < 15; ++i)
    {
        // CHECK: } do {
        // CHECK-NEXT:   [[V1:%[0-9]+]] = hl.ref @i
        // CHECK-NEXT:   [[V2:%[0-9]+]] = hl.pre.inc [[V1]] : !hl.lvalue<!hl.int>
        // CHECK-NEXT:   hl.break
        // CHECK-NEXT: }
        ++i;
        break;
        --i;
        continue;
    }
}
