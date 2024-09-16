// RUN: %vast-cc1 -vast-emit-mlir-after=vast-refs-to-ssa %s -o - | %file-check %s

int main()
{
    // CHECK: [[V0:%[0-9]+]] = ll.cell @a : !hl.lvalue<si32>
    int a;

    // CHECK: [[V1:%[0-9]+]] = ll.cell @b : !hl.lvalue<si32>
    // CHECK: [[V2:%[0-9]+]] = hl.const #core.integer<1> : si32
    // CHECK: [[V3:%[0-9]+]] = ll.cell_init [[V1]], [[V2]] : (!hl.lvalue<si32>, si32) -> !hl.lvalue<si32>
    int b = 1;
}
