// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-dce --vast-hl-lower-types --vast-hl-to-ll-vars | %file-check %s

int main()
{
    // CHECK: [[V0:%[0-9]+]] = ll.uninitialized_var : !hl.lvalue<si32>
    int a;

    // CHECK: [[V1:%[0-9]+]] = ll.uninitialized_var : !hl.lvalue<si32>
    // CHECK: [[V2:%[0-9]+]] = hl.const #core.integer<1> : si32
    // CHECK: [[V3:%[0-9]+]] = ll.initialize [[V1]], [[V2]] : (!hl.lvalue<si32>, si32) -> !hl.lvalue<si32>
    int b = 1;
}
