// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

int comma1(int a, int b)
{
    // CHECK: hl.bin.comma [[V1:%[0-9]+]], [[V2:%[0-9]+]] : (!hl.int, !hl.int) -> !hl.int
    return a++, b;
}
int comma2(long a, int b)
{
    // CHECK: hl.bin.comma [[V1:%[0-9]+]], [[V2:%[0-9]+]] : (!hl.long, !hl.int) -> !hl.int
    return a++, b;
}
