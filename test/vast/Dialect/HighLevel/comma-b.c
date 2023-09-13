// RUN: %vast-cc --ccopts -xc --from-source %s | %file-check %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -
void f0(int* a) {(*a)++; }
int f1(int* a) {(*a)++; return *a;}

int comma1(int a)
{
    // CHECK: hl.bin.comma [[V1:%[0-9]+]], [[V2:%[0-9]+]] : (!hl.void, !hl.int) -> !hl.int
    return f0(&a), f1(&a);
}
void comma2(int a)
{
    // CHECK: hl.bin.comma [[V1:%[0-9]+]], [[V2:%[0-9]+]] : (!hl.int, !hl.void) -> !hl.void
    return f1(&a), f0(&a);
}

void comma3(int a)
{
    // CHECK: hl.bin.comma [[V1:%[0-9]+]], [[V2:%[0-9]+]] : (!hl.void, !hl.void) -> !hl.void
    return f0(&a), f0(&a);
}
