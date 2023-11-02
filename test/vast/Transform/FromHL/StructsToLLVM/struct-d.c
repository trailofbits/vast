// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-lower-types --vast-hl-structs-to-llvm | %file-check %s

struct X { int a; struct X *x; };

int main()
{
    // CHECK: {{.*}} = hl.var "x" : !hl.lvalue<!hl.elaborated<!llvm.struct<"X", (si32, !hl.ptr<!hl.elaborated<!llvm.struct<"X">>>)>>> = {
    struct X x = { 2, 0 };
    x.a = 5;
}
