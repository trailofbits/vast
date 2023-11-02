// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-lower-types --vast-hl-structs-to-llvm | %file-check %s

struct X {};

int main()
{
    // CHECK: {{.*}} = hl.var "x" : !hl.lvalue<!hl.elaborated<!llvm.struct<"X", ()>>>
    struct X x;
}
