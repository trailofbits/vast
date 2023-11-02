// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-lower-types --vast-hl-structs-to-llvm | %file-check %s

struct Y;

struct X { const int x; struct Y *y; };

struct Y { const float x; };

int main()
{
    // CHECK: {{.*}} = hl.var "x" : !hl.lvalue<!hl.elaborated<!llvm.struct<"X", (si32, !hl.ptr<!hl.elaborated<!llvm.struct<"Y", (f32)>>>)>>> = {
    struct X x = { 2, 0 };

    // CHECK: {{.*}} = hl.var "y" : !hl.lvalue<!hl.elaborated<!llvm.struct<"Y", (f32)>>> = {
    struct Y y = { 2.0f };
}
