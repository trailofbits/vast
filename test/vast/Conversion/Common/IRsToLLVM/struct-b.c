// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-lower-types --vast-hl-to-ll-cf --vast-hl-to-ll-vars --vast-irs-to-llvm | %file-check %s

struct Y;
struct X { int x; struct Y *y; };

int main()
{
    // CHECK: {{.*}} = llvm.alloca {{.*}} x !llvm.struct<"X", (i32, ptr<struct<"Y", opaque>>)> : (i64) -> !llvm.ptr<struct<"X", (i32, ptr<struct<"Y", opaque>>)>>
    struct X x = { 2, 0 };
    return 0;
}
