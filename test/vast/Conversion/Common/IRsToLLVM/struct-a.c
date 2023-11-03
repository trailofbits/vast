// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-lower-types --vast-hl-to-ll-cf --vast-hl-to-ll-vars --vast-irs-to-llvm | %file-check %s

struct X { int x; };

int main()
{
    // CHECK: {{.*}} = llvm.alloca {{.*}} x !llvm.struct<"X", (i32)> : (i64) -> !llvm.ptr<struct<"X", (i32)>>
    struct X x;
    return 0;
}
