// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt-irs-to-llvm | %file-check %s

struct X { int a; struct X *x; };

int main()
{
    // CHECK: {{.*}} = llvm.alloca {{.*}} x !llvm.struct<"X", (i32, ptr<struct<"X">>)> : (i64) -> !llvm.ptr<struct<"X", (i32, ptr<struct<"X">>)>>
    struct X x = { 2, 0 };
    return 0;
}
