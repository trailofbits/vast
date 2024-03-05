// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt-irs-to-llvm | %file-check %s

struct Y;

struct X { const int x; struct Y *y; };

struct Y { const float x; };

int main()
{
    // CHECK: {{.*}}  = llvm.alloca {{.*}} x !llvm.struct<"X", (i32, ptr<struct<"Y", (f32)>>)> : (i64) -> !llvm.ptr<struct<"X", (i32, ptr<struct<"Y", (f32)>>)>>
    struct X x = { 2, 0 };

    // CHECK: {{.*}} = llvm.alloca {{.*}} x !llvm.struct<"Y", (f32)> : (i64) -> !llvm.ptr<struct<"Y", (f32)>>
    struct Y y = { 2.0f };

    return 0;
}
