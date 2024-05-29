// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-LABEL: hl.func @test1
int test1()
{
    // CHECK: core.scope
    {
        int a = 0;
    }

    int a = 0;
    // CHECK: return [[C1:%[0-9]+]] : !hl.int
    return a;
}

// CHECK-LABEL: hl.func @test2
void test2()
{
    // CHECK: core.scope
    // CHECK: hl.var @a : !hl.lvalue<!hl.int>
    {
        int a;
    }

    // CHECK: core.scope
    // CHECK: hl.var @a : !hl.lvalue<!hl.int>
    {
        int a;
    }

    // CHECK: core.scope
    // CHECK: hl.var @a : !hl.lvalue<!hl.int>
    {
        int a;
    }
}

// CHECK-LABEL: hl.func @test3
int test3()
{
    // CHECK: hl.var @b : !hl.lvalue<!hl.int>
    int b;

    // CHECK: core.scope
    {
        // CHECK: hl.var @a : !hl.lvalue<!hl.int>
        int a;
    }

    // CHECK-NOT: core.scope
    int a;
    // CHECK: return [[C3:%[0-9]+]] : !hl.int
    return 0;
}

// CHECK-LABEL: hl.func @test4
int test4()
{
    // CHECK-NOT: core.scope
    {
        int a = 0;
        // CHECK: hl.return
        return a;
    }
}
