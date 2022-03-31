// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK-LABEL: func @test1() -> !hl.int
int test1()
{
    // CHECK: hl.scope
    {
        int a = 0;
    }

    int a = 0;
    // CHECK: return [[C1:%[0-9]+]] : !hl.int
    return a;
}

// CHECK-LABEL: func @test2() -> !hl.void
void test2()
{
    // CHECK: hl.scope
    // CHECK: hl.var @a : !hl.int
    {
        int a;
    }

    // CHECK: hl.scope
    // CHECK: hl.var @a : !hl.int
    {
        int a;
    }

    // CHECK-NOT: hl.scope
    // CHECK: hl.var @a : !hl.int
    {
        int a;
    }
}

// CHECK-LABEL: func @test3() -> !hl.int
int test3()
{
    // CHECK: hl.var @b : !hl.int
    int b;

    // CHECK: hl.scope
    {
        // CHECK: hl.var @a : !hl.int
        int a;
    }

    // CHECK-NOT: hl.scope
    int a;
    // CHECK: return [[C3:%[0-9]+]] : !hl.int
    return 0;
}

// CHECK-LABEL: func @test4() -> !hl.int
int test4()
{
    // CHECK-NOT: hl.scope
    {
        int a = 0;
        // CHECK: hl.return
        return a;
    }
}
