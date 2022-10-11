// RUN: vast-cc --from-source %s | vast-opt --vast-hl-lower-types | FileCheck %s

// CHECK-LABEL: func @test1() -> si32
int test1()
{
    // CHECK: hl.scope
    {
        int a = 0;
    }

    int a = 0;
    // CHECK: return [[C1:%[0-9]+]] : si32
    return a;
}

// CHECK-LABEL: func @test2() -> none
void test2()
{
    // CHECK: hl.scope
    // CHECK: hl.var "a" : !hl.lvalue<si32>
    {
        int a;
    }

    // CHECK: hl.scope
    // CHECK: hl.var "a" : !hl.lvalue<si32>
    {
        int a;
    }

    // CHECK-NOT: hl.scope
    // CHECK: hl.var "a" : !hl.lvalue<si32>
    {
        int a;
    }
}

// CHECK-LABEL: func @test3() -> si32
int test3()
{
    // CHECK: hl.var "b" : !hl.lvalue<si32>
    int b;

    // CHECK: hl.scope
    {
        // CHECK: hl.var "a" : !hl.lvalue<si32>
        int a;
    }

    // CHECK-NOT: hl.scope
    int a;
    // CHECK: return [[C3:%[0-9]+]] : si32
    return 0;
}

// CHECK-LABEL: func @test4() -> si32
int test4()
{
    // CHECK-NOT: hl.scope
    {
        int a = 0;
        // CHECK: hl.return
        return a;
    }
}
