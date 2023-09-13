// RUN: %vast-cc --ccopts -xc --from-source %s | %file-check %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -

// CHECK-LABEL: hl.func @test1 () -> !hl.int
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

// CHECK-LABEL: hl.func @test2 ()
void test2()
{
    // CHECK: core.scope
    // CHECK: hl.var "a" : !hl.lvalue<!hl.int>
    {
        int a;
    }

    // CHECK: core.scope
    // CHECK: hl.var "a" : !hl.lvalue<!hl.int>
    {
        int a;
    }

    // CHECK: core.scope
    // CHECK: hl.var "a" : !hl.lvalue<!hl.int>
    {
        int a;
    }
}

// CHECK-LABEL: hl.func @test3 () -> !hl.int
int test3()
{
    // CHECK: hl.var "b" : !hl.lvalue<!hl.int>
    int b;

    // CHECK: core.scope
    {
        // CHECK: hl.var "a" : !hl.lvalue<!hl.int>
        int a;
    }

    // CHECK-NOT: core.scope
    int a;
    // CHECK: return [[C3:%[0-9]+]] : !hl.int
    return 0;
}

// CHECK-LABEL: hl.func @test4 () -> !hl.int
int test4()
{
    // CHECK-NOT: core.scope
    {
        int a = 0;
        // CHECK: hl.return
        return a;
    }
}
