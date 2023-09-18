// RUN: %vast-cc --ccopts -xc --from-source %s | %vast-opt --vast-hl-lower-types | %file-check %s

// CHECK: hl.func @test1 () -> si32
int test1()
{
    // CHECK: core.scope
    {
        int a = 0;
    }

    int a = 0;
    // CHECK: return [[C1:%[0-9]+]] : si32
    return a;
}

// CHECK: hl.func @test2 ()
void test2()
{
    // CHECK: core.scope
    // CHECK: hl.var "a" : !hl.lvalue<si32>
    {
        int a;
    }

    // CHECK: core.scope
    // CHECK: hl.var "a" : !hl.lvalue<si32>
    {
        int a;
    }

    // CHECK: core.scope
    // CHECK: hl.var "a" : !hl.lvalue<si32>
    {
        int a;
    }
}

// CHECK: hl.func @test3 () -> si32
int test3()
{
    // CHECK: hl.var "b" : !hl.lvalue<si32>
    int b;

    // CHECK: core.scope
    {
        // CHECK: hl.var "a" : !hl.lvalue<si32>
        int a;
    }

    // CHECK-NOT: core.scope
    int a;
    // CHECK: return [[C3:%[0-9]+]] : si32
    return 0;
}

// CHECK: hl.func @test4 () -> si32
int test4()
{
    // CHECK-NOT: core.scope
    {
        int a = 0;
        // CHECK: hl.return
        return a;
    }
}
