// RUN: ignore-test
// vast-cc --from-source %s | FileCheck %s
// vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.typedef "operation" : !hl.ptr<(!hl.int, !hl.int) -> !hl.int>
typedef int ( *operation ) ( int, int );

int apply( operation op, int a, int b )
{
    // CHECK: [[OP:%[0-9]+]] = hl.decl.ref "op" : !hl.ptr<(!hl.int, !hl.int) -> !hl.int>
    // CHECK: [[FN:%[0-9]+]] = hl.implicit_cast [[OP]] LValueToRValue
    // CHECK: [[R:%[0-9]+]] = hl.indirect_call [[FN]]([[A:%[0-9]+]], [[B:%[0-9]+]])
    return op( a, b );
}

int add( int a, int b ) { return a + b; }
int mul( int a, int b ) { return a * b; }

int main()
{
    // CHECK: [[F1:%[0-9]+]] = hl.decl.ref @add
    // CHECK: [[P1:%[0-9]+]] = hl.addressof [[F1]]
    // CHECK: hl.call @apply([[P1]], [[A1:%[0-9]+]], [[A2:%[0-9]+]])
    apply( &add, 1, 2 );

    // CHECK: [[F2:%[0-9]+]] = hl.decl.ref @mul
    // CHECK: [[P2:%[0-9]+]] = hl.addressof [[F2]]
    // CHECK: hl.call @apply([[P2]], [[A1:%[0-9]+]], [[A2:%[0-9]+]])
    apply( &mul, 3, 4 );
}
