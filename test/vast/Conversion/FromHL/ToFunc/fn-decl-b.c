// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-lower-types --vast-hl-to-func | FileCheck %s

// CHECK: func.func private @foo() -> none
void foo();

// CHECK: func.func private @boo(!hl.lvalue<si32>) -> si32
int boo( int x );
