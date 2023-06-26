// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-splice-trailing-scopes --vast-hl-to-func | FileCheck %s

// CHECK: func.func private @foo()
void foo();

// CHECK: func.func private @boo(!hl.lvalue<!hl.int>) -> !hl.int
int boo( int x );
