// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-to-func | FileCheck %s

// CHECK: func.func @boo(%arg0: !hl.lvalue<!hl.int>) -> !hl.int {
int boo( int x )
{
    return x;
}
