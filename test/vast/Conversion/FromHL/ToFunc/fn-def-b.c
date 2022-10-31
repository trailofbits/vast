// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-lower-types --vast-hl-to-func | FileCheck %s

// CHECK: func.func private @boo(%arg0: !hl.lvalue<si32>) -> si32 {
int boo( int x )
{
    return x;
}
