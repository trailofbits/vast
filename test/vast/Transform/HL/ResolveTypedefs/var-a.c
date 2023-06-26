// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-splice-trailing-scopes --vast-hl-dce --vast-hl-lower-types --vast-hl-resolve-typedefs | FileCheck %s

typedef int INT;

// CHECK: {{.* = hl.var "a".*!hl.lvalue<si32>.*}}
INT a = 0;

typedef INT IINT;

// CHECK: {{.* = hl.var "b".*!hl.lvalue<si32>.*}}
IINT b = 0;

typedef IINT IIINT;

// CHECK: {{.* = hl.var "c".*!hl.lvalue<si32>.*}}
IIINT c = 0;
