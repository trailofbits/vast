// RUN: %vast-cc --ccopts -xc --from-source %s | %vast-opt --vast-hl-dce --vast-hl-lower-types --vast-hl-resolve-typedefs | %file-check %s

typedef int INT;

struct X
{
    // CHECK: hl.field "a" : si32
    INT a;
};

// CHECK: [[V1:%[0-9]+]] = hl.const #hl.integer<0> : si32
// CHECK: [[V2:%[0-9]+]] = hl.initlist [[V1]] : (si32) -> !hl.elaborated<!hl.record<"X">>
struct X x = { 0 };
