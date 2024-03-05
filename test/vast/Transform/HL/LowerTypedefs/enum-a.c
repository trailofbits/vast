// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-dce --vast-hl-lower-types --vast-hl-lower-elaborated-types --vast-hl-lower-typedefs | %file-check %s


enum E { fst = 0 };

// CHECK:   hl.func @id (%arg0: !hl.lvalue<!hl.record<"E">>) -> !hl.record<"E"> {
enum E id(enum E a)
{
    return a;
}
