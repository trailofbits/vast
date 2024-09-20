// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.union @u : {
// CHECK:   hl.field @u32 : !hl.int< unsigned >
// CHECK:   hl.field @u16 : !hl.array<2, !hl.short< unsigned >>
// CHECK:   hl.field @u8 : !hl.char< unsigned >
// CHECK: }
// CHECK: hl.var @u : !hl.lvalue<!hl.elaborated<!hl.record<@u>>
union u {
    unsigned int   u32;
    unsigned short u16[2];
    unsigned char  u8;
} u;
