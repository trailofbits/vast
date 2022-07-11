// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.union "u" : {
// CHECK:   hl.field "u32" : !hl.int<unsigned>
// CHECK:   hl.field "u16" : !hl.array<2, !hl.short<unsigned>>
// CHECK:   hl.field "u8" : !hl.char<unsigned>
// CHECK: }
// CHECK: hl.var "u" : !hl.lvalue<!hl.named_type<<"u">>>
union u {
    unsigned int   u32;
    unsigned short u16[2];
    unsigned char  u8;
} u;
