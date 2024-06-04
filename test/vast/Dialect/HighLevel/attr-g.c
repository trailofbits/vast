// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

#pragma pack(4)

// CHECK: #hl.max_field_alignment<{{[0-9]+}}>
struct foo2 {
  short a;
  long  x;
  short y;
};

