// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -
struct bar;

struct foo {
  unsigned int count;
  char other;
  // CHECK: hl.field @array : !hl.count_attributed<CountedBy, !hl.array<{{.*}}>
  struct bar *array[] __attribute__((counted_by(count)));
};

