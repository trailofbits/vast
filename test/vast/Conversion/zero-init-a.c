// RUN: %vast-front -vast-emit-mlir=llvm -o - %s | %file-check %s

struct big { int i[sizeof (int) >= 4 && sizeof (void *) >= 4 ? 0x4000 : 4]; };
// CHECK: llvm.mlir.global external @gb()
// CHECK: [[V1:%[0-9]+]] = llvm.mlir.zero : !llvm.struct<"big"
// CHECK: llvm.return [[V1]]
struct big gb;

