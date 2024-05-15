// RUN: %vast-front -vast-emit-mlir-after=vast-hl-to-lazy-regions %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=LAZY

// RUN: %vast-front -vast-emit-mlir-after=vast-core-to-llvm %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=LLVM

#include <stdlib.h>
#include <stdio.h>

int main(int argc, char** argv)
{
    // LAZY: [[V4:%[0-9]+]] = core.lazy.op {
    // LAZY:   [[V13:%[0-9]+]] = hl.call @atoi({{.*}}) : (!hl.ptr<si8>) -> si32
    // LAZY:   hl.value.yield [[V13]] : si32
    // LAZY: } : si32
    // LAZY: [[V5:%[0-9]+]] = core.lazy.op {
    // LAZY:   [[V9:%[0-9]+]] = hl.const #core.integer<769> : si32
    // LAZY:   hl.value.yield [[V9]] : si32
    // LAZY: } : si32
    // LAZY: {{.*}} = core.select {{.*}}, [[V4]], [[V5]] : (si32, si32, si32) -> si32

    // LLVM:    [[V11:%[0-9]+]] = llvm.icmp "ne" {{.*}} : i32
    // LLVM:    llvm.cond_br [[V11]], ^bb1, ^bb2
    // LLVM: ^bb1:  // pred: ^bb0
    // LLVM:    [[V16:%[0-9]+]] = llvm.call @atoi({{.*}}) : (!llvm.ptr) -> i32
    // LLVM:    llvm.br ^bb3([[V16]] : i32)
    // LLVM: ^bb2:  // pred: ^bb0
    // LLVM:    [[V17:%[0-9]+]] = llvm.mlir.constant(769 : i32) : i32
    // LLVM:    llvm.br ^bb3([[V17]] : i32)
    // LLVM: ^bb3([[V20:%[0-9]+]]: i32):  // 2 preds: ^bb1, ^bb2
    // LLVM:    llvm.store [[V20]], {{.*}} : i32, !llvm.ptr
    int i = (argc >= 3)? atoi(argv[2]) : 769;
}
