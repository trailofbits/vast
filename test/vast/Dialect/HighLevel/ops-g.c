// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

void logic_assign_to_different_type() {
    // CHECK: hl.bin.lor [[A:%[0-9]+]], [[B:%[0-9]+]] : (!hl.long<unsigned>, !hl.int) -> !hl.int
    int a = (+1UL) || 0;
}
