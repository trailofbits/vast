// RUN: %vast-cc --ccopts -xc --from-source %s | %vast-opt --vast-hl-to-lazy-regions | FileCheck %s

void logic_assign_to_different_type() {
    // CHECK: [[L:%[0-9]+]] = core.lazy.op {
    // CHECK: hl.value.yield [[A:%[0-9]+]] : [[TL:!hl\..*]]
    // CHECK: } : !hl.long< unsigned >
    // CHECK: [[R:%[0-9]+]] = core.lazy.op {
    // CHECK: hl.value.yield [[A:%[0-9]+]] : [[TR:!hl\..*]]
    // CHECK: } : !hl.int
    // CHECK: core.bin.lor [[L]], [[R]] : ([[TL]], [[TR]]) -> !hl.int
    int lor = (+1UL) || 0;

    // CHECK: [[L:%[0-9]+]] = core.lazy.op {
    // CHECK: hl.value.yield [[A:%[0-9]+]] : [[TL:!hl\..*]]
    // CHECK: } : !hl.long< unsigned >
    // CHECK: [[R:%[0-9]+]] = core.lazy.op {
    // CHECK: hl.value.yield [[A:%[0-9]+]] : [[TR:!hl\..*]]
    // CHECK: } : !hl.int
    // CHECK: core.bin.land [[L]], [[R]] : ([[TL]], [[TR]]) -> !hl.int
    int land = (+1UL) && 0;
}
