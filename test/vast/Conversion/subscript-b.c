// RUN: %check-hl-lower-types %s | %file-check %s -check-prefix=STD_TYPES
// RUN: %check-lower-value-categories %s | %file-check %s -check-prefix=VAL_CAT
// RUN: %check-core-to-llvm %s | %file-check %s -check-prefix=C_LLVM

// STD_TYPES: {{.*}} = hl.var @arr1 : !hl.lvalue<!hl.array<3, si32>> = {


// VAL_CAT: {{.*}} = hl.var @arr1 : !hl.ptr<!hl.array<3, si32>> = {
// VAL_CAT:    hl.value.yield {{.*}} : !hl.array<3, si32>


// C_LLVM:  llvm.mlir.global internal constant @arr1() {addr_space = 0 : i32} : !llvm.array<3 x i32> {

int arr1[] = { 0, 2, 4 };

// STD_TYPES:  [[V2:%[0-9]+]] = hl.globref @arr1 : !hl.lvalue<!hl.array<3, si32>>
// STD_TYPES:  [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] ArrayToPointerDecay : !hl.lvalue<!hl.array<3, si32>> -> !hl.ptr<si32>
// STD_TYPES:  [[V4:%[0-9]+]] = hl.const #core.integer<0> : si32
// STD_TYPES:  [[V5:%[0-9]+]] = hl.subscript [[V3]] at [[[V4]] : si32] : !hl.ptr<si32> -> !hl.lvalue<si32>
// STD_TYPES:  [[V6:%[0-9]+]] = hl.implicit_cast [[V5]] LValueToRValue : !hl.lvalue<si32> -> si32


// VAL_CAT: [[V1:%[0-9]+]] = hl.globref @arr1 : !hl.ptr<!hl.array<3, si32>>
// VAL_CAT: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] ArrayToPointerDecay : !hl.ptr<!hl.array<3, si32>> -> !hl.ptr<si32>
// VAL_CAT: [[V3:%[0-9]+]] = hl.const #core.integer<0> : si32
// VAL_CAT: [[V4:%[0-9]+]] = ll.subscript [[V2]] at [[[V3]] : si32] : !hl.ptr<si32> -> !hl.ptr<si32>
// VAL_CAT: [[V5:%[0-9]+]] = ll.load [[V4]] : (!hl.ptr<si32>) -> si32


// C_LLVM: [[V0:%[0-9]+]] = llvm.mlir.addressof @arr1 : !llvm.ptr
// C_LLVM: [[V1:%[0-9]+]] = llvm.load [[V0]] : !llvm.ptr -> i32

void fn()
{
      (void)arr1[0];
}
