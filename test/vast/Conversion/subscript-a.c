// RUN: %check-hl-lower-types %s | %file-check %s -check-prefix=STD_TYPES
// RUN: %check-lower-value-categories %s | %file-check %s -check-prefix=VAL_CAT
// RUN: %check-core-to-llvm %s | %file-check %s -check-prefix=C_LLVM

// STD_TYPES:  [[V2:%[0-9]+]] = hl.ref @arr1 : !hl.lvalue<!hl.array<3, si32>>
// STD_TYPES:  [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] ArrayToPointerDecay : !hl.lvalue<!hl.array<3, si32>> -> !hl.ptr<si32>
// STD_TYPES:  [[V4:%[0-9]+]] = hl.const #core.integer<0> : si32
// STD_TYPES:  [[V5:%[0-9]+]] = hl.subscript [[V3]] at [[[V4]] : si32] : !hl.ptr<si32> -> !hl.lvalue<si3

// VAL_CAT: [[V0:%[0-9]+]] = ll.alloca : !hl.ptr<!hl.array<3, si32>>
// VAL_CAT: [[V4:%[0-9]+]] = hl.initlist {{.*}} : (si32, si32, si32) -> !hl.array<3, si32>
// VAL_CAT: ll.store [[V0]], [[V4]] : !hl.ptr<!hl.array<3, si32>>, !hl.array<3, si32>

// VAL_CAT: [[V5:%[0-9]+]] = hl.implicit_cast [[V0]] ArrayToPointerDecay : !hl.ptr<!hl.array<3, si32>> -> !hl.ptr<si32>
// VAL_CAT: [[V6:%[0-9]+]] = hl.const #core.integer<0> : si32
// VAL_CAT: [[V7:%[0-9]+]] = ll.subscript [[V5]] at [[[V6]] : si32] : !hl.ptr<si32> -> !hl.ptr<si32>
// VAL_CAT: [[V8:%[0-9]+]] = ll.load [[V7]] : (!hl.ptr<si32>) -> si32

// C_LLVM: [[V1:%[0-9]+]] = llvm.alloca {{.*}} x !llvm.array<3 x i32> : (i64) -> !llvm.ptr
// C_LLVM: [[V2:%[0-9]+]] = llvm.load [[V1]] : !llvm.ptr -> i32

void fn()
{
    int arr1[] = { 0, 2, 4 };

    (void)arr1[0];
}
