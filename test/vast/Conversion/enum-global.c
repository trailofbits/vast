// RUN: %vast-front -vast-emit-mlir-after=vast-hl-lower-enum-decls %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=ENUM

// RUN: %vast-front -vast-emit-mlir-after=vast-irs-to-llvm %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=LLVM

// REQUIRES: erase-enum-type-from-data-layout

// ENUM-NOT: hl.enum
// ENUM-NOT: hl.enum.const
enum E : char {
    E_a = 0
};

// ENUM: hl.var @a : !hl.lvalue<!hl.char> = {
// ENUM:    [[A2:%[0-9]+]] = hl.const #core.integer<0> : !hl.int
// ENUM:    [[A3:%[0-9]+]] = hl.implicit_cast [[A2]] IntegralCast : !hl.int -> !hl.char
// ENUM:    hl.value.yield [[A3]] : !hl.char
// ENUM:  }

// LLVM:  llvm.mlir.global internal constant @a() {{.*}} : i8 {
// LLVM:    [[A1:%[0-9]+]] = llvm.trunc {{.*}} : i32 to i8
// LLVM:    llvm.return [[A1]] : i8
// LLVM:  }
enum E a = 0;

typedef enum E(*fn_ptr)(enum E);

// ENUM: hl.var @p : !hl.lvalue<!hl.ptr<!hl.paren<!core.fn<(!hl.lvalue<!hl.char>) -> (!hl.char)>>>> = {

// LLVM:  llvm.mlir.global internal constant @p() {{.*}} : !llvm.ptr {
// LLVM:    [[P1:%[0-9]+]] = llvm.mlir.zero : !llvm.ptr
// LLVM:    llvm.return [[P1]] : !llvm.ptr
// LLVM:  }
fn_ptr p = 0;
