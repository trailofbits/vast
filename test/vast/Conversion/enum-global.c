// RUN: %vast-front -c -vast-snapshot-at="vast-hl-lower-enums;vast-irs-to-llvm" %s
// RUN: %file-check %s -input-file=$(basename %s .c).vast-hl-lower-enums -check-prefix=ENUM
// RUN: %file-check %s -input-file=$(basename %s .c).vast-irs-to-llvm -check-prefix=LLVM

enum E : char {
    E_a = 0
};

// ENUM:  {{.*}} = hl.var "a" : !hl.lvalue<!hl.char> = {
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

// ENUM:  {{.*}} = hl.var "p" : !hl.lvalue<!hl.ptr<!hl.paren<!core.fn<(!hl.lvalue<!hl.char>) -> (!hl.char)>>>> = {

// LLVM:  llvm.mlir.global internal constant @p() {{.*}} : !llvm.ptr {
// LLVM:    [[P1:%[0-9]+]] = llvm.mlir.zero : !llvm.ptr
// LLVM:    llvm.return [[P1]] : !llvm.ptr
// LLVM:  }
fn_ptr p = 0;
