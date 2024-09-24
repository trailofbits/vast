// RUN: %vast-front -vast-emit-mlir-after=vast-hl-lower-enum-decls %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=ENUM

// ENUM-NOT: hl.enum
// ENUM-NOT: hl.enum.const
enum E : int {
    E_a = 0,
    E_b = 1
};

// ENUM: hl.func{{.*}}
int main() {

    // ENUM: hl.var @a : !hl.lvalue<!hl.int> = {
    // ENUM:   {{.*}} = hl.const #core.integer<0> : !hl.int
    // ENUM:   hl.value.yield {{.*}} : !hl.int
    // ENUM: }
    int a = E_a;

    // ENUM: hl.var @b : !hl.lvalue<!hl.int> = {
    // ENUM:   {{.*}} = hl.const #core.integer<0> : !hl.int
    // ENUM:   hl.value.yield {{.*}} : !hl.int
    // ENUM: }
    enum E b = E_a;

    // ENUM: hl.var @c : !hl.lvalue<!hl.int> = {
    // ENUM:   {{.*}} = hl.const #core.integer<0> : !hl.int
    // ENUM:   hl.value.yield {{.*}} : !hl.int
    // ENUM: }
    enum E c = 0;

    // ENUM: hl.var @d : !hl.lvalue<!hl.int> = {
    // ENUM:    {{.*}} = hl.ref @a : !hl.lvalue<!hl.int>
    // ENUM:    hl.value.yield {{.*}} : !hl.int
    // ENUM: }
    enum E d = a;
}
