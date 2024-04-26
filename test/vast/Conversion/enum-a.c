// RUN: %vast-front -vast-snapshot-at="vast-hl-lower-enums" %s
// RUN: %file-check %s -input-file=$(basename %s .c).vast-hl-lower-enums -check-prefix=ENUM

enum E : int {
    E_a = 0,
    E_b = 1
};

// ENUM: module{{.*}}
// ENUM-NEXT: hl.func{{.*}}
int main() {

    // ENUM: {{.*}} = hl.var "a" : !hl.lvalue<!hl.int> = {
    // ENUM:   {{.*}} = hl.const #core.integer<0> : !hl.int
    // ENUM:   hl.value.yield {{.*}} : !hl.int
    // ENUM: }
    int a = E_a;

    // ENUM: {{.*}} = hl.var "b" : !hl.lvalue<!hl.int> = {
    // ENUM:   {{.*}} = hl.const #core.integer<0> : !hl.int
    // ENUM:   hl.value.yield {{.*}} : !hl.int
    // ENUM: }
    enum E b = E_a;

    // ENUM: {{.*}} = hl.var "c" : !hl.lvalue<!hl.int> = {
    // ENUM:   {{.*}} = hl.const #core.integer<0> : !hl.int
    // ENUM:   hl.value.yield {{.*}} : !hl.int
    // ENUM: }
    enum E c = 0;

    // ENUM: {{.*}} = hl.var "d" : !hl.lvalue<!hl.int> = {
    // ENUM:    {{.*}} = hl.ref {{.*}} : (!hl.lvalue<!hl.int>) -> !hl.lvalue<!hl.int>
    // ENUM:    hl.value.yield {{.*}} : !hl.int
    // ENUM: }
    enum E d = a;
}
