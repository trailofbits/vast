// RUN: %vast-front -c -vast-snapshot-at="vast-hl-lower-enums" %s
// RUN: %file-check %s -input-file=$(basename %s .c).vast-hl-lower-enums -check-prefix=ENUM

enum E : char {
    E_a = 0,
    E_b = E_a + 1
};

// ENUM: hl.func @id {{.*}} ({{.*}}!hl.lvalue<!hl.char>) -> !hl.char {
enum E id(enum E e) {
    return e;
}

int main() {
    // ENUM: {{.*}} = hl.var "a" : !hl.lvalue<!hl.char> = {
    // ENUM:   [[A7:%[0-9]+]] = hl.call @id({{.*}}) : (!hl.char) -> !hl.char
    // ENUM:   hl.value.yield [[A7]] : !hl.char
    // ENUM: }
    enum E a = id(E_b);
    // ENUM: {{.*}} = hl.var "b" : !hl.lvalue<!hl.char> = {
    // ENUM:   [[B6:%[0-9]+]] = hl.implicit_cast {{.*}} IntegralCast : !hl.int -> !hl.char
    // ENUM:   [[B7:%[0-9]+]] = hl.call @id([[B6]]) : (!hl.char) -> !hl.char
    // ENUM:   hl.value.yield [[B7]] : !hl.char
    // ENUM: }
    enum E b = id(0);

    // ENUM: {{.*}} = hl.var "c" : !hl.lvalue<!hl.int> = {
    // ENUM:   [[C7:%[0-9]+]] = hl.call @id({{.*}}) : (!hl.char) -> !hl.char
    // ENUM:   [[C8:%[0-9]+]] = hl.implicit_cast [[C7]] IntegralCast : !hl.char -> !hl.int
    // ENUM:   hl.value.yield [[C8]] : !hl.int
    // ENUM: }
    int c = id(E_b);

    // ENUM: {{.*}} = hl.var "d" : !hl.lvalue<!hl.int> = {
    // ENUM:   [[D6:%[0-9]+]] = hl.implicit_cast {{.*}} IntegralCast : !hl.int -> !hl.char
    // ENUM:   [[D7:%[0-9]+]] = hl.call @id([[D6]]) : (!hl.char) -> !hl.char
    // ENUM:   [[D8:%[0-9]+]] = hl.implicit_cast [[D7]] IntegralCast : !hl.char -> !hl.int
    // ENUM:   hl.value.yield [[D8]] : !hl.int
    // ENUM: }
    int d = id(0);
}
