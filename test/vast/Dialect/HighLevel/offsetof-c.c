// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

struct foo {
    struct {
        union {
            struct {
                int bar;
            };
        };
    };
};

int main(void) {
    // CHECK: hl.offsetof.expr type : !hl.elaborated<!hl.record<@foo>>{{.*}}"anonymous[{{[0-9]+}}]"{{.*}}"anonymous[{{[0-9]+}}]"{{.*}}"anonymous[{{[0-9]+}}]"{{.*}}"bar"
    (void) __builtin_offsetof(struct foo, bar);
    return 0;
}
