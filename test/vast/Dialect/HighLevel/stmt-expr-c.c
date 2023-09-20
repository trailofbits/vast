// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s

int main() {
    // CHECK: hl.stmt.expr : !hl.void
    // CHECK: [[FIVE:%[0-9]+]] = hl.const
    // CHECK: hl.value.yield [[FIVE]]
    ({;;;;});
}
