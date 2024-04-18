// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=hl %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=HL
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=hl -vast-canonicalize %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=CAN

int arithmetic() {
    return 5 + 10;
}

// HL: hl.func @arithmetic {{.*}} () -> !hl.int
// HL:  hl.const #core.integer<5>
// HL:  hl.const #core.integer<10>
// HL:  hl.add

// CAN: hl.func @arithmetic {{.*}} () -> !hl.int
// CAN:  hl.const #core.integer<15>
