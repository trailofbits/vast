// RUN: vast-cc --from-source %s | FileCheck %s

int main()
{
    int a;
}

// CHECK-LABEL: func @main() -> !hl.int
// CHECK: hl.var( a ): !hl.int
