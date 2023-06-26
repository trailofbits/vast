// RUN: vast-cc --from-source %s | vast-opt --vast-hl-splice-trailing-scopes --vast-hl-lower-types | FileCheck %s

// CHECK-LABEL: hl.func external @main () -> si32
int main()
{
    // CHECK: [[V2:%[0-9]+]] = hl.const #hl.strlit<"hello"> : !hl.lvalue<memref<6xsi8>>
    const char *hello = "hello";
}
