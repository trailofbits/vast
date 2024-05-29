// RUN: %check-lower-value-categories %s | %file-check %s -check-prefix=VAL_CAT
// RUN: %check-core-to-llvm %s | %file-check %s -check-prefix=C_LLVM

struct X { int x; };

// VAL_CAT: {{.*}} = ll.alloca : !hl.ptr<!hl.record<"X">>

// C_LLVM: {{.*}} = llvm.alloca {{.*}} x !llvm.struct<"X", (i32)> : (i64) -> !llvm.ptr
int main()
{
    struct X x;
    return 0;
}
