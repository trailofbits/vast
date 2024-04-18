// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=hl %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=HL
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=llvm %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=MLIR
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-llvm %s -o %t.ll
// RUN: %file-check --input-file=%t.ll %s -check-prefix=LLVM

extern void abort(void);
extern void __assert_fail(const char *, const char *, unsigned int, const char *) __attribute__((__nothrow__, __leaf__)) __attribute__((__noreturn__));
void reach_error() { __assert_fail("0", "diamond_2-1.c", 3, "reach_error"); }
extern unsigned int __VERIFIER_nondet_uint(void);

void __VERIFIER_assert(int cond)
{
    if (!(cond))
    {
    ERROR:
    {
        reach_error();
        abort();
    }
    }
    return;
}

int main(void)
{
    unsigned int x = 0;
    unsigned int y = __VERIFIER_nondet_uint();

    while (x < 99)
    {
        if (y % 2 == 0)
            x++;
        else
            x += 2;

        if (y % 2 == 0)
            x += 2;
        else
            x -= 2;

        if (y % 2 == 0)
            x += 2;
        else
            x += 2;

        if (y % 2 == 0)
            x += 2;
        else
            x -= 2;

        if (y % 2 == 0)
            x += 2;
        else
            x += 2;

        if (y % 2 == 0)
            x += 2;
        else
            x -= 4;

        if (y % 2 == 0)
            x += 2;
        else
            x += 4;

        if (y % 2 == 0)
            x += 2;
        else
            x += 2;

        if (y % 2 == 0)
            x += 2;
        else
            x -= 4;

        if (y % 2 == 0)
            x += 2;
        else
            x -= 4;
    }

    __VERIFIER_assert((x % 2) == (y % 2));
}

// HL: hl.func @main {{.*}} () -> !hl.int

// MLIR: llvm.func @main() -> i32

// LLVM: define i32 @main()