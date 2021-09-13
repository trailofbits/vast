// RUN: vast-cc --from-source %s | FileCheck %s
int main()
{
    // CHECK: hl.var( c ): !hl.char
    char c;

    // CHECK: hl.var( uc ): !hl<"unsigned char">
    unsigned char uc;

    // CHECK: hl.var( sc ): !hl.char
    signed char sc;
}
