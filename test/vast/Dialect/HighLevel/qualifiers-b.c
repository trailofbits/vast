// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

int main()
{
    // CHECK: hl.var "c" : !hl.char
    char c;

    // CHECK: hl.var "uc" : !hl.char<unsigned>
    unsigned char uc;

    // CHECK: hl.var "sc" : !hl.char
    signed char sc;
}
