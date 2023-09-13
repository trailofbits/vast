// RUN: %vast-cc --ccopts -xc --from-source %s | %file-check %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -

int main()
{
    // CHECK: hl.var "c" : !hl.lvalue<!hl.char>
    char c;

    // CHECK: hl.var "uc" : !hl.lvalue<!hl.char< unsigned >>
    unsigned char uc;

    // CHECK: hl.var "sc" : !hl.lvalue<!hl.char>
    signed char sc;
}
