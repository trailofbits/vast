// RUN: vast-cc --from-source %s | vast-opt --vast-hl-lower-types | FileCheck %s

struct X
{
    int member_x;
    int member_y;
};

// CHECK-LABEL: func @main() -> i32
int main()
{
    struct X var_a;
    var_a.member_x = 1;
    var_a.member_y = 2;

    return 0;
}
