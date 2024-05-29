// RUN: %vast-front -o %t %s && %t | %file-check %s

#include <stdio.h>

float inc_float(float f) { return f + 1.0f; }
double inc_double(double d) { return d + 1.0; }

void print_f(float f) { printf("f: %f\n", f); }
void print_d(double d) { printf("d: %f\n", d); }

// CHECK: f: 1.500000
// CHECK: f: 0.500000
// CHECK: d: 1.500000
// CHECK: d: 12.000000
// CHECK: d: 0.000000
// CHECK: f: 1.010000
// CHECK: d: 0.000000
int main()
{
    print_f(inc_float(0.5f));
    print_f(inc_float(-0.5f));
    print_d(inc_float(0.5f));

    print_d(inc_double(11.0));
    print_d(inc_double(-1.0));
    print_f(inc_double(0.01));

    print_d(0.00000001);
}
