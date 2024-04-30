// RUN: %vast-front -c -vast-pipeline=with-abi -o %t.vast.o %s && %cc -c -xc %s.driver -o %t.clang.o  && %cc %t.vast.o %t.clang.o -o %t && (%t; test $? -eq 0)

float float_sum(float a, float b)
{
    return a + b;
}

double double_sum(double a, double b)
{
    return a + b;
}
