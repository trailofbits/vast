// RUN: %vast-front -c -vast-pipeline=with-abi -o %t.vast.o %s && %cc -c -xc %s.driver -o %t.clang.o  && %cc %t.vast.o %t.clang.o -o %t && %t | %file-check %s.driver

struct d_data
{
    double x;
    double y;
};

double sum(struct d_data d)
{
    return d.x + d.y;
}
