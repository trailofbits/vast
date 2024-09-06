// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-ude | %file-check %s

// CHECK: hl.struct @used
struct used
{
    int l;
    int h;
};

// CHECK: hl.union @data
union data
{
    unsigned long long b;
    struct used s;
};

int main()
{
    union data d;
    d.b = 0;
}
