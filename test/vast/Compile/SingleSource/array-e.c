// RUN: %vast-front -o %t %s && (%t; test $? -eq 3) \
// RUN:                      && (%t 0 0 0; test $? -eq 12)

struct point
{
    int vs[3];
};

int main(int argc, char **argv)
{
    struct point points[10];

    for (int i = 0; i < 10; ++i)
    {
        struct point t = { i, i + 1, i - 1};
        points[i] = t;
    }

    struct point v = points[argc];
    return v.vs[0] + v.vs[1] + v.vs[2];
}
