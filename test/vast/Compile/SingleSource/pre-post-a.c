// RUN: %vast-front -o %t %s && (%t 1 1 1 1; test $? -eq 0)

int pre_inc(int *v) { return ++(*v); }
int post_inc(int *v) { return (*v)++; }

#define CHECK(a, b) \
    if (a != b) { return 4; }

int main(int argc, char **argv) {
    int v = argc;

    CHECK(post_inc(&v), 5);
    CHECK(v, 6);
    CHECK(pre_inc(&v), 7);
    CHECK(v, 7);

    return 0;
}
