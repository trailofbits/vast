// RUN: %vast-front -o %t %s && (%t 1; test $? -eq 2) && \
// RUN:                         (%t 1 1 1; test $? -eq 4)

void set(int *ptr, int v) { *ptr = v; }

int main(int argc, char **argv) {
    int v = 0;
    set(&v, argc);
    return argc;
}
