#include <fcntl.h>
#include <unistd.h>

void test(int name) {
    close(name);
}
