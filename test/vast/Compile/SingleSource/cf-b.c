// RUN: %vast-front -o %t %s && (%t hel con prcsu bre ta | %file-check %s -check-prefix=FST) \
// RUN:                      && (%t test ab tail | %file-check %s -check-prefix=SND)

#include <stdio.h>

// FST: -hel
// FST: -p

// SND: -test
// SND: -a
// SND: -tail

int main(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == 'c')
            continue;
        if (argv[i][0] == 'b')
            break;
        int j = 0;
        putchar('-');
        while (argv[i][j] != '\0') {
            if (argv[i][j] == 'r') {
                putchar('\n');
                return 0;
            }
            if (argv[i][j] == 'b') {
                break;
            }
            putchar(argv[i][j]);
            ++j;
        }
        putchar('\n');
    }
}
