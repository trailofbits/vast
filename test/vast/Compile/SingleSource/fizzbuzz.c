// RUN: %vast-front -o %t %s && %t | %file-check %s

#include <stdio.h>

int main(void) {
    for(int i = 1; i <= 100; ++i)
    {
        if (i % 3 == 0)
            printf("Fizz");
        if (i % 5 == 0)
            printf("Buzz");
        if ((i % 3 != 0) && (i % 5 != 0))
            printf("%d", i);
        printf("\n");
    }
    return 0;
}
// CHECK: 1
// CHECK: 2
// CHECK: Fizz
// CHECK: 4
// CHECK: Buzz
// CHECK: Fizz
// CHECK: 7
// CHECK: 8
// CHECK: Fizz
// CHECK: Buzz
// CHECK: 11
// CHECK: Fizz
// CHECK: 13
// CHECK: 14
// CHECK: FizzBuzz
// CHECK: 16
// CHECK: 17
// CHECK: Fizz
// CHECK: 19
// CHECK: Buzz
// CHECK: Fizz
// CHECK: 22
// CHECK: 23
// CHECK: Fizz
// CHECK: Buzz
// CHECK: 26
// CHECK: Fizz
// CHECK: 28
// CHECK: 29
// CHECK: FizzBuzz
// CHECK: 31
// CHECK: 32
// CHECK: Fizz
// CHECK: 34
// CHECK: Buzz
// CHECK: Fizz
// CHECK: 37
// CHECK: 38
// CHECK: Fizz
// CHECK: Buzz
// CHECK: 41
// CHECK: Fizz
// CHECK: 43
// CHECK: 44
// CHECK: FizzBuzz
// CHECK: 46
// CHECK: 47
// CHECK: Fizz
// CHECK: 49
// CHECK: Buzz
// CHECK: Fizz
// CHECK: 52
// CHECK: 53
// CHECK: Fizz
// CHECK: Buzz
// CHECK: 56
// CHECK: Fizz
// CHECK: 58
// CHECK: 59
// CHECK: FizzBuzz
// CHECK: 61
// CHECK: 62
// CHECK: Fizz
// CHECK: 64
// CHECK: Buzz
// CHECK: Fizz
// CHECK: 67
// CHECK: 68
// CHECK: Fizz
// CHECK: Buzz
// CHECK: 71
// CHECK: Fizz
// CHECK: 73
// CHECK: 74
// CHECK: FizzBuzz
// CHECK: 76
// CHECK: 77
// CHECK: Fizz
// CHECK: 79
// CHECK: Buzz
// CHECK: Fizz
// CHECK: 82
// CHECK: 83
// CHECK: Fizz
// CHECK: Buzz
// CHECK: 86
// CHECK: Fizz
// CHECK: 88
// CHECK: 89
// CHECK: FizzBuzz
// CHECK: 91
// CHECK: 92
// CHECK: Fizz
// CHECK: 94
// CHECK: Buzz
// CHECK: Fizz
// CHECK: 97
// CHECK: 98
// CHECK: Fizz
// CHECK: Buzz
