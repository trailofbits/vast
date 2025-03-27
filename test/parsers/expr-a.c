// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s -check-prefix=HL
// RUN: %vast-front -vast-show-locs -vast-loc-attrs -vast-emit-mlir=hl %s -o - | %vast-opt -vast-hl-to-lazy-regions -o %t.mlir
// RUN: %vast-detect-parsers -vast-hl-to-parser -vast-parser-reconcile-casts -reconcile-unrealized-casts %t.mlir -o - | %file-check %s -check-prefix=PARSER

#include <stdio.h>
#include <ctype.h>

// HL: hl.func @parse_number
// PARSER: hl.func @parse_number
int parse_number(const char **input) {
    int value = 0;
    while (isdigit(**input)) {
        value = value * 10 + (**input - '0');
        (*input)++;
    }
    return value;
}

// HL: hl.func @add_two_numbers
// PARSER: hl.func @add_two_numbers
int add_two_numbers(const char *input) {
    int num1 = parse_number(&input);
    while (isspace(*input)) input++; // Skip spaces
    input++; // Skip '+'
    int num2 = parse_number(&input);
    return num1 + num2;
}

int main() {
    char input[100];

    printf("Enter an expression (e.g., 3 + 5): ");
    fgets(input, sizeof(input), stdin);

    int result = add_two_numbers(input);
    printf("Result: %d\n", result);

    return 0;
}
