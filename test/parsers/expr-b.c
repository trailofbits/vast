// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s -check-prefix=HL

#include <stdio.h>
#include <ctype.h>

typedef enum { PLUS = '+', MINUS = '-', MUL = '*', DIV = '/', END = '\0' } TokenType;

typedef struct {
    TokenType type;
    int value;
} Token;

const char *input;

// HL: hl.func @get_next_token
Token get_next_token() {
    while (isspace(*input)) input++; // Skip spaces

    if (isdigit(*input)) {
        int value = 0;
        while (isdigit(*input)) {
            value = value * 10 + (*input - '0');
            input++;
        }
        return (Token){ .type = END, .value = value }; // Number token
    } else if (*input == '+' || *input == '-' || *input == '*' || *input == '/') {
        TokenType type = *input;
        input++;
        return (Token){ .type = type }; // Operator token
    }

    return (Token){ .type = END }; // End of input
}

int parse_factor();  // Forward declaration

// Parsing a term (factor possibly with '*' or '/')
int parse_term() {
    int result = parse_factor();
    Token token = get_next_token();

    while (token.type == MUL || token.type == DIV) {
        if (token.type == MUL) {
            result *= parse_factor();
        } else if (token.type == DIV) {
            result /= parse_factor();
        }
        token = get_next_token();
    }

    return result;
}

// HL: hl.func @parse_factor
// Parsing a factor (number)
int parse_factor() {
    Token token = get_next_token();
    if (token.type == END) {
        return token.value;
    }
    return 0; // Fallback
}

// HL: hl.func @parse_expression
// Parsing an expression (term possibly with '+' or '-')
int parse_expression() {
    int result = parse_term();
    Token token = get_next_token();

    while (token.type == PLUS || token.type == MINUS) {
        if (token.type == PLUS) {
            result += parse_term();
        } else if (token.type == MINUS) {
            result -= parse_term();
        }
        token = get_next_token();
    }

    return result;
}

int main() {
    char buffer[100];

    printf("Enter an arithmetic expression (e.g., 3 + 5 * 2): ");
    fgets(buffer, sizeof(buffer), stdin);

    input = buffer;
    int result = parse_expression();
    printf("Result: %d\n", result);

    return 0;
}
