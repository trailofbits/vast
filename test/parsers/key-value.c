// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s -check-prefix=HL
// RUN: %vast-front -vast-show-locs -vast-loc-attrs -vast-emit-mlir=hl %s -o - | %vast-opt -vast-hl-to-lazy-regions -o %t.mlir
// RUN: %vast-detect-parsers -vast-hl-to-parser -vast-parser-reconcile-casts -reconcile-unrealized-casts %t.mlir -o - | %file-check %s -check-prefix=PARSER

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

// A simple structure to store key-value pairs
typedef struct {
    char *key;
    char *value;
} KeyValuePair;

// Parsing part: A function to trim whitespace from a string
char *trim_whitespace(char *str) {
    char *end;

    // Trim leading space
    while (isspace((unsigned char)*str)) str++;

    if (*str == 0) // All spaces
        return str;

    // Trim trailing space
    end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;

    // Write new null terminator
    *(end + 1) = '\0';

    return str;
}

// Parsing part: A function to parse a key-value pair
// HL: hl.func @parse_key_value
// PARSER: hl.func @parse_key_value
KeyValuePair parse_key_value(char *line) {
    KeyValuePair kvp;
    char *delimiter = strchr(line, '=');

    if (delimiter != NULL) {
        // Split the string at '='
        *delimiter = '\0';
        kvp.key = trim_whitespace(line);
        kvp.value = trim_whitespace(delimiter + 1);
    } else {
        // Invalid format, set key and value to NULL
        kvp.key = NULL;
        kvp.value = NULL;
    }

    return kvp;
}

// Non-parsing part: Handling the parsed data
// HL: hl.func @handle_key_value
// PARSER: hl.func @handle_key_value
void handle_key_value(KeyValuePair kvp) {
    if (kvp.key && kvp.value) {
        printf("Key: %s, Value: %s\n", kvp.key, kvp.value);
    } else {
        printf("Invalid key-value pair.\n");
    }
}

// Parsing part: Read the file line by line
void parse_file(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        // Remove newline character from the line
        line[strcspn(line, "\n")] = '\0';

        // Parse the line to extract key-value
        KeyValuePair kvp = parse_key_value(line);

        // Handle the parsed key-value pair
        handle_key_value(kvp);
    }

    fclose(file);
}

// Example usage
int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return 1;
    }

    // Parsing part: Parse the input file
    parse_file(argv[1]);

    return 0;
}
