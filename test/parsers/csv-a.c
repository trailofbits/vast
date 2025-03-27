// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s -check-prefix=HL
// RUN: %vast-front -vast-show-locs -vast-loc-attrs -vast-emit-mlir=hl %s -o - | %vast-opt -vast-hl-to-lazy-regions -o %t.mlir
// RUN: %vast-detect-parsers -vast-hl-to-parser -vast-parser-reconcile-casts -reconcile-unrealized-casts %t.mlir -o - | %file-check %s -check-prefix=PARSER

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LEN 1024
#define MAX_FIELD_LEN 256

int parse_csv_line(char *line, char *fields[]);

// Non-parsing part: file handling and utility functions
// HL: hl.func @read_csv_file
// PARSER: hl.func @read_csv_file
void read_csv_file(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Could not open file");
        exit(EXIT_FAILURE);
    }

    char line[MAX_LINE_LEN];

    // Parsing part: parsing lines and extracting fields
    while (fgets(line, sizeof(line), file)) {
        char *fields[MAX_FIELD_LEN];
        int field_count = parse_csv_line(line, fields);

        // Example use of parsed fields
        printf("Parsed %d fields:\n", field_count);
        for (int i = 0; i < field_count; ++i) {
            printf("Field %d: %s\n", i, fields[i]);
        }
        printf("\n");

        // Free the allocated memory for fields
        for (int i = 0; i < field_count; ++i) {
            free(fields[i]);
        }
    }

    fclose(file);
}

// Parsing part: core CSV parsing logic
int parse_csv_line(char *line, char *fields[]) {
    int count = 0;
    char *start = line;
    int in_quotes = 0;

    while (*start) {
        // Skip whitespace
        while (*start == ' ' || *start == '\t') start++;

        // Handle quotes
        if (*start == '\"') {
            in_quotes = 1;
            start++;
        }

        // Capture the beginning of the field
        char *field_start = start;

        // Extract the field
        while (*start && (in_quotes || (*start != ',' && *start != '\n'))) {
            if (in_quotes && *start == '\"') {
                if (*(start + 1) == '\"') {
                    start += 2; // Skip escaped quote
                } else {
                    in_quotes = 0; // End of quoted field
                    start++;
                    break;
                }
            } else {
                start++;
            }
        }

        // Allocate memory for the field and store it
        int length = start - field_start;
        fields[count] = (char *)malloc(length + 1);
        strncpy(fields[count], field_start, length);
        fields[count][length] = '\0';
        count++;

        // Skip comma or newline
        if (*start == ',') start++;
    }

    return count;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <csv-file>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    read_csv_file(argv[1]);
    return 0;
}
