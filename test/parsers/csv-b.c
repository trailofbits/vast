// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s -check-prefix=HL

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Parsing part: A function to split a CSV line into tokens
// HL: hl.func @parse_csv_line
char **parse_csv_line(char *line, int *count) {
    int capacity = 10; // Initial capacity for fields
    char **fields = malloc(capacity * sizeof(char *));
    *count = 0;

    char *token = strtok(line, ",");
    while (token != NULL) {
        if (*count >= capacity) {
            capacity *= 2;
            fields = realloc(fields, capacity * sizeof(char *));
        }
        // Trim whitespace and add token to fields
        fields[*count] = strdup(token);
        (*count)++;

        token = strtok(NULL, ",");
    }
    return fields;
}

// Non-parsing part: A function to handle the parsed CSV fields
// HL: hl.func @handle_csv_fields
void handle_csv_fields(char **fields, int count) {
    printf("Parsed fields:\n");
    for (int i = 0; i < count; ++i) {
        printf("Field %d: %s\n", i + 1, fields[i]);
    }
}

// Parsing part: Read the file line by line
void parse_csv_file(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        // Remove newline character from the line
        line[strcspn(line, "\n")] = '\0';

        int count;
        // Parse the line to extract CSV fields
        char **fields = parse_csv_line(line, &count);

        // Handle the parsed fields
        handle_csv_fields(fields, count);

        // Free allocated memory for fields
        for (int i = 0; i < count; ++i) {
            free(fields[i]);
        }
        free(fields);
    }

    fclose(file);
}

// Example usage
int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return 1;
    }

    // Parsing part: Parse the CSV file
    parse_csv_file(argv[1]);

    return 0;
}
