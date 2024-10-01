// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s -check-prefix=HL

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

// Structure to store a key-value pair
typedef struct {
    char *key;
    char *value;
} KeyValuePair;

// Structure to store an entire JSON-like object
typedef struct {
    KeyValuePair *pairs;
    size_t pair_count;
} JsonObject;

// Parsing part: Function to trim whitespace from a string
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

// Parsing part: Parse a key-value pair from a line like '"key": "value"'
// HL: hl.func @parse_key_value_pair
KeyValuePair parse_key_value_pair(char *line) {
    KeyValuePair kvp = {NULL, NULL};

    // Find the colon separator
    char *delimiter = strchr(line, ':');
    if (delimiter == NULL) {
        return kvp; // Return empty pair if format is invalid
    }

    // Split the line into key and value
    *delimiter = '\0';
    char *key = trim_whitespace(line);
    char *value = trim_whitespace(delimiter + 1);

    // Remove quotes from key and value
    if (key[0] == '\"') key++;
    if (key[strlen(key) - 1] == '\"') key[strlen(key) - 1] = '\0';
    if (value[0] == '\"') value++;
    if (value[strlen(value) - 1] == '\"') value[strlen(value) - 1] = '\0';

    // Store trimmed and processed key-value pair
    kvp.key = strdup(key);
    kvp.value = strdup(value);

    return kvp;
}

// Parsing part: Parse a JSON-like object from multiple lines
// HL: hl.func @parse_json_object
JsonObject parse_json_object(FILE *file) {
    JsonObject obj;
    obj.pairs = NULL;
    obj.pair_count = 0;

    char line[256];
    size_t capacity = 10;
    obj.pairs = malloc(capacity * sizeof(KeyValuePair));

    // Read the file line by line until the closing '}'
    while (fgets(line, sizeof(line), file)) {
        // Trim whitespace and skip empty lines or braces
        char *trimmed = trim_whitespace(line);
        if (strlen(trimmed) == 0 || trimmed[0] == '{' || trimmed[0] == '}') {
            continue;
        }

        // Parse a key-value pair from the line
        KeyValuePair kvp = parse_key_value_pair(trimmed);

        // Resize array if necessary
        if (obj.pair_count >= capacity) {
            capacity *= 2;
            obj.pairs = realloc(obj.pairs, capacity * sizeof(KeyValuePair));
        }

        // Add key-value pair to object
        obj.pairs[obj.pair_count++] = kvp;
    }

    return obj;
}

// Non-parsing part: Function to handle a parsed JsonObject
// HL: hl.func @handle_json_object
void handle_json_object(JsonObject obj) {
    printf("Parsed JSON Object:\n");
    for (size_t i = 0; i < obj.pair_count; ++i) {
        printf("  %s: %s\n", obj.pairs[i].key, obj.pairs[i].value);
    }
    printf("\n");
}

// Non-parsing part: Free memory allocated for a JsonObject
// HL: hl.func @free_json_object
void free_json_object(JsonObject obj) {
    for (size_t i = 0; i < obj.pair_count; ++i) {
        free(obj.pairs[i].key);
        free(obj.pairs[i].value);
    }
    free(obj.pairs);
}

// Main function: Parse a JSON-like file
int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return 1;
    }

    // Open the file
    FILE *file = fopen(argv[1], "r");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    // Read and parse each JSON-like object in the file
    while (!feof(file)) {
        JsonObject obj = parse_json_object(file);

        // Handle the parsed JSON-like object
        if (obj.pair_count > 0) {
            handle_json_object(obj);
            free_json_object(obj);
        }
    }

    fclose(file);
    return 0;
}
