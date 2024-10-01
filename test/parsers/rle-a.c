// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s -check-prefix=HL

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// Function to decompress RLE compressed data
// HL: hl.func @decompress_rle
void decompress_rle(uint8_t *compressed, size_t length, uint8_t **decompressed, size_t *decompressed_length) {
    size_t idx = 0;
    size_t out_idx = 0;

    // Estimate decompressed length
    *decompressed_length = length * 2; // Upper bound
    *decompressed = (uint8_t *)malloc(*decompressed_length);

    while (idx < length) {
        uint8_t count = compressed[idx++];
        uint8_t value = compressed[idx++];

        // Write 'count' instances of 'value' to the output
        for (int i = 0; i < count; ++i) {
            (*decompressed)[out_idx++] = value;
        }
    }
    *decompressed_length = out_idx; // Update with the actual length
}

// Function to parse a binary file containing compressed data
// HL: hl.func @parse_binary_file
void parse_binary_file(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        return;
    }

    // Read header (4 bytes) - contains length of compressed data
    uint32_t compressed_length;
    fread(&compressed_length, sizeof(uint32_t), 1, file);

    // Allocate memory for compressed data
    uint8_t *compressed_data = (uint8_t *)malloc(compressed_length);
    fread(compressed_data, sizeof(uint8_t), compressed_length, file);

    // Close the file after reading
    fclose(file);

    printf("Read compressed data of length: %u bytes\n", compressed_length);

    // Decompression (Non-Parsing Part)
    uint8_t *decompressed_data;
    size_t decompressed_length;
    decompress_rle(compressed_data, compressed_length, &decompressed_data, &decompressed_length);

    printf("Decompressed data length: %zu bytes\n", decompressed_length);

    // Print decompressed data
    printf("Decompressed Data: ");
    for (size_t i = 0; i < decompressed_length; ++i) {
        printf("%c", decompressed_data[i]);
    }
    printf("\n");

    // Free memory
    free(compressed_data);
    free(decompressed_data);
}

int main() {
    const char *filename = "compressed_data.bin";
    parse_binary_file(filename);
    return 0;
}
