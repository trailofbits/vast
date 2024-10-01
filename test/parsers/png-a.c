// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s -check-prefix=HL

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// Function to read a 4-byte big-endian integer from a buffer
// HL: hl.func @read_uint32_be
uint32_t read_uint32_be(uint8_t *buffer) {
    return (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
}

// Function to parse the IHDR chunk of a PNG file
// HL: hl.func @parse_png_ihdr_chunk
void parse_png_ihdr_chunk(FILE *file) {
    uint8_t length_bytes[4];
    fread(length_bytes, sizeof(uint8_t), 4, file);
    uint32_t length = read_uint32_be(length_bytes);

    char chunk_type[5] = {0};
    fread(chunk_type, sizeof(char), 4, file);

    if (strcmp(chunk_type, "IHDR") != 0) {
        printf("First chunk is not IHDR. Invalid PNG file.\n");
        return;
    }

    // Read IHDR data
    uint8_t ihdr_data[13];
    fread(ihdr_data, sizeof(uint8_t), 13, file);

    // Read the CRC (4 bytes, we will skip verifying it for simplicity)
    uint8_t crc[4];
    fread(crc, sizeof(uint8_t), 4, file);

    // Extract IHDR information
    uint32_t width = read_uint32_be(&ihdr_data[0]);
    uint32_t height = read_uint32_be(&ihdr_data[4]);
    uint8_t bit_depth = ihdr_data[8];
    uint8_t color_type = ihdr_data[9];
    uint8_t compression_method = ihdr_data[10];
    uint8_t filter_method = ihdr_data[11];
    uint8_t interlace_method = ihdr_data[12];

    printf("PNG IHDR Chunk:\n");
    printf(" - Width: %u pixels\n", width);
    printf(" - Height: %u pixels\n", height);
    printf(" - Bit Depth: %u\n", bit_depth);
    printf(" - Color Type: %u\n", color_type);
    printf(" - Compression Method: %u\n", compression_method);
    printf(" - Filter Method: %u\n", filter_method);
    printf(" - Interlace Method: %u\n", interlace_method);
}

// Function to parse a basic PNG file
// HL: hl.func @parse_png
void parse_png(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        return;
    }

    // Read and validate the PNG signature (8 bytes)
    uint8_t signature[8];
    fread(signature, sizeof(uint8_t), 8, file);
    uint8_t png_signature[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    if (memcmp(signature, png_signature, 8) != 0) {
        printf("Invalid PNG signature.\n");
        fclose(file);
        return;
    }
    printf("Valid PNG signature detected.\n");

    // Parse chunks, focusing on IHDR for simplicity
    parse_png_ihdr_chunk(file);

    // Close the file
    fclose(file);
}

int main() {
    const char *filename = "image.png";
    parse_png(filename);
    return 0;
}
