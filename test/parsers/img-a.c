// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s -check-prefix=HL
// RUN: %vast-front -vast-show-locs -vast-loc-attrs -vast-emit-mlir=hl %s -o - | %vast-opt -vast-hl-to-lazy-regions -o %t.mlir
// RUN: %vast-detect-parsers -vast-hl-to-parser -vast-parser-reconcile-casts -reconcile-unrealized-casts %t.mlir -o - | %file-check %s -check-prefix=PARSER

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

// Function to display the image as ASCII (for grayscale images)
// HL: hl.func @display_ascii_image
// PARSER: hl.func @display_ascii_image
void display_ascii_image(uint8_t *pixel_data, uint16_t width, uint16_t height) {
    printf("Displaying image as ASCII:\n");
    for (uint16_t y = 0; y < height; ++y) {
        for (uint16_t x = 0; x < width; ++x) {
            // Map pixel values (0-255) to ASCII characters
            uint8_t pixel = pixel_data[y * width + x];
            char ascii_char = (pixel < 128) ? '#' : ' ';
            printf("%c", ascii_char);
        }
        printf("\n");
    }
}

// Function to parse a binary file containing a SIMPL image
// HL: hl.func @parse_simpl_image
// PARSER: hl.func @parse_simpl_image
void parse_simpl_image(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        return;
    }

    // Read and validate the magic number (4 bytes)
    char magic_number[5] = {0};
    fread(magic_number, sizeof(char), 4, file);
    if (strncmp(magic_number, "SIML", 4) != 0) {
        printf("Invalid file format!\n");
        fclose(file);
        return;
    }
    printf("Valid SIMPL image format detected.\n");

    // Read the width (2 bytes) and height (2 bytes)
    uint16_t width, height;
    fread(&width, sizeof(uint16_t), 1, file);
    fread(&height, sizeof(uint16_t), 1, file);

    // Read bits per pixel (1 byte)
    uint8_t bpp;
    fread(&bpp, sizeof(uint8_t), 1, file);

    // Only support 8-bit grayscale images in this example
    if (bpp != 8) {
        printf("Unsupported bits per pixel: %u\n", bpp);
        fclose(file);
        return;
    }

    printf("Width: %u, Height: %u, Bits per Pixel: %u\n", width, height, bpp);

    // Read pixel data (width * height bytes)
    size_t pixel_data_size = width * height;
    uint8_t *pixel_data = (uint8_t *)malloc(pixel_data_size);
    fread(pixel_data, sizeof(uint8_t), pixel_data_size, file);

    // Close the file
    fclose(file);

    // Display the image as ASCII (Non-Parsing Part)
    display_ascii_image(pixel_data, width, height);

    // Free memory
    free(pixel_data);
}

int main() {
    const char *filename = "image.simpl";
    parse_simpl_image(filename);
    return 0;
}
