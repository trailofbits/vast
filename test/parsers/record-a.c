// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s -check-prefix=HL
// RUN: %vast-front -vast-show-locs -vast-loc-attrs -vast-emit-mlir=hl %s -o - | %vast-opt -vast-hl-to-lazy-regions -o %t.mlir
// RUN: %vast-detect-parsers -vast-hl-to-parser -vast-parser-reconcile-casts -reconcile-unrealized-casts %t.mlir -o - | %file-check %s -check-prefix=PARSER

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Function to calculate a simple checksum (XOR all bytes)
// HL: hl.func @calculate_checksum
// PARSER: hl.func @calculate_checksum
uint32_t calculate_checksum(uint8_t *data, size_t length) {
    uint32_t checksum = 0;
    for (size_t i = 0; i < length; ++i) {
        checksum ^= data[i];
    }
    return checksum;
}

// Function to parse a binary file containing multiple records
// HL: hl.func @parse_binary_records
// PARSER: hl.func @parse_binary_records
void parse_binary_records(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        return;
    }

    // Go to the end of the file to find the file length
    fseek(file, 0, SEEK_END);
    size_t file_length = ftell(file);
    rewind(file);

    // Read the file contents into a buffer
    uint8_t *buffer = (uint8_t *)malloc(file_length);
    fread(buffer, sizeof(uint8_t), file_length, file);

    // Close the file after reading
    fclose(file);

    size_t idx = 0;
    printf("Parsing records:\n");

    // Parsing Part: Read records until the checksum
    while (idx < file_length - 4) { // Last 4 bytes are the checksum
        // Read record type (2 bytes)
        uint16_t record_type = (buffer[idx] << 8) | buffer[idx + 1];
        idx += 2;

        // Read record length (2 bytes)
        uint16_t record_length = (buffer[idx] << 8) | buffer[idx + 1];
        idx += 2;

        // Read record data based on length
        uint8_t *record_data = &buffer[idx];
        idx += record_length;

        // Print record information
        printf("Record Type: %u, Length: %u, Data: ", record_type, record_length);
        for (uint16_t i = 0; i < record_length; ++i) {
            printf("%02X ", record_data[i]);
        }
        printf("\n");
    }

    // Parsing Part: Read and verify the checksum (last 4 bytes)
    uint32_t checksum = (buffer[idx] << 24) | (buffer[idx + 1] << 16) | (buffer[idx + 2] << 8) | buffer[idx + 3];
    printf("Read checksum: %08X\n", checksum);

    // Non-Parsing Part: Calculate checksum of the data
    uint32_t calculated_checksum = calculate_checksum(buffer, file_length - 4);
    printf("Calculated checksum: %08X\n", calculated_checksum);

    // Validate checksum
    if (checksum == calculated_checksum) {
        printf("Checksum verification passed.\n");
    } else {
        printf("Checksum verification failed.\n");
    }

    // Free memory
    free(buffer);
}

int main() {
    const char *filename = "records_data.bin";
    parse_binary_records(filename);
    return 0;
}
