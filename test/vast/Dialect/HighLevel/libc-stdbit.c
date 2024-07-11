// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-DAG: hl.func @stdc_leading_zeros_uc
// CHECK-DAG: hl.func @stdc_leading_zeros_us
// CHECK-DAG: hl.func @stdc_leading_zeros_ui
// CHECK-DAG: hl.func @stdc_leading_zeros_ul
// CHECK-DAG: hl.func @stdc_leading_zeros_ull

// CHECK-DAG: hl.func @stdc_leading_ones_uc
// CHECK-DAG: hl.func @stdc_leading_ones_us
// CHECK-DAG: hl.func @stdc_leading_ones_ui
// CHECK-DAG: hl.func @stdc_leading_ones_ul
// CHECK-DAG: hl.func @stdc_leading_ones_ull

// CHECK-DAG: hl.func @stdc_trailing_zeros_uc
// CHECK-DAG: hl.func @stdc_trailing_zeros_us
// CHECK-DAG: hl.func @stdc_trailing_zeros_ui
// CHECK-DAG: hl.func @stdc_trailing_zeros_ul
// CHECK-DAG: hl.func @stdc_trailing_zeros_ull

// CHECK-DAG: hl.func @stdc_trailing_ones_uc
// CHECK-DAG: hl.func @stdc_trailing_ones_us
// CHECK-DAG: hl.func @stdc_trailing_ones_ui
// CHECK-DAG: hl.func @stdc_trailing_ones_ul
// CHECK-DAG: hl.func @stdc_trailing_ones_ull

// CHECK-DAG: hl.func @stdc_first_leading_zero_uc
// CHECK-DAG: hl.func @stdc_first_leading_zero_us
// CHECK-DAG: hl.func @stdc_first_leading_zero_ui
// CHECK-DAG: hl.func @stdc_first_leading_zero_ul
// CHECK-DAG: hl.func @stdc_first_leading_zero_ull

// CHECK-DAG: hl.func @stdc_first_leading_one_uc
// CHECK-DAG: hl.func @stdc_first_leading_one_us
// CHECK-DAG: hl.func @stdc_first_leading_one_ui
// CHECK-DAG: hl.func @stdc_first_leading_one_ul
// CHECK-DAG: hl.func @stdc_first_leading_one_ull

// CHECK-DAG: hl.func @stdc_first_trailing_zero_uc
// CHECK-DAG: hl.func @stdc_first_trailing_zero_us
// CHECK-DAG: hl.func @stdc_first_trailing_zero_ui
// CHECK-DAG: hl.func @stdc_first_trailing_zero_ul
// CHECK-DAG: hl.func @stdc_first_trailing_zero_ull

// CHECK-DAG: hl.func @stdc_first_trailing_one_uc
// CHECK-DAG: hl.func @stdc_first_trailing_one_us
// CHECK-DAG: hl.func @stdc_first_trailing_one_ui
// CHECK-DAG: hl.func @stdc_first_trailing_one_ul
// CHECK-DAG: hl.func @stdc_first_trailing_one_ull

// CHECK-DAG: hl.func @stdc_count_zeros_uc
// CHECK-DAG: hl.func @stdc_count_zeros_us
// CHECK-DAG: hl.func @stdc_count_zeros_ui
// CHECK-DAG: hl.func @stdc_count_zeros_ul
// CHECK-DAG: hl.func @stdc_count_zeros_ull

// CHECK-DAG: hl.func @stdc_count_ones_uc
// CHECK-DAG: hl.func @stdc_count_ones_us
// CHECK-DAG: hl.func @stdc_count_ones_ui
// CHECK-DAG: hl.func @stdc_count_ones_ul
// CHECK-DAG: hl.func @stdc_count_ones_ull

// CHECK-DAG: hl.func @stdc_has_single_bit_uc
// CHECK-DAG: hl.func @stdc_has_single_bit_us
// CHECK-DAG: hl.func @stdc_has_single_bit_ui
// CHECK-DAG: hl.func @stdc_has_single_bit_ul
// CHECK-DAG: hl.func @stdc_has_single_bit_ull

// CHECK-DAG: hl.func @stdc_bit_width_uc
// CHECK-DAG: hl.func @stdc_bit_width_us
// CHECK-DAG: hl.func @stdc_bit_width_ui
// CHECK-DAG: hl.func @stdc_bit_width_ul
// CHECK-DAG: hl.func @stdc_bit_width_ull

// CHECK-DAG: hl.func @stdc_bit_floor_uc
// CHECK-DAG: hl.func @stdc_bit_floor_us
// CHECK-DAG: hl.func @stdc_bit_floor_ui
// CHECK-DAG: hl.func @stdc_bit_floor_ul
// CHECK-DAG: hl.func @stdc_bit_floor_ull

// CHECK-DAG: hl.func @stdc_bit_ceil_uc
// CHECK-DAG: hl.func @stdc_bit_ceil_us
// CHECK-DAG: hl.func @stdc_bit_ceil_ui
// CHECK-DAG: hl.func @stdc_bit_ceil_ul
// CHECK-DAG: hl.func @stdc_bit_ceil_ull

#include <stdbit.h>
int main() {
    int a = stdc_trailing_zeros(0);
}
