// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=hl %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=HL
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=llvm %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=MLIR
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-llvm %s -o %t.ll
// RUN: %file-check --input-file=%t.ll %s -check-prefix=LLVM
// REQUIRES: call-lowering

// This file is part of the SV-Benchmarks collection of verification tasks:
// https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks
//
// SPDX-FileCopyrightText: 2022 Jana (Philipp) Berger
//
// SPDX-License-Identifier: GPL-3.0-or-later

// Prototype declarations of the functions used to communicate with the model checkers
extern unsigned long __VERIFIER_nondet_ulong();
extern long __VERIFIER_nondet_long();
extern unsigned char __VERIFIER_nondet_uchar();
extern char __VERIFIER_nondet_char();
extern unsigned short __VERIFIER_nondet_ushort();
extern short __VERIFIER_nondet_short();
extern float __VERIFIER_nondet_float();
extern double __VERIFIER_nondet_double();

extern void abort(void);
extern void __assert_fail(const char *, const char *, unsigned int, const char *) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));
void reach_error() { __assert_fail("0", "Req1_Prop1_Batch17dependencies.c", 13, "reach_error"); }
void __VERIFIER_assert(int cond) { if(!(cond)) { ERROR: {reach_error();abort();} } return; }
void assume_abort_if_not(int cond) { if(!cond) { abort(); } }



#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))
#define abs(a) (((a) < 0 ) ? -(a) : (a))





// Function prototypes


// Internal control logic variables
unsigned char isInitial = 0;

// Signal variables
unsigned char var_1_1 = 25;
unsigned char var_1_2 = 25;
unsigned char var_1_3 = 0;
unsigned char var_1_4 = 0;
signed long int var_1_5 = 4;
unsigned char var_1_7 = 0;
unsigned char var_1_8 = 1;
unsigned char var_1_9 = 0;
unsigned char var_1_10 = 0;
double var_1_11 = 7.3;
double var_1_12 = 1.25;
double var_1_13 = 128.8;
signed char var_1_14 = -1;
signed char var_1_15 = 10;
signed short int var_1_16 = 8;
unsigned long int var_1_18 = 128;
unsigned long int var_1_19 = 3963666122;

// Calibration values

// Last'ed variables
unsigned char last_1_var_1_8 = 1;
signed char last_1_var_1_14 = -1;

// Additional functions


void initially(void) {
}



void step(void) {
	// From: Req2Batch17dependencies
	if (last_1_var_1_8 && last_1_var_1_8) {
		if (last_1_var_1_8) {
			var_1_5 = ((abs (last_1_var_1_14)) - (min (var_1_4 , var_1_2)));
		} else {
			var_1_5 = 8;
		}
	}


	// From: Req3Batch17dependencies
	if (var_1_5 < var_1_3) {
		var_1_8 = ((var_1_7 || (last_1_var_1_8 || var_1_9)) && var_1_10);
	} else {
		var_1_8 = (! var_1_10);
	}


	// From: Req1Batch17dependencies
	var_1_1 = (50 + (min (var_1_2 , (min (var_1_3 , var_1_4)))));


	// From: Req5Batch17dependencies
	if ((64.4f + 1.5f) <= var_1_13) {
		if (var_1_1 >= var_1_4) {
			var_1_14 = var_1_15;
		}
	}


	// From: Req7Batch17dependencies
	if ((var_1_9 && var_1_8) || (var_1_8 && var_1_10)) {
		var_1_18 = (var_1_19 - (min ((1991720936u - var_1_1) , var_1_2)));
	} else {
		var_1_18 = var_1_2;
	}


	// From: Req4Batch17dependencies
	if (var_1_4 == (8 + var_1_14)) {
		var_1_11 = (max (var_1_12 , var_1_13));
	}


	// From: Req6Batch17dependencies
	if (var_1_12 >= 9.6) {
		if (var_1_10) {
			var_1_16 = (2 - var_1_1);
		} else {
			if ((var_1_18 * (var_1_15 & var_1_1)) <= var_1_3) {
				var_1_16 = var_1_18;
			} else {
				var_1_16 = var_1_4;
			}
		}
	}
}



void updateVariables() {
	var_1_2 = __VERIFIER_nondet_uchar();
	assume_abort_if_not(var_1_2 >= 0);
	assume_abort_if_not(var_1_2 <= 127);
	var_1_3 = __VERIFIER_nondet_uchar();
	assume_abort_if_not(var_1_3 >= 0);
	assume_abort_if_not(var_1_3 <= 127);
	var_1_4 = __VERIFIER_nondet_uchar();
	assume_abort_if_not(var_1_4 >= 0);
	assume_abort_if_not(var_1_4 <= 127);
	var_1_7 = __VERIFIER_nondet_uchar();
	assume_abort_if_not(var_1_7 >= 0);
	assume_abort_if_not(var_1_7 <= 1);
	var_1_9 = __VERIFIER_nondet_uchar();
	assume_abort_if_not(var_1_9 >= 1);
	assume_abort_if_not(var_1_9 <= 1);
	var_1_10 = __VERIFIER_nondet_uchar();
	assume_abort_if_not(var_1_10 >= 1);
	assume_abort_if_not(var_1_10 <= 1);
	var_1_12 = __VERIFIER_nondet_double();
	assume_abort_if_not((var_1_12 >= -922337.2036854765600e+13F && var_1_12 <= -1.0e-20F) || (var_1_12 <= 9223372.036854765600e+12F && var_1_12 >= 1.0e-20F ));
	var_1_13 = __VERIFIER_nondet_double();
	assume_abort_if_not((var_1_13 >= -922337.2036854765600e+13F && var_1_13 <= -1.0e-20F) || (var_1_13 <= 9223372.036854765600e+12F && var_1_13 >= 1.0e-20F ));
	var_1_15 = __VERIFIER_nondet_char();
	assume_abort_if_not(var_1_15 >= -127);
	assume_abort_if_not(var_1_15 <= 126);
	var_1_19 = __VERIFIER_nondet_ulong();
	assume_abort_if_not(var_1_19 >= 2147483647);
	assume_abort_if_not(var_1_19 <= 4294967294);
}



void updateLastVariables() {
	last_1_var_1_8 = var_1_8;
	last_1_var_1_14 = var_1_14;
}

int property() {
	return ((((((var_1_1 == ((unsigned char) (50 + (min (var_1_2 , (min (var_1_3 , var_1_4))))))) && ((last_1_var_1_8 && last_1_var_1_8) ? (last_1_var_1_8 ? (var_1_5 == ((signed long int) ((abs (last_1_var_1_14)) - (min (var_1_4 , var_1_2))))) : (var_1_5 == ((signed long int) 8))) : 1)) && ((var_1_5 < var_1_3) ? (var_1_8 == ((unsigned char) ((var_1_7 || (last_1_var_1_8 || var_1_9)) && var_1_10))) : (var_1_8 == ((unsigned char) (! var_1_10))))) && ((var_1_4 == (8 + var_1_14)) ? (var_1_11 == ((double) (max (var_1_12 , var_1_13)))) : 1)) && (((64.4f + 1.5f) <= var_1_13) ? ((var_1_1 >= var_1_4) ? (var_1_14 == ((signed char) var_1_15)) : 1) : 1)) && ((var_1_12 >= 9.6) ? (var_1_10 ? (var_1_16 == ((signed short int) (2 - var_1_1))) : (((var_1_18 * (var_1_15 & var_1_1)) <= var_1_3) ? (var_1_16 == ((signed short int) var_1_18)) : (var_1_16 == ((signed short int) var_1_4)))) : 1)) && (((var_1_9 && var_1_8) || (var_1_8 && var_1_10)) ? (var_1_18 == ((unsigned long int) (var_1_19 - (min ((1991720936u - var_1_1) , var_1_2))))) : (var_1_18 == ((unsigned long int) var_1_2)))
;
}
int main(void) {
	isInitial = 1;
	initially();

	while (1) {
		updateLastVariables();

		updateVariables();
		step();
		__VERIFIER_assert(property());
		isInitial = 0;
	}

	return 0;
}

// HL: hl.func @main () -> !hl.int

// MLIR: llvm.func @main() -> i32

// LLVM: define i32 @main()