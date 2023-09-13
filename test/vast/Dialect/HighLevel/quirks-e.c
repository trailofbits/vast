// RUN: %vast-cc --ccopts -xc --from-source %s | %file-check %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -
// REQUIRES: qualified-type

// adapted from https://gist.github.com/fay59/5ccbe684e6e56a7df8815c3486568f01

typedef int empty_array_t[0];
typedef struct {} empty_struct_t;
typedef int array_t[10];
typedef struct { int f; } struct_t;
typedef float vector_t __attribute__((ext_vector_type(4)));

// {} can initialize structs and arrays and vectors, but not scalars:
empty_array_t ea = {};
empty_struct_t es = {};
array_t a = {};
struct_t s = {};
vector_t v = {};

// {0} can initialize any data type, including empty arrays/structs.
empty_array_t eaa = {0};
empty_struct_t ess = {0};
array_t aa = {0};
struct_t bb = {0};
vector_t cc = {0};
void* dd = {0}; // <-- happy!
int ee = {0}; // <-- happy!
