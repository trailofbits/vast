// RUN: %vast-front %s -vast-emit-mlir=hl -o - | %file-check %s
// RUN: %vast-front %s -vast-emit-mlir=hl -o - > %t && %vast-opt %t | diff -B %t -

// CHECK: unsup.decl "Namespace::test" :
namespace test {
struct FILE;
extern FILE* file_handle;
} // namespace test

test::FILE* getHandle(void) {
  return test::file_handle;
}
