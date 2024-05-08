// RUN: %vast-front %s -vast-emit-mlir=hl -o - | %file-check %s
// RUN: %vast-front %s -vast-emit-mlir=hl -o - > %t && %vast-opt %t | diff -B %t -

// CHECK: unsup.decl "CXXRecord::Student"
struct  Student {
  // CHECK: unsup.stmt "CXXConstructExpr"
  unsigned int id;
  char name[40];
};

Student aux;

void StudentRegistry(struct Student *students, int index)
{
   aux = students[index];
}
