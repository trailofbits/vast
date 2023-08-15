// RUN: vast-front %s -vast-emit-mlir=hl -o - | FileCheck %s
// RUN: vast-front %s -vast-emit-mlir=hl -o - > %t && vast-opt %t | diff -B %t -

// CHECK: hl.struct
struct  Student {
  unsigned int id;
  char name[40];
};

Student aux;

void StudentRegistry(struct Student *students, int index)
{
   aux = students[index];
}
