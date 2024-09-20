// RUN: %vast-cc1 -vast-emit-mlir-after=vast-lower-value-categories %s -o - | %file-check %s

void scoped_store()
{
    // CHECK: [[A:%[0-9]+]] = ll.alloca : !hl.ptr<si32>
    int a;
    {
        // CHECK: [[B:%[0-9]+]] = ll.alloca : !hl.ptr<si32>
        // CHECK: [[AV:%[0-9]+]] = ll.load [[A]]
        // CHECK: ll.store [[B]], [[AV]]
        int b = a;
    }
}

void shadowed_store() {
    // CHECK: [[A1:%[0-9]+]] = ll.alloca : !hl.ptr<si32>
    int a;
    {
        // CHECK: [[A2:%[0-9]+]] = ll.alloca : !hl.ptr<si32>
        int a;
        // CHECK: [[B:%[0-9]+]] = ll.alloca : !hl.ptr<si32>
        // CHECK: [[AV2:%[0-9]+]] = ll.load [[A2]]
        // CHECK: ll.store [[B]], [[AV2]]
        int b = a;
    }
}

void param_store(int p) {
    // CHECK: [[P:%[0-9]+]] = ll.alloca : !hl.ptr<si32>
    // CHECK: ll.store [[P]], %arg0

    // CHECK: [[B:%[0-9]+]] = ll.alloca : !hl.ptr<si32>
    // CHECK: [[PV:%[0-9]+]] = ll.load [[P]]
    // CHECK: ll.store [[B]], [[PV]]
    int b = p;
}

void scoped_param_store(int p) {
    // CHECK: [[P:%[0-9]+]] = ll.alloca : !hl.ptr<si32>
    // CHECK: ll.store [[P]], %arg0

    {
        // CHECK: [[B:%[0-9]+]] = ll.alloca : !hl.ptr<si32>
        // CHECK: [[PV:%[0-9]+]] = ll.load [[P]]
        // CHECK: ll.store [[B]], [[PV]]
        int b = p;
    }
}

void shadowed_param_store(int p) {
    // CHECK: [[P1:%[0-9]+]] = ll.alloca : !hl.ptr<si32>
    // CHECK: ll.store [[P]], %arg0

    {
        // CHECK: [[P2:%[0-9]+]] = ll.alloca : !hl.ptr<si32>
        int p;

        // CHECK: [[B:%[0-9]+]] = ll.alloca : !hl.ptr<si32>
        // CHECK: [[PV2:%[0-9]+]] = ll.load [[P2]]
        // CHECK: ll.store [[B]], [[PV2]]
        int b = p;
    }
}
