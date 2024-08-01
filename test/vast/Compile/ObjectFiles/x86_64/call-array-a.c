// RUN: %vast-front -target x86_64 -c -vast-pipeline=with-abi -S -emit-llvm -o %t.vast.ll %s && %cc -c -S -emit-llvm -xc %s.driver -o %t.clang.ll  && %cc %t.vast.ll %t.clang.ll -o %t && (%t; test $? -eq 0)
// REQUIRES: clang

int sum(int array[2])
{
    return array[0] + array[1];
}

int vast_tests() {
    int arr[2] = { 0, 5 };
    if (sum(arr) != 5)
        return 11;

    arr[0] = -5;
    if (sum(arr) != 0)
        return 12;

    return 0;
}
