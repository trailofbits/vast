import sys

keywords=[["uint16_t","unsigned short"],["uint64_t","unsigned long long"],["uint32_t","unsigned int"],["int16_t","short"],["int32_t","int"]
    ,["int64_t","long long"]]

def usefulData(line):
    return line.startswith("mlir_tablegen")

def filterResult(data):
    res=[]
    skip5=0
    for line in data:
        if skip5<5:
            skip5+=1
            continue
        for keyword in keywords:
            line=line.replace(keyword[0],keyword[1])
        res.append(line)
    return res

def readData(INPUT):
    with open(INPUT,"r") as fin:
        return fin.readlines()

def writeData(OUTPUT,data):
    with open(OUTPUT,"w") as fout:
        for line in data:
            fout.write(line)

def main():
    assert len(sys.argv)==3
    INPUT=sys.argv[1]
    OUTPUT=sys.argv[2]
    data=filterResult(readData(INPUT))
    writeData(OUTPUT,data)

main()
