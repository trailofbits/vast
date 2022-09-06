testFolder="~/vastTest/testC/"
for file in ~/vastTest/testC/*; do
    #echo $file
    base_name=$(basename ${file})
    #echo $base_name
    #echo "~/GitRepo/vast/builds/ninja-multi-default/bin/vast-cc $file --from-source  > ~/vastTest/testMlir/$base_name.mlir"
    ~/GitRepo/vast/builds/ninja-multi-default/bin/vast-cc $file --from-source  --ccopts -xc > ~/vastTest/testMlir/$base_name.mlir
done
