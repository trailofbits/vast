testFolder="~/vastTest/allMlirTests/"
rm error.txt
for file in ~/vastTest/testMlir/*; do
    echo $file
    base_name=$(basename ${file})
    #echo $base_name
    ~/GitRepo/vast/builds/ninja-multi-default/bin/vast-opt $file --vast-hl-lower-types --vast-hl-structs-to-tuples > ~/vastTest/optMlir/$base_name.mlir 2>> ./error.txt
    echo "------------" >> ./error.txt
done
