testFolder="~/vastTest/testMlir/"
for file in ~/vastTest/testMlir/*; do
    echo $file
    base_name=$(basename ${file})
    #echo $base_name
    ~/GitRepo/vast/builds/ninja-multi-default/bin/vast-mutate $file ~/vastTest/allMlirTests/ -n 10 -verbose
done
