./csmith.sh
python ./transferCsmith.py ./random.c ./random.py.c
/home/spica/GitRepo/vast/builds/ninja-multi-default/bin/vast-cc ./random.py.c --from-source --ccopts -xc > ./random.py.c.mlir
