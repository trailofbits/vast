# Usage of scripts
## Test related:
  + csmith.sh: call CSmith to generate a random C program; used by runVastCCOnce.sh. 
  + transferCsmith.py: use a CSmith generated program as input, output a transferred program to vast-cc; used by runVastCCOnce.sh
  + runVastCCOnce.sh: this script execute the workflow ( csmith.sh -> transferCsmith.sh -> vast-cc) and generate an IR file as output. 
  + runVastOptOnce.sh: send an IR file to vast-opt. 
  + runOnce.sh: execute whole workflow once, runVastCCOnce - > runVastOptOnce. 
## Others 
  + vast-cc.sh: For all c files in `testFolder`, vast-cc use them as input and generate IR files as output. 
  + vast-mutate.sh: For all IR files in `testFolder`, vast-mutate generate 10 different IR mutations as output.
  + vast-opt.sh: For all IR files in `testFolder`, vast-opt use them as input and generate output.
