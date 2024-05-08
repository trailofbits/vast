## Debugging

VAST makes use of the MLIR infrastructure to facilitate the reproduction of crashes within the pipeline. You can refer to the MLIR documentation on [crash and failure reproduction](https://mlir.llvm.org/docs/PassManagement/#crash-and-failure-reproduction) for more details. We provide a similar set of options in vast-front to aid in debugging.

### Generating Crash Reproducers

To generate a minimal reproducer for a crashed pipeline of `vast-front`, use the following option:

```
-vast-emit-crash-reproducer="reproducer.mlir" 
```

This option disables multithreading to ensure a comprehensive crash report. You can then load and examine the crash report using the following command:

```
vast-opt -run-reproducer reproducer.mlir
```

### Pipeline Information

To obtain a detailed insight into the pipeline, you can use the following option of `vast-front`:

```
-vast-print-pipeline
```

This option dumps the pipeline string to the standard error stream. You can use this information for a more specific investigation of the pipeline. Execute the pipeline with the printed string using the following command:

```
vast-opt --pass-pipeline="pipeline-string"
```

### Debug Pipeline

With the `-vast-debug` option, you get more detailed crash reports. It shows MLIR operations when there's an error and provides current stack traces.

Sometimes it is needed to examine the results of conversion steps more closely to discover what went wrong. `-vast-snapshot-at=pass1;...;passN` will instrument conversion pipeline to store a result of `passN` into a file after it is applied. Name of the file will be in the form of: `basename.pass_name`.
Passing `"*"` in the string will result in output after every step.
