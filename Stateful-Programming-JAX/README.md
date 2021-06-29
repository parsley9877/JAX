# Functional Programming with JAX

## Introduction

This directory contains a simple example of handling states, with jit-compiled function of a class.

You can set your desired initial state in `flags.cfg`. Run the program with `bazel`, using below commannds:

## Running the Experiment

```
bazel build app
bazel-bin/app --flagfile=flags.cfg
```

## Sample Generated Output
![alt text](https://github.com/parsley9877/jax-tutorial/blob/main/Stateful-Programming-JAX/output_sample.png?raw=true)
