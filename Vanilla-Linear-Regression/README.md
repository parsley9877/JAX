# Vanilla Linear Regression

## Introduction

We use a known linear model to generate training data first, then we train a linear model on training data.

You can set your desired hyperparameters in `flags.cfg`. Run the program with `bazel`, using below commannds:

## Running the Experiment

```
bazel build app
bazel-bin/app --flagfile=flags.cfg
```
## Sample Generated Output
![alt text](https://github.com/parsley9877/jax-tutorial/blob/main/Vanilla-Linear-Regression/trained_model.png?raw=true)
