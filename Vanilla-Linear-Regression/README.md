# Vanilla Linear Regression

We use a known linear model to generate training data first, then we train a linear model on training data.
You can set your desired hyperparameters in 'flags.cfg'. Run the program with 'bazel', using below commannds:
'''
bazel build app
bazel-bin/app --flagfile=flags.cfg
'''

