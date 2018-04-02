# Intro
This is an (prototype-)implementation of:

> LEE, Jason D., et al. Practical large-scale optimization for max-norm regularization. In: Advances in Neural Information Processing Systems. 2010. S. 1297-1305.

with an [Eigen](http://eigen.tuxfamily.org)-based **C++-core**, wrapped for **usage through Python** (only!).

The implementation (and example) is targeting the [collaborative-filtering]() setting, optimizing a low-rank (max-norm as rank-regularizer) matrix-factorization.

(A similar pure-python version lives [here](https://github.com/sschnug/MaxNormRegCollaborativeFiltering_SGD) including further references).

# Algorithm-implementation (Theory)
- In terms of the presented algorithms in the above paper:
  - Stochastic-gradient + Bound-constrained (formulas 6 + 10)
  - Sparse processing (explicit treating of observations only)
- Vanilla-SGD without momentum
  - Remark: Adagrad (not part of this code) works badly (less good solutions; quite some overhead)
  - Remark: Mini-batch works badly (slow convergence)
    - (Alternatives: Hogwild / Jellyfish)

# Algorithm-implementation (Practice)
- Build around Eigen's Array-class
  - Eigen (dev-version) is shipped
- Single-core
- Partially vectorized
  - E.g. batch-wise metric-calculation
- Deterministic
- Single-precision floats!

# Install
Remember:

- Prototype!
  - Core-algorithms look good (although a hardcoded-batchsize can be found for example)
  - Install/Setup is waggly (and not much tested)
  - Only python3 was tested

Steps:

- Prepare python, numpy, matplotlib, pybind11
- Prepare necessary build-tools needed (see pybind11)
- ```sudo python3 setup.py install```
- (**Warning**: Hardcoded ```CFLAGS``` in setup.py: change for your machine!)

# Example
There is a full collaborative-filtering example based on [Movielens](https://grouplens.org/datasets/movielens/) data (1M; easily runs on 20M/full) in ```examples/movielens.py```.

For a minimally-tuned parameter-setting using the 20M-dataset, output (which is quite a good result; even compared to modern results in the no side-information setting) could look like:

    Train:  18000236
    Test:  2000027
    Epoch:  0
      lr:  0.05000000074505806
      time epoch train (s):  39.12699890136719
      Train RMSE:  0.8113546371459961
      Train MAE:  0.6227260231971741
      time epoch train-metrics calc (s):  57.15599822998047
      Test RMSE:  0.8319584131240845
      Test MAE:  0.638018786907196
      time epoch test-metrics calc (s):  6.484000205993652
    ...
    Epoch:  99
      lr:  0.0003116064181085676
      time epoch train (s):  38.430999755859375
      Train RMSE:  0.6998533010482788
      Train MAE:  0.5307825207710266
      time epoch train-metrics calc (s):  55.64799880981445
      Test RMSE:  0.7872740030288696
      Test MAE:  0.6015247106552124
      time epoch test-metrics calc (s):  6.059999942779541

![Plot](https://i.imgur.com/kuJXhlr.png)

(Looking at the plot, it can be argued, that further param-tuning should be done!)
