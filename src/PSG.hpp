#ifndef PGD_HPP
#define PGD_HPP
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <stdlib.h>
#include <cmath>
#include <tuple>
#include <Eigen/Eigen>

typedef Eigen::Array<int, Eigen::Dynamic,1> ArrayXi;
typedef Eigen::Array<float, Eigen::Dynamic,1> ArrayXf;
typedef Eigen::Array<float, Eigen::Dynamic,1> ArrayXd;
typedef Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> ArrayXXf;


class PSG
{
  int N;
  int verbosity;
  int n;
  int m;
  int k;
  float b;
  float b_sqrt;
  int prng_seed;

  float lr;
  float decay;

  float min_rating;
  float max_rating;

  ArrayXi u;
  ArrayXi v;
  ArrayXf r;

  int N_test;
  ArrayXi u_test;
  ArrayXi v_test;
  ArrayXf r_test;
  bool use_testset = false;

  ArrayXXf L;
  ArrayXXf R;
  ArrayXi perm;

  // Internal statistics
  float last_epoch_time = -1;
  float last_train_metric_time = -1;
  float last_test_metric_time = -1;
  std::vector<double> train_rmse_history;
  std::vector<double> test_rmse_history;
  std::vector<double> train_mae_history;
  std::vector<double> test_mae_history;

  // Internal functions
  ArrayXf predict_batch(ArrayXi& u, ArrayXi& v, int lo_ind, int hi_ind);
  std::tuple<float, float, float> calculate_metrics(ArrayXi& u, ArrayXi& v, ArrayXf& r);
  void permute();
  ArrayXf& project(ArrayXf& v);
  float calc_train_RMSE();
  float calc_test_RMSE();
  void run_epoch();

public:
  PSG(int verbosity, int n, int m, int k, float b, int prng_seed, float prng_uni_scale,
      ArrayXi& u, ArrayXi& v, ArrayXf& r)
    :  N(u.rows())
    ,  verbosity(verbosity)
    ,  n(n)
    ,  m(m)
    ,  k(k)
    ,  b(b)
    ,  b_sqrt(std::sqrt(b))
    ,  prng_seed(prng_seed)                        // TODO: not used!
    ,  L(ArrayXXf::Random(n, k) * prng_uni_scale)  // uniform(-1,1) * small number
    ,  R(ArrayXXf::Random(m, k) * prng_uni_scale)  // ...
    ,  u(u)
    ,  v(v)
    ,  r(r)
    ,  perm(ArrayXi::LinSpaced(N, 0, N))          // identity permutation
  {
  };

  void set_testset(ArrayXi& u, ArrayXi& v, ArrayXf& r);
  void optimize(int epochs, float eta, float decay);
  ArrayXXf get_L();
  ArrayXXf get_R();
  std::pair<std::vector<double>, std::vector<double> > get_train_metrics_history();
  std::pair<std::vector<double>, std::vector<double> > get_test_metrics_history();
};

#endif
