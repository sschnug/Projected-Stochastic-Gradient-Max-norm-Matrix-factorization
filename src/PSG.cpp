#include "PSG.hpp"
#include <chrono>
#include <algorithm>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace std::chrono;

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;

template<typename Scalar>
struct CwiseClampOp {
  CwiseClampOp(const Scalar& inf, const Scalar& sup) : m_inf(inf), m_sup(sup) {}
  const Scalar operator()(const Scalar& x) const {return x<m_inf ? m_inf : (x>m_sup ? m_sup : x); }
  Scalar m_inf, m_sup;
};

void PSG::permute()
{
  // TODO why is this reinit needed?
  // otherwise segfault -> min/max(perm) is out of range
  this->perm = ArrayXi::LinSpaced(this->N, 0, this->N);
  std::random_shuffle(this->perm.data(), this->perm.data() + this->N);
}

ArrayXf& PSG::project(ArrayXf& v)
{
  float v_norm_squared = (v*v).sum();
  float v_norm = std::sqrt(v_norm_squared);

  if(v_norm_squared >= this->b)
  {
    v = (this->b_sqrt * v) / v_norm;
  }

  return v;
}

ArrayXf PSG::predict_batch(ArrayXi& u, ArrayXi& v,
                           int lo_ind, int hi_ind)
{
  ArrayXi full_k_select = ArrayXi::LinSpaced(this->k, 0, this->k);
  ArrayXi ind_select = ArrayXi::LinSpaced(hi_ind - lo_ind, lo_ind, hi_ind);

  ArrayXi u_batch = u(ind_select);
  ArrayXi v_batch = v(ind_select);

  ArrayXf preds = (this->L(u_batch, full_k_select) *
                   this->R(v_batch, full_k_select)).rowwise().sum().unaryExpr(
                     CwiseClampOp<float>(this->min_rating, this->max_rating));

  return preds;
}

std::tuple<float, float, float> PSG::calculate_metrics(ArrayXi& u, ArrayXi& v, ArrayXf& r)
{
  const int BATCHSIZE = 4096;

  auto t0 = Time::now();

  double error_rmse = 0.0;
  double error_mae = 0.0;

  int offset = 0;
  int N = u.rows();

  while(offset < N)
  {
    int ub = std::min(offset + BATCHSIZE, N);
    int batch_size = ub - offset;

    ArrayXi ind_select = ArrayXi::LinSpaced(batch_size, offset, ub);
    ArrayXf batch_preds = this->predict_batch(u, v, offset, ub);
    ArrayXf batch_r = r(ind_select);

    ArrayXd error_raw = batch_preds - batch_r;

    error_rmse += (error_raw * error_raw).sum();
    error_mae += error_raw.abs().sum();

    offset += BATCHSIZE;
  }

  auto t1 = Time::now();
  ms d = std::chrono::duration_cast<ms>(t1 - t0);
  double time_used = double(d.count() / 1000.);

  return std::make_tuple(
          std::sqrt(error_rmse / N),
          error_mae / N,
          time_used);
}

void PSG::run_epoch()
{
  auto t0 = Time::now();

  // Permute samples
  this->permute();

  // Iterate through sparse observations
  for(int ind=0; ind<(this->N); ++ind)
  {
    // Access observation
    int perm_ind = this->perm[ind];

    int u = this->u[perm_ind];
    int v = this->v[perm_ind];
    float r = this->r[perm_ind];

    // Current prediction
    float pred = (this->L.row(u) * this->R.row(v)).sum();

    // Update rule
    float grad = pred - r;
    ArrayXf L_pre = this->L.row(u) - this->lr * grad * this->R.row(v);
    ArrayXf R_pre = this->R.row(v) - this->lr * grad * this->L.row(u);

    // Projection
    this->L.row(u) = this->project(L_pre);
    this->R.row(v) = this->project(R_pre);
  }

  // Decrease learning-rate
  this->lr *= this->decay;

  auto t1 = Time::now();
  ms d = std::chrono::duration_cast<ms>(t1 - t0);
  this->last_epoch_time = double(d.count() / 1000.);
}

void PSG::optimize(int epochs, float eta, float decay)
{
  // Preprocessing
  this->min_rating = this->r.minCoeff();
  this->max_rating = this->r.maxCoeff();  // TODO do in 1 pass?

  this->lr = eta;
  this->decay = decay;

  for(int epoch=0; epoch < epochs; ++epoch)
  {
    if(verbosity > 0)
    {
      py::print("Epoch: ", epoch);
      py::print("  lr: ", this->lr);
    }

    // Do PSG steps
    this->run_epoch();

    // Calculate current train-metrics
    std::tuple<float, float, float> calc_metrics_train_res =
      this->calculate_metrics(this->u, this->v, this->r);
    this->last_train_metric_time = std::get<2>(calc_metrics_train_res);

    std::tuple<float, float, float> calc_metrics_test_res;
    if(this->use_testset)
    {
      calc_metrics_test_res =
        this->calculate_metrics(this->u_test, this->v_test, this->r_test);
        this->last_test_metric_time = std::get<2>(calc_metrics_test_res);
    }

    if(verbosity > 0)
    {
      py::print("  time epoch train (s): ", this->last_epoch_time);


      py::print("  Train RMSE: ", std::get<0>(calc_metrics_train_res));
      py::print("  Train MAE: ", std::get<1>(calc_metrics_train_res));
      py::print("  time epoch train-metrics calc (s): ", this->last_train_metric_time);

      if(this->use_testset)
      {
        py::print("  Test RMSE: ", std::get<0>(calc_metrics_test_res));
        py::print("  Test MAE: ", std::get<1>(calc_metrics_test_res));
        py::print("  time epoch test-metrics calc (s): ", this->last_test_metric_time);
      }
    }

    // Internal stats
    this->train_rmse_history.push_back(std::get<0>(calc_metrics_train_res));
    this->train_mae_history.push_back(std::get<1>(calc_metrics_train_res));
    if(this->use_testset)
    {
      this->test_rmse_history.push_back(std::get<0>(calc_metrics_test_res));
      this->test_mae_history.push_back(std::get<1>(calc_metrics_test_res));
    }
  }
}

void PSG::set_testset(ArrayXi& u, ArrayXi& v, ArrayXf& r)
{
  this->u_test = u;
  this->v_test = v;
  this->r_test = r;
  this->use_testset = true;
  this->N_test = u.rows();
}

ArrayXXf PSG::get_L()
{
  return this->L;
}

ArrayXXf PSG::get_R()
{
  return this->R;
}

std::pair<std::vector<double>, std::vector<double> > PSG::get_train_metrics_history()
{
  return std::make_pair(this->train_rmse_history, this->train_mae_history);
}

std::pair<std::vector<double>, std::vector<double> > PSG::get_test_metrics_history()
{
  return std::make_pair(this->test_rmse_history, this->test_mae_history);
}
