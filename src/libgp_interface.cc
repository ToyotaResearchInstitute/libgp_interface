#include <libgp_interface.h>

LibgpInterface::LibgpInterface() {}
LibgpInterface::~LibgpInterface() {}

void LibgpInterface::Initialize(unsigned int dim, const std::string &cov_kernel,
                                const std::vector<double> &hyp_params) {
  gp_ = std::make_shared<libgp::GaussianProcess>(dim, cov_kernel);
  size_t num_hyp = gp_->covf().get_param_dim();

  assert(hyp_params.size() == num_hyp);

  Eigen::VectorXd params(num_hyp);
  for (size_t i = 0; i < num_hyp; i++) {
    params[i] = hyp_params[i];
  }

  gp_->covf().set_loghyper(params);
}

void LibgpInterface::Train(const std::vector<double> &x,
                           const std::vector<double> &y) {
  // Check for input size.
  int input_dim = GetInputDim();
  size_t num_samples = y.size();
  assert(x.size() == num_samples * input_dim);

  for (size_t i = 0; i < num_samples; i++) {
    gp_->add_pattern(&x[i * input_dim], y[i]);
  }
}

void LibgpInterface::Predict(int num_samples, const std::vector<double> &x,
                             std::vector<double> *y_pred) {
  // Check for input size.
  int input_dim = GetInputDim();
  assert(x.size() == num_samples * input_dim);

  // Generate predictions.
  y_pred->clear();
  for (int i = 0; i < num_samples; i++) {
    y_pred->push_back(gp_->f(&x[i * input_dim]));
  }
}

void LibgpInterface::Predict(int num_samples, const std::vector<double> &x,
                             std::vector<double> *y_pred,
                             std::vector<double> *y_var) {
  // Check for input size.
  int input_dim = GetInputDim();
  assert(x.size() == num_samples * input_dim);

  // Generate predictions.
  y_pred->clear();
  y_var->clear();
  for (int i = 0; i < num_samples; i++) {
    y_pred->push_back(gp_->f(&x[i * input_dim]));
    y_var->push_back(gp_->var(&x[i * input_dim]));
  }
}

int LibgpInterface::GetInputDim() {
  return gp_->get_input_dim();
}

int LibgpInterface::GetNumSamples() {
  return gp_->get_sampleset_size();
}

void LibgpInterface::Clear() {
  gp_->clear_sampleset();
}
