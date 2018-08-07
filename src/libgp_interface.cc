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
  // Check input size.
  int input_dim = GetInputDim();
  size_t num_samples = y.size();
  assert(x.size() == num_samples * input_dim);

  for (size_t i = 0; i < num_samples; i++) {
    gp_->add_pattern(&x[i * input_dim], y[i]);
  }
}

void LibgpInterface::Train(const Eigen::MatrixXf &x, const Eigen::VectorXf &y) {
  // Check input size.
  int input_dim = GetInputDim();
  assert(x.rows() == input_dim);
  assert(x.cols() == y.size());

  double *x_array;
  Eigen::Map<Eigen::MatrixXd>(x_array, x.rows(), x.cols());

  for (size_t i = 0; i < y.size(); i++) {
    gp_->add_pattern(&x_array[i * input_dim], y(i));
  }
}

void LibgpInterface::Train(const std::vector<double> &x, const double y) {
  assert(x.size() == GetInputDim());
  gp_->add_pattern(&x[0], y);
}

void LibgpInterface::Train(const Eigen::VectorXd &x, const double y) {
  // Check input size.
  int input_dim = GetInputDim();
  assert(x.size() == input_dim);

  double *x_array;
  Eigen::Map<Eigen::VectorXd>(x_array, x.size());
  gp_->add_pattern(&x_array[0], y);
}

double LibgpInterface::Predict(const std::vector<double> &x) {
  assert(x.size() == GetInputDim());
  return gp_->f(&x[0]);
}

double LibgpInterface::Predict(const Eigen::VectorXd &x) {
  assert(x.size() == GetInputDim());

  double *x_array;
  Eigen::Map<Eigen::MatrixXd>(x_array, x.rows(), x.cols());

  return gp_->f(&x[0]);
}

double LibgpInterface::Predict(const std::vector<double> &x, double *y_var) {
  assert(x.size() == GetInputDim());
  (*y_var) = gp_->var(&x[0]);
  return gp_->f(&x[0]);
}

double LibgpInterface::Predict(const Eigen::VectorXd &x, double *y_var) {
  assert(x.size() == GetInputDim());

  double *x_array;
  Eigen::Map<Eigen::MatrixXd>(x_array, x.rows(), x.cols());

  (*y_var) = gp_->var(&x_array[0]);
  return gp_->f(&x_array[0]);
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

void LibgpInterface::Predict(int num_samples, const Eigen::MatrixXd &x,
                             Eigen::VectorXd *y_pred) {
  // Check for input size.
  int input_dim = GetInputDim();
  assert(x.size() == num_samples * input_dim);

  double *x_array;
  Eigen::Map<Eigen::MatrixXd>(x_array, x.rows(), x.cols());
  // Generate predictions.
  y_pred->resize(num_samples);
  for (int i = 0; i < num_samples; i++) {
    (*y_pred)(i) = gp_->f(&x_array[i * input_dim]);
  }
}

void LibgpInterface::Predict(int num_samples, const Eigen::MatrixXd &x,
                             Eigen::VectorXd *y_pred,
                             Eigen::VectorXd *y_var) {
  // Check for input size.
  int input_dim = GetInputDim();
  assert(x.size() == num_samples * input_dim);

  double *x_array;
  Eigen::Map<Eigen::MatrixXd>(x_array, x.rows(), x.cols());

  // Generate predictions.
  y_pred->resize(num_samples);
  y_var->resize(num_samples);
  for (int i = 0; i < num_samples; i++) {
    (*y_pred)(i) = gp_->f(&x_array[i * input_dim]);
    (*y_var)(i) = gp_->var(&x_array[i * input_dim]);
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
