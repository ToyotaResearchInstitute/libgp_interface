#pragma once

#include <memory>
#include <string>
#include <vector>

#include <gp.h>

// A simple interface for training and generating predictions using mblum's libgp.
class LibgpInterface {
public:
  LibgpInterface();
  ~LibgpInterface();

  // Initialize the GP with input dimension size, kernel type and
  // hyperparameters.
  void Initialize(unsigned int dim, const std::string &cov_kernel,
                  const std::vector<double> &hyp_params);

  // Train the GP with a set of training pairs (x_i, y_i).
  void Train(const std::vector<double> &x, const std::vector<double> &y);

  // Generate predictions given test points x, and outputs predictions y_pred.
  void Predict(int num_samples, const std::vector<double> &x,
               std::vector<double> *y_pred);

  // Generate predictions with variance given test points x, and outputs
  // predictions y_pred and y_var.
  void Predict(int num_samples, const std::vector<double> &x,
               std::vector<double> *y_pred, std::vector<double> *y_var);

  // Returns the dimension of the input vector.
  int GetInputDim();

  // Returns the number of samples used to train the GP.
  int GetNumSamples();

  // Clears all training data.
  void Clear();

private:
  // The gp object.
  std::shared_ptr<libgp::GaussianProcess> gp_;
};
