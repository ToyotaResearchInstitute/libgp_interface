// Copyright 2018 Toyota Research Institute.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gp.h>

// A simple interface for training and generating predictions using mblum's
// libgp.
class LibgpInterface {
public:
  LibgpInterface();
  ~LibgpInterface();

  // Initialize the GP with input dimension size, kernel type and
  // hyperparameters.
  void Initialize(unsigned int dim, const std::string &cov_kernel,
                  const std::vector<double> &hyp_params);

  // Batch train the GP with a set of training pairs (x_i, y_i).
  // Inputs:
  //  x:  a vector/Eigen matrix that is of size MxN, where M is the input dim
  //      and N is the number of training examples. The vector is appended
  //      rowwise.
  //  y:  a vector of size N.
  void Train(const std::vector<double> &x, const std::vector<double> &y);

  void Train(const Eigen::MatrixXf &x, const Eigen::VectorXf &y);

  // Train the GP with a single training pair (x, y).
  // Inputs:
  //  x:  a vector that is of size M, the input dim.
  //  y:  a double.
  void Train(const std::vector<double> &x, const double y);

  void Train(const Eigen::VectorXd &x, const double y);

  // Generate a single prediction with variance given test point x, and outputs
  // the prediction `y_pred`.
  // Inputs:
  //  x:  a vector of size M,the input dimension.
  // Returns:
  //  y_pred:   a double.
  double Predict(const std::vector<double> &x);

  double Predict(const Eigen::VectorXd &x);

  // Generate a single prediction with variance given test point x, and outputs
  // a tuple with (y_pred, y_var).
  // Inputs:
  //  x:  a vector of size M,the input dimension.
  // Outputs:
  //  returns a double `y_pred`, and populates the output parameter `y_var`.
  double Predict(const std::vector<double> &x, double *y_var);

  double Predict(const Eigen::VectorXd &x, double *y_var);

  // Batch generate predictions given test points x, and outputs predictions
  // `y_pred`.
  // Inputs:
  //  num_samples: Number of samples, N.
  //  x:  a vector/Eigen matrix that is of size MxN, where M is the input dim
  //      and N is the number of test samples. The vector is appended rowwise.
  // Outputs:
  //  y_pred:  a vector of size N.
  void Predict(int num_samples, const std::vector<double> &x,
               std::vector<double> *y_pred);

  void Predict(int num_samples, const Eigen::MatrixXd &x,
               Eigen::VectorXd *y_pred);

  // Batch generate predictions with variance given test points x, and outputs
  // predictions `y_pred` with variance `y_var`.
  // Inputs:
  //  num_samples: Number of samples, N.
  //  x:  a vector/Eigen matrix that is of size MxN, where M is the input dim
  //      and N is the number of test samples. The vector is appended rowwise.
  // Outputs:
  //  y_pred:  a vector of size N.
  //  y_var:   a vector of size N.
  void Predict(int num_samples, const std::vector<double> &x,
               std::vector<double> *y_pred, std::vector<double> *y_var);

  void Predict(int num_samples, const Eigen::MatrixXd &x,
               Eigen::VectorXd *y_pred, Eigen::VectorXd *y_var);

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
