#include <iostream>

#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.25;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values
   */

  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  is_initialized_ = false; // To be set when the first ProcessMeasurement called

  // State dimension
  n_x_ = 5;

  // Predicted state vector
  x_pred_ = VectorXd(n_x_);
  x_pred_.fill(0.0);

  // Augmented state dimension, include Process noise for linear and angular acceleration
  n_aug_ = n_x_ + 2;

  // Design parameters lambda
  lambda_ = 3 - n_aug_;

  // dimension of radar measurement
  n_zrad_ = 3;

  // dimension of radar measurement
  n_zlas_ = 2;

  // Initialize state covariance matrix, taken from class assignment
  P_ << 0.5, 0, 0, 0, 0,
        0, 0.5, 0, 0, 0,
        0, 0, 0.5, 0, 0,
        0, 0, 0, 0.2, 0,
        0, 0, 0, 0, 0.2;
  
  // Predicted state covariance matrix
  P_pred_ = MatrixXd(n_x_, n_x_);
  P_pred_.fill(0.0);

  // Measurement noise covariance matrix, Radar
  R_rad_ = MatrixXd(n_zrad_, n_zrad_);
  R_rad_ << (std_radr_*std_radr_), 0, 0,
            0, (std_radphi_*std_radphi_), 0,
            0, 0, (std_radrd_*std_radrd_);

  // Measurement noise covariance matrix, Lidar
  R_las_ = MatrixXd(n_zlas_, n_zlas_);
  R_las_ << (std_laspx_*std_laspx_), 0,
            0, (std_laspy_*std_laspy_);

  // Predicted Sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred_.fill(0.0);

  // Weights for calculate Sigma points mean and covariancee
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (auto i = 1; i < (2 * n_aug_ + 1); i++)
  {
    weights_(i) = 1 / (2 * (lambda_ + n_aug_));
  }

  // Measurement transformation matrix, from state vector and measurement space
  H_ = MatrixXd(n_zlas_, n_x_);
  H_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
  /**
   * Check if Kalman filter has been initialized. If no, then initialize state
   * and covariance matrix according to source of input measurement (Lidar/Radar)
   */
  if (!is_initialized_)
  {
    std::cout << "Kalman Filter Initialization" << std::endl;
    // intialize Kalman filter
    if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      this->x_ << meas_package.raw_measurements_[0], // initial position x
          meas_package.raw_measurements_[1],         // initial position y
          0,                                         // initial velocity, v
          0,                                         // initial angular position
          0;                                         // initial angular accerelation
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      // Need to calculate initial state from Radar measurement
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      double rho_d = meas_package.raw_measurements_[2];
      this->x_ << rho * cos(phi),                   // initial position x
          rho * sin(phi),                           // initial position y
          0,                                        // initial velocity, v (unknow)
          phi,                                      // initial angular position
          0;                                        // initial angular accerelation (unknow)
    }
    time_us_ = meas_package.timestamp_;             // timestamp of the first measurement
    is_initialized_ = true;
    return;
  }

  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0f; // convert uSec to Sec.
  time_us_ = meas_package.timestamp_;

  // // Do Prediction, use Sigma points to predict state
  Prediction(delta_t);

  // Measurement Update, depend on source of input measurement
  if (meas_package.sensor_type_ == MeasurementPackage::LASER)
  {
    UpdateLidar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
  {
    UpdateRadar(meas_package);
  }
}

// Output predicted state vector (x_pred_) and predicted state covariance matrix (P_pred_)
void UKF::Prediction(double delta_t_)
{
  /*
   * Create augmented state vector (x_aug) and augmented state covariance matrix (P_aug)
   */
  // Augment state vector x with process noise, std_a_ and std_yawdd_
  VectorXd x_aug = VectorXd(n_aug_); // n_aug_ = n_x_ + 2
  x_aug.head(5) = x_;
  x_aug(5) = 0.0;
  x_aug(6) = 0.0;

  // Augment covariance matrix with process noise covariance matrix
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_*std_a_;
  P_aug(6, 6) = std_yawdd_*std_yawdd_;

  // Calculate sqrt(P_aug) matrix
  MatrixXd L = P_aug.llt().matrixL();

  // Generate Sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.col(0) = x_aug;
  for (auto i = 0; i < n_aug_; i++)
  {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + n_aug_ + 1) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  // Predict Sigma points
  PredictSigmaPoints(Xsig_aug, delta_t_);

  // Calculate predicted Sigma points mean and covariance, x_pred_ and P_pred_
  CalPredictMeanCov();
}

void UKF::PredictSigmaPoints(const Eigen::MatrixXd &Xsig_aug_, double delta_t_)
{
  std::cout << "Predict Sigma Points" << std::endl;
  // Go through each column of Xsig_aug_ matrix
  for (auto i = 0; i < (2 * n_aug_ + 1); i++)
  {
    double px = Xsig_aug_(0, i);
    double py = Xsig_aug_(1, i);
    double v = Xsig_aug_(2, i);
    double yaw = Xsig_aug_(3, i);
    double yawd = Xsig_aug_(4, i);
    double nu_a = Xsig_aug_(5, i);
    double nu_yawdd = Xsig_aug_(6, i);

    double delta_t2_ = delta_t_*delta_t_;
    // Check if yawd = 0, avoid dividing with zero
    // State part
    if (fabs(yawd) > 1e-6)
    {
      this->Xsig_pred_(0, i) = px + ((v / yawd) * (sin(yaw + yawd * delta_t_) - sin(yaw)));
      this->Xsig_pred_(1, i) = py + ((v / yawd) * (-1 * cos(yaw + yawd * delta_t_) + cos(yaw)));
    }
    else
    {
      this->Xsig_pred_(0, i) = px + (v * cos(yaw) * delta_t_);
      this->Xsig_pred_(1, i) = py + (v * sin(yaw) * delta_t_);
    }
    this->Xsig_pred_(2, i) = v;
    this->Xsig_pred_(3, i) = yaw + yawd * delta_t_;
    this->Xsig_pred_(4, i) = yawd;

    // Noise part
    this->Xsig_pred_(0, i) += (delta_t2_ / 2) * cos(yaw) * nu_a;
    this->Xsig_pred_(1, i) += (delta_t2_ / 2) * sin(yaw) * nu_a;
    this->Xsig_pred_(2, i) += delta_t_ * nu_a;
    this->Xsig_pred_(3, i) += (delta_t2_ / 2) * nu_yawdd;
    this->Xsig_pred_(4, i) += delta_t_ * nu_yawdd;
  }
}

// Calcuate x_pred_ and P_pred_
void UKF::CalPredictMeanCov(void)
{
  std::cout << "Predict Mean & Covariance" << std::endl;
  // Calculate predicted state mean
  x_pred_.fill(0.0);
  for (auto i = 0; i < (2 * n_aug_ + 1); i++)
  {
    x_pred_ += weights_(i) * this->Xsig_pred_.col(i);
  }
  std::cout << "x_pred_" << std::endl;

  // Calculate predicted state covariance
  P_pred_.fill(0.0);
  for (auto i = 0; i < (2 * n_aug_ + 1); i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // Normalization, make Heading angle to be in between 0 to 2*M_PI
    while (x_diff(3) > M_PI) x_diff(3) -= 2*M_PI;
    while (x_diff(3) < -1*M_PI) x_diff(3) += 2*M_PI;

    P_pred_ += weights_(i) * x_diff * x_diff.transpose();
  }
  std::cout << "P_pred_" << std::endl;
}

void UKF::UpdateLidar(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Use lidar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  
  // Project predicted state mean (x_pred_) to measurement space (z_pred).
  VectorXd z_pred = H_*x_pred_;
  VectorXd z_actual = VectorXd(n_zlas_);
  z_actual << meas_package.raw_measurements_[0],
              meas_package.raw_measurements_[1];

  // Calculate error vector (y)
  VectorXd y = VectorXd(n_zlas_);
  y = z_actual - z_pred;

  // Calculate Predicted measurement covariance (S)
  MatrixXd Ht = H_.transpose();
  MatrixXd S = MatrixXd(n_zlas_, n_zlas_);
  S = ( H_ * P_pred_ * Ht ) + R_las_;

  // Calculate Kalman gain (K)
  MatrixXd K = P_pred_ * Ht * S.inverse();

  /**
   * Now, Update state (x_)
  */
  // Update State vector
  x_ = x_pred_ + ( K*y );

  // Update covariance matrix (P_)
  MatrixXd I = Eigen::MatrixXd::Identity(n_x_, n_x_);
  P_ = (I - K*H_) * P_pred_;
}

void UKF::UpdateRadar(MeasurementPackage meas_package)
{
  /**
   * Project predicted Sigma points (Xsig_pred_) to Measurement space (Zsig_pred).
   * Then calculate predicted measurement mean (z_pred), and predicted measurement covariance (S)
   */
  MatrixXd Zsig_pred = MatrixXd(n_zrad_, 2 * n_aug_ + 1);
  Zsig_pred.fill(0.0);

  // Project Sigma points to measurement space, get predicted measurement (Zsig_pred)
  for (auto i = 0; i < (2 * n_aug_ + 1); i++)
  {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    Zsig_pred(0, i) = sqrt((px*px) + (py*py));                                               // rho
    Zsig_pred(1, i) = atan2(py, px);                                                         // phi
    Zsig_pred(2, i) = ((px * cos(yaw) * v) + (py * sin(yaw) * v)) / sqrt((px*px) + (py*py)); // rho_d
  }

  // Calculate predicted measurement mean (z_pred)
  VectorXd z_pred = VectorXd(n_zrad_);
  z_pred.fill(0.0);
  for (auto i = 0; i < (2 * n_aug_ + 1); i++)
  {
    z_pred += weights_(i) * Zsig_pred.col(i);
  }

  // Claculate predicted measurement covariance (S)
  MatrixXd S = MatrixXd(n_zrad_, n_zrad_);
  S.fill(0.0);
  for (auto i = 0; i < (2 * n_aug_ + 1); i++)
  {
    VectorXd z_diff = VectorXd(n_zrad_);
    z_diff = Zsig_pred.col(i) - z_pred;

    // angle normalization, make angle to be in between 0 to 2*M_PI
    while (z_diff(1) > M_PI) z_diff(1) -= 2*M_PI;
    while (z_diff(1) < -1*M_PI) z_diff(1) += 2*M_PI;

    S += weights_(i) * z_diff * z_diff.transpose();
  }
  // Add measurement noise covariance matrix
  S += R_rad_;

  // UKF Update, Calculate Cross-Correlation matrix (T) and Kalman gain (K)
  // Calculate Cross-Correlation matrix (T)
  MatrixXd T = MatrixXd(n_x_, n_zrad_);
  T.fill(0.0);
  for (auto i = 0; i < (2 * n_aug_ + 1); i++)
  {
    VectorXd x_diff = VectorXd(n_x_);
    x_diff = Xsig_pred_.col(i) - x_pred_;
    // // Do angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2*M_PI;
    while (x_diff(3) < -1*M_PI) x_diff(3) += 2*M_PI;

    VectorXd z_diff = VectorXd(n_zrad_);
    z_diff = Zsig_pred.col(i) - z_pred;
    // angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2*M_PI;
    while (z_diff(1) < -1*M_PI) z_diff(1) += 2*M_PI;

    T += weights_(i) * x_diff * z_diff.transpose();
  }

  // Calculate Kalman gain (K)
  MatrixXd K = T * S.inverse();

  /**
   * Now, State Update. Using actual measurement to adjust predicted state (x_pred) to
   * actual state (x_) and predicted state covariance (P_pred_) to actual covariance (P_)
   */
  VectorXd z_actual = VectorXd(n_zrad_);
  z_actual << meas_package.raw_measurements_(0),
              meas_package.raw_measurements_(1),
              meas_package.raw_measurements_(2);
  VectorXd z_diff = z_actual - z_pred;
  // angle normalization
  while (z_diff(1) > M_PI) z_diff(1) -= 2*M_PI;
  while (z_diff(1) < -1*M_PI) z_diff(1) += 2*M_PI;

  // Update State vector (x_)
  x_ = x_pred_ + K*(z_diff);

  // Update State Covariance matrix (P_)
  P_ = P_pred_ - (K * S * K.transpose());
}