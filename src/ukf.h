#ifndef UKF_H
#define UKF_H

#include "Eigen/Dense"
#include "measurement_package.h"

class UKF {
 public:
  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Subprocess of Predicts sigma points, project sigma points to the next time step
   * using process model.
   * @param Xsig_aug_  Augmented Sigma points matrix at time k, size (n_aug_, 2*n_aug_+1)
   * @param delta_t Time between k and k+1 in s
   */
  void PredictSigmaPoints(const Eigen::MatrixXd& Xsig_aug_, double delta_t_);

  /**
   * After project Xsig_aug_ from time k to k+1, resulting Xsig_pred_, we now
   * need to calculate its mean and covariance
  */
  void CalPredictMeanCov(void);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);


  // initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  // if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  // if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // predicted sigma points matrix
  // Update in PredictSigmaPoints method
  Eigen::MatrixXd Xsig_pred_;
 
  // Measurement noise covariance, for Radar
  Eigen::MatrixXd R_rad_;

  // Measurement noise covariance, for Lidar
  Eigen::MatrixXd R_las_;

  //  Measurement transformation matrix, for Lidar measurement
  Eigen::MatrixXd H_;

  // time when the state is true, in us
  long long time_us_;
  
  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  // Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  // Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  // Radar measurement noise standard deviation radius in m
  double std_radr_;

  // Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  // Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  // Weights of sigma points
  Eigen::VectorXd weights_;

  // State dimension
  int n_x_;

  // Augmented state dimension
  int n_aug_;

  // Sigma point spreading parameter
  double lambda_;

  // Radar Measurement dimension [rho(m), phi(rad), rho_dot(m/s)]
  int n_zrad_;

  // Lidar Measurement dimension [px(m), py(m)]
  int n_zlas_;

};

#endif  // UKF_H