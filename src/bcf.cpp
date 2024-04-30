#include <cpp11.hpp>
#include "stochtree_types.h"
#include <stochtree/cpp_api.h>
#include <stochtree/leaf_model.h>
#include <functional>
#include <memory>
#include <variant>
#include <vector>

[[cpp11::register]]
cpp11::external_pointer<StochTree::BCFModelWrapper> bcf_init_cpp(bool univariate_treatment = true) {
    std::unique_ptr<StochTree::BCFModelWrapper> bcf_ptr = std::make_unique<StochTree::BCFModelWrapper>(univariate_treatment);
    return cpp11::external_pointer<StochTree::BCFModelWrapper>(bcf_ptr.release());
}

[[cpp11::register]]
void bcf_add_train_with_weights_cpp(
        cpp11::external_pointer<StochTree::BCFModelWrapper> bcf_wrapper, cpp11::doubles_matrix<> X_train_mu, 
        cpp11::doubles_matrix<> X_train_tau, cpp11::doubles_matrix<> Z_train, 
        cpp11::doubles y_train, cpp11::doubles weights_train, bool treatment_binary
) {
    // Data dimensions
    int n = X_train_mu.nrow();
    int X_train_mu_cols = X_train_mu.ncol();
    int X_train_tau_cols = X_train_tau.ncol();
    int Z_train_cols = Z_train.ncol();
    
    // Pointers to R data
    double* X_train_mu_data_ptr = REAL(PROTECT(X_train_mu));
    double* X_train_tau_data_ptr = REAL(PROTECT(X_train_tau));
    double* Z_train_data_ptr = REAL(PROTECT(Z_train));
    double* y_train_data_ptr = REAL(PROTECT(y_train));
    double* weights_train_data_ptr = REAL(PROTECT(weights_train));
    
    // Load training data into BCF model
    bcf_wrapper->LoadTrain(
            y_train_data_ptr, n, X_train_mu_data_ptr, X_train_mu_cols, 
            X_train_tau_data_ptr, X_train_tau_cols, Z_train_data_ptr, 
            Z_train_cols, treatment_binary, weights_train_data_ptr
    );

    // UNPROTECT the SEXPs created to point to the R data
    UNPROTECT(5);
}

[[cpp11::register]]
void bcf_add_train_no_weights_cpp(
        cpp11::external_pointer<StochTree::BCFModelWrapper> bcf_wrapper, cpp11::doubles_matrix<> X_train_mu, 
        cpp11::doubles_matrix<> X_train_tau, cpp11::doubles_matrix<> Z_train, 
        cpp11::doubles y_train, bool treatment_binary
) {
    // Data dimensions
    int n = X_train_mu.nrow();
    int X_train_mu_cols = X_train_mu.ncol();
    int X_train_tau_cols = X_train_tau.ncol();
    int Z_train_cols = Z_train.ncol();
    
    // Pointers to R data
    double* X_train_mu_data_ptr = REAL(PROTECT(X_train_mu));
    double* X_train_tau_data_ptr = REAL(PROTECT(X_train_tau));
    double* Z_train_data_ptr = REAL(PROTECT(Z_train));
    double* y_train_data_ptr = REAL(PROTECT(y_train));
    
    // Load training data into BCF model
    bcf_wrapper->LoadTrain(
            y_train_data_ptr, n, X_train_mu_data_ptr, X_train_mu_cols, 
            X_train_tau_data_ptr, X_train_tau_cols, Z_train_data_ptr, 
            Z_train_cols, treatment_binary
    );
        
    // UNPROTECT the SEXPs created to point to the R data
    UNPROTECT(4);
}

[[cpp11::register]]
void bcf_add_test_cpp(
        cpp11::external_pointer<StochTree::BCFModelWrapper> bcf_wrapper, cpp11::doubles_matrix<> X_test_mu, 
        cpp11::doubles_matrix<> X_test_tau, cpp11::doubles_matrix<> Z_test
) {
    // Data dimensions
    int n = X_test_mu.nrow();
    int X_test_mu_cols = X_test_mu.ncol();
    int X_test_tau_cols = X_test_tau.ncol();
    int Z_test_cols = Z_test.ncol();
    
    // Pointers to R data
    double* X_test_mu_data_ptr = REAL(PROTECT(X_test_mu));
    double* X_test_tau_data_ptr = REAL(PROTECT(X_test_tau));
    double* Z_test_data_ptr = REAL(PROTECT(Z_test));

    // Load test data into BCF model
    bcf_wrapper->LoadTest(
            X_test_mu_data_ptr, n, X_test_mu_cols, 
            X_test_tau_data_ptr, X_test_tau_cols, 
            Z_test_data_ptr, Z_test_cols
    );
    
    // UNPROTECT the SEXPs created to point to the R data
    UNPROTECT(3);
}

[[cpp11::register]]
void bcf_reset_global_var_samples_cpp(
        cpp11::external_pointer<StochTree::BCFModelWrapper> bcf_wrapper, 
        cpp11::doubles data_vector
) {
    // Data dimensions
    int n = data_vector.size();

    // Pointer to R data
    double* data_ptr = REAL(PROTECT(data_vector));

    // Map Eigen array to data in the R container
    bcf_wrapper->ResetGlobalVarSamples(data_ptr, n);
    
    // UNPROTECT the SEXP created to point to the R data
    UNPROTECT(1);
}

[[cpp11::register]]
void bcf_reset_prognostic_leaf_var_samples_cpp(
        cpp11::external_pointer<StochTree::BCFModelWrapper> bcf_wrapper, 
        cpp11::doubles data_vector
) {
    // Data dimensions
    int n = data_vector.size();
    
    // Pointer to R data
    double* data_ptr = REAL(PROTECT(data_vector));
    
    // Map Eigen array to data in the R container
    bcf_wrapper->ResetPrognosticLeafVarSamples(data_ptr, n);
    
    // UNPROTECT the SEXP created to point to the R data
    UNPROTECT(1);
}

[[cpp11::register]]
void bcf_reset_treatment_leaf_var_samples_cpp(
        cpp11::external_pointer<StochTree::BCFModelWrapper> bcf_wrapper, 
        cpp11::doubles data_vector
) {
    // Data dimensions
    int n = data_vector.size();
    
    // Pointer to R data
    double* data_ptr = REAL(PROTECT(data_vector));
    
    // Map Eigen array to data in the R container
    bcf_wrapper->ResetTreatmentLeafVarSamples(data_ptr, n);
    
    // UNPROTECT the SEXP created to point to the R data
    UNPROTECT(1);
}

[[cpp11::register]]
void bcf_reset_treatment_coding_samples_cpp(
        cpp11::external_pointer<StochTree::BCFModelWrapper> bcf_wrapper, 
        cpp11::doubles data_vector
) {
    // Data dimensions
    int n = data_vector.size();
    
    // Pointer to R data
    double* data_ptr = REAL(PROTECT(data_vector));
    
    // Map Eigen array to data in the R container
    bcf_wrapper->ResetTreatedCodingSamples(data_ptr, n);
    
    // UNPROTECT the SEXP created to point to the R data
    UNPROTECT(1);
}

[[cpp11::register]]
void bcf_reset_control_coding_samples_cpp(
        cpp11::external_pointer<StochTree::BCFModelWrapper> bcf_wrapper, 
        cpp11::doubles data_vector
) {
    // Data dimensions
    int n = data_vector.size();
    
    // Pointer to R data
    double* data_ptr = REAL(PROTECT(data_vector));
    
    // Map Eigen array to data in the R container
    bcf_wrapper->ResetControlCodingSamples(data_ptr, n);
    
    // UNPROTECT the SEXP created to point to the R data
    UNPROTECT(1);
}

[[cpp11::register]]
void bcf_reset_train_prediction_samples_cpp(
        cpp11::external_pointer<StochTree::BCFModelWrapper> bcf_wrapper, 
        cpp11::doubles_matrix<> muhat, cpp11::doubles tauhat, cpp11::doubles_matrix<> yhat, 
        int num_obs, int num_samples, int treatment_dim
) {
    // Pointers to R data
    double* muhat_data_ptr = REAL(PROTECT(muhat));
    double* tauhat_data_ptr = REAL(PROTECT(tauhat));
    double* yhat_data_ptr = REAL(PROTECT(yhat));
    
    // Map Eigen array to data in the R container
    bcf_wrapper->ResetTrainPredictionSamples(muhat_data_ptr, tauhat_data_ptr, yhat_data_ptr, num_obs, num_samples, treatment_dim);
    
    // UNPROTECT the SEXP created to point to the R data
    UNPROTECT(3);
}

[[cpp11::register]]
void bcf_reset_test_prediction_samples_cpp(
        cpp11::external_pointer<StochTree::BCFModelWrapper> bcf_wrapper, 
        cpp11::doubles_matrix<> muhat, cpp11::doubles tauhat, cpp11::doubles_matrix<> yhat, 
        int num_obs, int num_samples, int treatment_dim
) {
    // Pointers to R data
    double* muhat_data_ptr = REAL(PROTECT(muhat));
    double* tauhat_data_ptr = REAL(PROTECT(tauhat));
    double* yhat_data_ptr = REAL(PROTECT(yhat));
    
    // Map Eigen array to data in the R container
    bcf_wrapper->ResetTestPredictionSamples(muhat_data_ptr, tauhat_data_ptr, yhat_data_ptr, num_obs, num_samples, treatment_dim);
    
    // UNPROTECT the SEXP created to point to the R data
    UNPROTECT(3);
}

[[cpp11::register]]
void sample_bcf_univariate_cpp(
        cpp11::external_pointer<StochTree::BCFModelWrapper> bcf_wrapper, 
        cpp11::external_pointer<StochTree::ForestContainer> forest_samples_mu, 
        cpp11::external_pointer<StochTree::ForestContainer> forest_samples_tau, 
        cpp11::external_pointer<std::mt19937> rng, 
        int cutpoint_grid_size, double sigma_leaf_mu, double sigma_leaf_tau, 
        double alpha_mu, double alpha_tau, double beta_mu, double beta_tau, 
        int min_samples_leaf_mu, int min_samples_leaf_tau, double nu, double lamb, 
        double a_leaf_mu, double a_leaf_tau, double b_leaf_mu, double b_leaf_tau, 
        double sigma2, int num_trees_mu, int num_trees_tau, double b1, double b0, 
        cpp11::integers feature_types_mu_int, cpp11::integers feature_types_tau_int, 
        int num_gfr, int num_burnin, int num_mcmc, double leaf_init_mu, double leaf_init_tau
) {
    // Convert feature_types
    std::vector<StochTree::FeatureType> feature_types_mu(feature_types_mu_int.size());
    for (int i = 0; i < feature_types_mu_int.size(); i++) {
        feature_types_mu.at(i) = static_cast<StochTree::FeatureType>(feature_types_mu_int.at(i));
    }
    std::vector<StochTree::FeatureType> feature_types_tau(feature_types_tau_int.size());
    for (int i = 0; i < feature_types_tau_int.size(); i++) {
        feature_types_tau.at(i) = static_cast<StochTree::FeatureType>(feature_types_tau_int.at(i));
    }
    
    // Run the sampler
    bcf_wrapper->SampleBCF(forest_samples_mu.get(), forest_samples_tau.get(), rng.get(), 
                           cutpoint_grid_size, sigma_leaf_mu, sigma_leaf_tau, alpha_mu, alpha_tau, 
                           beta_mu, beta_tau, min_samples_leaf_mu, min_samples_leaf_tau, nu, lamb, 
                           a_leaf_mu, a_leaf_tau, b_leaf_mu, b_leaf_tau, sigma2, num_trees_mu, num_trees_tau, 
                           b1, b0, feature_types_mu, feature_types_tau, num_gfr, num_burnin, num_mcmc, 
                           leaf_init_mu, leaf_init_tau);
}

[[cpp11::register]]
void sample_bcf_multivariate_cpp(
        cpp11::external_pointer<StochTree::BCFModelWrapper> bcf_wrapper, 
        cpp11::external_pointer<StochTree::ForestContainer> forest_samples_mu, 
        cpp11::external_pointer<StochTree::ForestContainer> forest_samples_tau, 
        cpp11::external_pointer<std::mt19937> rng, 
        int cutpoint_grid_size, double sigma_leaf_mu, cpp11::doubles_matrix<> sigma_leaf_tau_r, 
        double alpha_mu, double alpha_tau, double beta_mu, double beta_tau, 
        int min_samples_leaf_mu, int min_samples_leaf_tau, double nu, double lamb, 
        double a_leaf_mu, double a_leaf_tau, double b_leaf_mu, double b_leaf_tau, 
        double sigma2, int num_trees_mu, int num_trees_tau, double b1, double b0, 
        cpp11::integers feature_types_mu_int, cpp11::integers feature_types_tau_int, 
        int num_gfr, int num_burnin, int num_mcmc, double leaf_init_mu, double leaf_init_tau
) {
    // Convert feature_types
    std::vector<StochTree::FeatureType> feature_types_mu(feature_types_mu_int.size());
    for (int i = 0; i < feature_types_mu_int.size(); i++) {
        feature_types_mu.at(i) = static_cast<StochTree::FeatureType>(feature_types_mu_int.at(i));
    }
    std::vector<StochTree::FeatureType> feature_types_tau(feature_types_tau_int.size());
    for (int i = 0; i < feature_types_tau_int.size(); i++) {
        feature_types_tau.at(i) = static_cast<StochTree::FeatureType>(feature_types_tau_int.at(i));
    }
    
    // Convert sigma_leaf_tau
    Eigen::MatrixXd sigma_leaf_tau;
    int num_row = sigma_leaf_tau_r.nrow();
    int num_col = sigma_leaf_tau_r.ncol();
    sigma_leaf_tau.resize(num_row, num_col);
    for (int i = 0; i < num_row; i++) {
        for (int j = 0; j < num_col; j++) {
            sigma_leaf_tau(i,j) = sigma_leaf_tau_r(i,j);
        }
    }
    
    // Run the sampler
    bcf_wrapper->SampleBCF(forest_samples_mu.get(), forest_samples_tau.get(), rng.get(), 
                           cutpoint_grid_size, sigma_leaf_mu, sigma_leaf_tau, alpha_mu, alpha_tau, 
                           beta_mu, beta_tau, min_samples_leaf_mu, min_samples_leaf_tau, nu, lamb, 
                           a_leaf_mu, a_leaf_tau, b_leaf_mu, b_leaf_tau, sigma2, num_trees_mu, num_trees_tau, 
                           b1, b0, feature_types_mu, feature_types_tau, num_gfr, num_burnin, num_mcmc, 
                           leaf_init_mu, leaf_init_tau);
}
