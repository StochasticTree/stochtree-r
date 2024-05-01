// Generated by cpp11: do not edit by hand
// clang-format off

#include "stochtree_types.h"
#include "cpp11/declarations.hpp"
#include <R_ext/Visibility.h>

// data.cpp
cpp11::external_pointer<StochTree::ForestDataset> create_forest_dataset_cpp();
extern "C" SEXP _stochtree_create_forest_dataset_cpp() {
  BEGIN_CPP11
    return cpp11::as_sexp(create_forest_dataset_cpp());
  END_CPP11
}
// data.cpp
int dataset_num_rows_cpp(cpp11::external_pointer<StochTree::ForestDataset> dataset);
extern "C" SEXP _stochtree_dataset_num_rows_cpp(SEXP dataset) {
  BEGIN_CPP11
    return cpp11::as_sexp(dataset_num_rows_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestDataset>>>(dataset)));
  END_CPP11
}
// data.cpp
int dataset_num_covariates_cpp(cpp11::external_pointer<StochTree::ForestDataset> dataset);
extern "C" SEXP _stochtree_dataset_num_covariates_cpp(SEXP dataset) {
  BEGIN_CPP11
    return cpp11::as_sexp(dataset_num_covariates_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestDataset>>>(dataset)));
  END_CPP11
}
// data.cpp
int dataset_num_basis_cpp(cpp11::external_pointer<StochTree::ForestDataset> dataset);
extern "C" SEXP _stochtree_dataset_num_basis_cpp(SEXP dataset) {
  BEGIN_CPP11
    return cpp11::as_sexp(dataset_num_basis_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestDataset>>>(dataset)));
  END_CPP11
}
// data.cpp
bool dataset_has_basis_cpp(cpp11::external_pointer<StochTree::ForestDataset> dataset);
extern "C" SEXP _stochtree_dataset_has_basis_cpp(SEXP dataset) {
  BEGIN_CPP11
    return cpp11::as_sexp(dataset_has_basis_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestDataset>>>(dataset)));
  END_CPP11
}
// data.cpp
bool dataset_has_variance_weights_cpp(cpp11::external_pointer<StochTree::ForestDataset> dataset);
extern "C" SEXP _stochtree_dataset_has_variance_weights_cpp(SEXP dataset) {
  BEGIN_CPP11
    return cpp11::as_sexp(dataset_has_variance_weights_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestDataset>>>(dataset)));
  END_CPP11
}
// data.cpp
void forest_dataset_add_covariates_cpp(cpp11::external_pointer<StochTree::ForestDataset> dataset_ptr, cpp11::doubles_matrix<> covariates);
extern "C" SEXP _stochtree_forest_dataset_add_covariates_cpp(SEXP dataset_ptr, SEXP covariates) {
  BEGIN_CPP11
    forest_dataset_add_covariates_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestDataset>>>(dataset_ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles_matrix<>>>(covariates));
    return R_NilValue;
  END_CPP11
}
// data.cpp
void forest_dataset_add_basis_cpp(cpp11::external_pointer<StochTree::ForestDataset> dataset_ptr, cpp11::doubles_matrix<> basis);
extern "C" SEXP _stochtree_forest_dataset_add_basis_cpp(SEXP dataset_ptr, SEXP basis) {
  BEGIN_CPP11
    forest_dataset_add_basis_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestDataset>>>(dataset_ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles_matrix<>>>(basis));
    return R_NilValue;
  END_CPP11
}
// data.cpp
void forest_dataset_update_basis_cpp(cpp11::external_pointer<StochTree::ForestDataset> dataset_ptr, cpp11::doubles_matrix<> basis);
extern "C" SEXP _stochtree_forest_dataset_update_basis_cpp(SEXP dataset_ptr, SEXP basis) {
  BEGIN_CPP11
    forest_dataset_update_basis_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestDataset>>>(dataset_ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles_matrix<>>>(basis));
    return R_NilValue;
  END_CPP11
}
// data.cpp
void forest_dataset_add_weights_cpp(cpp11::external_pointer<StochTree::ForestDataset> dataset_ptr, cpp11::doubles weights);
extern "C" SEXP _stochtree_forest_dataset_add_weights_cpp(SEXP dataset_ptr, SEXP weights) {
  BEGIN_CPP11
    forest_dataset_add_weights_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestDataset>>>(dataset_ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(weights));
    return R_NilValue;
  END_CPP11
}
// data.cpp
cpp11::external_pointer<StochTree::ColumnVector> create_column_vector_cpp(cpp11::doubles outcome);
extern "C" SEXP _stochtree_create_column_vector_cpp(SEXP outcome) {
  BEGIN_CPP11
    return cpp11::as_sexp(create_column_vector_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(outcome)));
  END_CPP11
}
// data.cpp
cpp11::external_pointer<StochTree::RandomEffectsDataset> create_rfx_dataset_cpp();
extern "C" SEXP _stochtree_create_rfx_dataset_cpp() {
  BEGIN_CPP11
    return cpp11::as_sexp(create_rfx_dataset_cpp());
  END_CPP11
}
// data.cpp
int rfx_dataset_num_rows_cpp(cpp11::external_pointer<StochTree::RandomEffectsDataset> dataset);
extern "C" SEXP _stochtree_rfx_dataset_num_rows_cpp(SEXP dataset) {
  BEGIN_CPP11
    return cpp11::as_sexp(rfx_dataset_num_rows_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::RandomEffectsDataset>>>(dataset)));
  END_CPP11
}
// data.cpp
bool rfx_dataset_has_group_labels_cpp(cpp11::external_pointer<StochTree::RandomEffectsDataset> dataset);
extern "C" SEXP _stochtree_rfx_dataset_has_group_labels_cpp(SEXP dataset) {
  BEGIN_CPP11
    return cpp11::as_sexp(rfx_dataset_has_group_labels_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::RandomEffectsDataset>>>(dataset)));
  END_CPP11
}
// data.cpp
bool rfx_dataset_has_basis_cpp(cpp11::external_pointer<StochTree::RandomEffectsDataset> dataset);
extern "C" SEXP _stochtree_rfx_dataset_has_basis_cpp(SEXP dataset) {
  BEGIN_CPP11
    return cpp11::as_sexp(rfx_dataset_has_basis_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::RandomEffectsDataset>>>(dataset)));
  END_CPP11
}
// data.cpp
bool rfx_dataset_has_variance_weights_cpp(cpp11::external_pointer<StochTree::RandomEffectsDataset> dataset);
extern "C" SEXP _stochtree_rfx_dataset_has_variance_weights_cpp(SEXP dataset) {
  BEGIN_CPP11
    return cpp11::as_sexp(rfx_dataset_has_variance_weights_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::RandomEffectsDataset>>>(dataset)));
  END_CPP11
}
// data.cpp
void rfx_dataset_add_group_labels_cpp(cpp11::external_pointer<StochTree::RandomEffectsDataset> dataset_ptr, cpp11::integers group_labels);
extern "C" SEXP _stochtree_rfx_dataset_add_group_labels_cpp(SEXP dataset_ptr, SEXP group_labels) {
  BEGIN_CPP11
    rfx_dataset_add_group_labels_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::RandomEffectsDataset>>>(dataset_ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::integers>>(group_labels));
    return R_NilValue;
  END_CPP11
}
// data.cpp
void rfx_dataset_add_basis_cpp(cpp11::external_pointer<StochTree::RandomEffectsDataset> dataset_ptr, cpp11::doubles_matrix<> basis);
extern "C" SEXP _stochtree_rfx_dataset_add_basis_cpp(SEXP dataset_ptr, SEXP basis) {
  BEGIN_CPP11
    rfx_dataset_add_basis_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::RandomEffectsDataset>>>(dataset_ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles_matrix<>>>(basis));
    return R_NilValue;
  END_CPP11
}
// data.cpp
void rfx_dataset_add_weights_cpp(cpp11::external_pointer<StochTree::RandomEffectsDataset> dataset_ptr, cpp11::doubles weights);
extern "C" SEXP _stochtree_rfx_dataset_add_weights_cpp(SEXP dataset_ptr, SEXP weights) {
  BEGIN_CPP11
    rfx_dataset_add_weights_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::RandomEffectsDataset>>>(dataset_ptr), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(weights));
    return R_NilValue;
  END_CPP11
}
// predictor.cpp
cpp11::writable::doubles_matrix<> predict_forest_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, cpp11::external_pointer<StochTree::ForestDataset> dataset);
extern "C" SEXP _stochtree_predict_forest_cpp(SEXP forest_samples, SEXP dataset) {
  BEGIN_CPP11
    return cpp11::as_sexp(predict_forest_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestContainer>>>(forest_samples), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestDataset>>>(dataset)));
  END_CPP11
}
// predictor.cpp
cpp11::writable::doubles_matrix<> predict_forest_raw_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, cpp11::external_pointer<StochTree::ForestDataset> dataset);
extern "C" SEXP _stochtree_predict_forest_raw_cpp(SEXP forest_samples, SEXP dataset) {
  BEGIN_CPP11
    return cpp11::as_sexp(predict_forest_raw_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestContainer>>>(forest_samples), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestDataset>>>(dataset)));
  END_CPP11
}
// predictor.cpp
cpp11::writable::doubles_matrix<> predict_forest_raw_single_forest_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, cpp11::external_pointer<StochTree::ForestDataset> dataset, int forest_num);
extern "C" SEXP _stochtree_predict_forest_raw_single_forest_cpp(SEXP forest_samples, SEXP dataset, SEXP forest_num) {
  BEGIN_CPP11
    return cpp11::as_sexp(predict_forest_raw_single_forest_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestContainer>>>(forest_samples), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestDataset>>>(dataset), cpp11::as_cpp<cpp11::decay_t<int>>(forest_num)));
  END_CPP11
}
// random_effects.cpp
cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container_cpp(int num_components, int num_groups);
extern "C" SEXP _stochtree_rfx_container_cpp(SEXP num_components, SEXP num_groups) {
  BEGIN_CPP11
    return cpp11::as_sexp(rfx_container_cpp(cpp11::as_cpp<cpp11::decay_t<int>>(num_components), cpp11::as_cpp<cpp11::decay_t<int>>(num_groups)));
  END_CPP11
}
// random_effects.cpp
cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model_cpp(int num_components, int num_groups);
extern "C" SEXP _stochtree_rfx_model_cpp(SEXP num_components, SEXP num_groups) {
  BEGIN_CPP11
    return cpp11::as_sexp(rfx_model_cpp(cpp11::as_cpp<cpp11::decay_t<int>>(num_components), cpp11::as_cpp<cpp11::decay_t<int>>(num_groups)));
  END_CPP11
}
// random_effects.cpp
cpp11::external_pointer<StochTree::RandomEffectsTracker> rfx_tracker_cpp(cpp11::integers group_labels);
extern "C" SEXP _stochtree_rfx_tracker_cpp(SEXP group_labels) {
  BEGIN_CPP11
    return cpp11::as_sexp(rfx_tracker_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::integers>>(group_labels)));
  END_CPP11
}
// random_effects.cpp
cpp11::external_pointer<StochTree::LabelMapper> rfx_label_mapper_cpp(cpp11::external_pointer<StochTree::RandomEffectsTracker> rfx_tracker);
extern "C" SEXP _stochtree_rfx_label_mapper_cpp(SEXP rfx_tracker) {
  BEGIN_CPP11
    return cpp11::as_sexp(rfx_label_mapper_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::RandomEffectsTracker>>>(rfx_tracker)));
  END_CPP11
}
// random_effects.cpp
void rfx_model_sample_random_effects_cpp(cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model, cpp11::external_pointer<StochTree::RandomEffectsDataset> rfx_dataset, cpp11::external_pointer<StochTree::ColumnVector> residual, cpp11::external_pointer<StochTree::RandomEffectsTracker> rfx_tracker, cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container, double global_variance, cpp11::external_pointer<std::mt19937> rng);
extern "C" SEXP _stochtree_rfx_model_sample_random_effects_cpp(SEXP rfx_model, SEXP rfx_dataset, SEXP residual, SEXP rfx_tracker, SEXP rfx_container, SEXP global_variance, SEXP rng) {
  BEGIN_CPP11
    rfx_model_sample_random_effects_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel>>>(rfx_model), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::RandomEffectsDataset>>>(rfx_dataset), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ColumnVector>>>(residual), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::RandomEffectsTracker>>>(rfx_tracker), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::RandomEffectsContainer>>>(rfx_container), cpp11::as_cpp<cpp11::decay_t<double>>(global_variance), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<std::mt19937>>>(rng));
    return R_NilValue;
  END_CPP11
}
// random_effects.cpp
cpp11::writable::doubles rfx_container_predict_cpp(cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container, cpp11::external_pointer<StochTree::RandomEffectsDataset> rfx_dataset, cpp11::external_pointer<StochTree::LabelMapper> label_mapper);
extern "C" SEXP _stochtree_rfx_container_predict_cpp(SEXP rfx_container, SEXP rfx_dataset, SEXP label_mapper) {
  BEGIN_CPP11
    return cpp11::as_sexp(rfx_container_predict_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::RandomEffectsContainer>>>(rfx_container), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::RandomEffectsDataset>>>(rfx_dataset), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::LabelMapper>>>(label_mapper)));
  END_CPP11
}
// random_effects.cpp
int rfx_container_num_samples_cpp(cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container);
extern "C" SEXP _stochtree_rfx_container_num_samples_cpp(SEXP rfx_container) {
  BEGIN_CPP11
    return cpp11::as_sexp(rfx_container_num_samples_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::RandomEffectsContainer>>>(rfx_container)));
  END_CPP11
}
// random_effects.cpp
int rfx_container_num_components_cpp(cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container);
extern "C" SEXP _stochtree_rfx_container_num_components_cpp(SEXP rfx_container) {
  BEGIN_CPP11
    return cpp11::as_sexp(rfx_container_num_components_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::RandomEffectsContainer>>>(rfx_container)));
  END_CPP11
}
// random_effects.cpp
int rfx_container_num_groups_cpp(cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container);
extern "C" SEXP _stochtree_rfx_container_num_groups_cpp(SEXP rfx_container) {
  BEGIN_CPP11
    return cpp11::as_sexp(rfx_container_num_groups_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::RandomEffectsContainer>>>(rfx_container)));
  END_CPP11
}
// random_effects.cpp
void rfx_model_set_working_parameter_cpp(cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model, cpp11::doubles working_param_init);
extern "C" SEXP _stochtree_rfx_model_set_working_parameter_cpp(SEXP rfx_model, SEXP working_param_init) {
  BEGIN_CPP11
    rfx_model_set_working_parameter_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel>>>(rfx_model), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(working_param_init));
    return R_NilValue;
  END_CPP11
}
// random_effects.cpp
void rfx_model_set_group_parameters_cpp(cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model, cpp11::doubles_matrix<> group_params_init);
extern "C" SEXP _stochtree_rfx_model_set_group_parameters_cpp(SEXP rfx_model, SEXP group_params_init) {
  BEGIN_CPP11
    rfx_model_set_group_parameters_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel>>>(rfx_model), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles_matrix<>>>(group_params_init));
    return R_NilValue;
  END_CPP11
}
// random_effects.cpp
void rfx_model_set_working_parameter_covariance_cpp(cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model, cpp11::doubles_matrix<> working_param_cov_init);
extern "C" SEXP _stochtree_rfx_model_set_working_parameter_covariance_cpp(SEXP rfx_model, SEXP working_param_cov_init) {
  BEGIN_CPP11
    rfx_model_set_working_parameter_covariance_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel>>>(rfx_model), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles_matrix<>>>(working_param_cov_init));
    return R_NilValue;
  END_CPP11
}
// random_effects.cpp
void rfx_model_set_group_parameter_covariance_cpp(cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model, cpp11::doubles_matrix<> group_param_cov_init);
extern "C" SEXP _stochtree_rfx_model_set_group_parameter_covariance_cpp(SEXP rfx_model, SEXP group_param_cov_init) {
  BEGIN_CPP11
    rfx_model_set_group_parameter_covariance_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel>>>(rfx_model), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles_matrix<>>>(group_param_cov_init));
    return R_NilValue;
  END_CPP11
}
// random_effects.cpp
void rfx_model_set_variance_prior_shape_cpp(cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model, double shape);
extern "C" SEXP _stochtree_rfx_model_set_variance_prior_shape_cpp(SEXP rfx_model, SEXP shape) {
  BEGIN_CPP11
    rfx_model_set_variance_prior_shape_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel>>>(rfx_model), cpp11::as_cpp<cpp11::decay_t<double>>(shape));
    return R_NilValue;
  END_CPP11
}
// random_effects.cpp
void rfx_model_set_variance_prior_scale_cpp(cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model, double scale);
extern "C" SEXP _stochtree_rfx_model_set_variance_prior_scale_cpp(SEXP rfx_model, SEXP scale) {
  BEGIN_CPP11
    rfx_model_set_variance_prior_scale_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel>>>(rfx_model), cpp11::as_cpp<cpp11::decay_t<double>>(scale));
    return R_NilValue;
  END_CPP11
}
// random_effects.cpp
cpp11::writable::integers rfx_tracker_get_unique_group_ids_cpp(cpp11::external_pointer<StochTree::RandomEffectsTracker> rfx_tracker);
extern "C" SEXP _stochtree_rfx_tracker_get_unique_group_ids_cpp(SEXP rfx_tracker) {
  BEGIN_CPP11
    return cpp11::as_sexp(rfx_tracker_get_unique_group_ids_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::RandomEffectsTracker>>>(rfx_tracker)));
  END_CPP11
}
// sampler.cpp
void sample_gfr_one_iteration_cpp(cpp11::external_pointer<StochTree::ForestDataset> data, cpp11::external_pointer<StochTree::ColumnVector> residual, cpp11::external_pointer<StochTree::ForestContainer> forest_samples, cpp11::external_pointer<StochTree::ForestTracker> tracker, cpp11::external_pointer<StochTree::TreePrior> split_prior, cpp11::external_pointer<std::mt19937> rng, cpp11::integers feature_types, int cutpoint_grid_size, cpp11::doubles_matrix<> leaf_model_scale_input, cpp11::doubles variable_weights, double global_variance, int leaf_model_int, bool pre_initialized);
extern "C" SEXP _stochtree_sample_gfr_one_iteration_cpp(SEXP data, SEXP residual, SEXP forest_samples, SEXP tracker, SEXP split_prior, SEXP rng, SEXP feature_types, SEXP cutpoint_grid_size, SEXP leaf_model_scale_input, SEXP variable_weights, SEXP global_variance, SEXP leaf_model_int, SEXP pre_initialized) {
  BEGIN_CPP11
    sample_gfr_one_iteration_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestDataset>>>(data), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ColumnVector>>>(residual), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestContainer>>>(forest_samples), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestTracker>>>(tracker), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::TreePrior>>>(split_prior), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<std::mt19937>>>(rng), cpp11::as_cpp<cpp11::decay_t<cpp11::integers>>(feature_types), cpp11::as_cpp<cpp11::decay_t<int>>(cutpoint_grid_size), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles_matrix<>>>(leaf_model_scale_input), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(variable_weights), cpp11::as_cpp<cpp11::decay_t<double>>(global_variance), cpp11::as_cpp<cpp11::decay_t<int>>(leaf_model_int), cpp11::as_cpp<cpp11::decay_t<bool>>(pre_initialized));
    return R_NilValue;
  END_CPP11
}
// sampler.cpp
void sample_mcmc_one_iteration_cpp(cpp11::external_pointer<StochTree::ForestDataset> data, cpp11::external_pointer<StochTree::ColumnVector> residual, cpp11::external_pointer<StochTree::ForestContainer> forest_samples, cpp11::external_pointer<StochTree::ForestTracker> tracker, cpp11::external_pointer<StochTree::TreePrior> split_prior, cpp11::external_pointer<std::mt19937> rng, cpp11::integers feature_types, int cutpoint_grid_size, cpp11::doubles_matrix<> leaf_model_scale_input, cpp11::doubles variable_weights, double global_variance, int leaf_model_int, bool pre_initialized);
extern "C" SEXP _stochtree_sample_mcmc_one_iteration_cpp(SEXP data, SEXP residual, SEXP forest_samples, SEXP tracker, SEXP split_prior, SEXP rng, SEXP feature_types, SEXP cutpoint_grid_size, SEXP leaf_model_scale_input, SEXP variable_weights, SEXP global_variance, SEXP leaf_model_int, SEXP pre_initialized) {
  BEGIN_CPP11
    sample_mcmc_one_iteration_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestDataset>>>(data), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ColumnVector>>>(residual), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestContainer>>>(forest_samples), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestTracker>>>(tracker), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::TreePrior>>>(split_prior), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<std::mt19937>>>(rng), cpp11::as_cpp<cpp11::decay_t<cpp11::integers>>(feature_types), cpp11::as_cpp<cpp11::decay_t<int>>(cutpoint_grid_size), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles_matrix<>>>(leaf_model_scale_input), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(variable_weights), cpp11::as_cpp<cpp11::decay_t<double>>(global_variance), cpp11::as_cpp<cpp11::decay_t<int>>(leaf_model_int), cpp11::as_cpp<cpp11::decay_t<bool>>(pre_initialized));
    return R_NilValue;
  END_CPP11
}
// sampler.cpp
double sample_sigma2_one_iteration_cpp(cpp11::external_pointer<StochTree::ColumnVector> residual, cpp11::external_pointer<std::mt19937> rng, double nu, double lambda);
extern "C" SEXP _stochtree_sample_sigma2_one_iteration_cpp(SEXP residual, SEXP rng, SEXP nu, SEXP lambda) {
  BEGIN_CPP11
    return cpp11::as_sexp(sample_sigma2_one_iteration_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ColumnVector>>>(residual), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<std::mt19937>>>(rng), cpp11::as_cpp<cpp11::decay_t<double>>(nu), cpp11::as_cpp<cpp11::decay_t<double>>(lambda)));
  END_CPP11
}
// sampler.cpp
double sample_tau_one_iteration_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, cpp11::external_pointer<std::mt19937> rng, double a, double b, int sample_num);
extern "C" SEXP _stochtree_sample_tau_one_iteration_cpp(SEXP forest_samples, SEXP rng, SEXP a, SEXP b, SEXP sample_num) {
  BEGIN_CPP11
    return cpp11::as_sexp(sample_tau_one_iteration_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestContainer>>>(forest_samples), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<std::mt19937>>>(rng), cpp11::as_cpp<cpp11::decay_t<double>>(a), cpp11::as_cpp<cpp11::decay_t<double>>(b), cpp11::as_cpp<cpp11::decay_t<int>>(sample_num)));
  END_CPP11
}
// sampler.cpp
cpp11::external_pointer<std::mt19937> rng_cpp(int random_seed);
extern "C" SEXP _stochtree_rng_cpp(SEXP random_seed) {
  BEGIN_CPP11
    return cpp11::as_sexp(rng_cpp(cpp11::as_cpp<cpp11::decay_t<int>>(random_seed)));
  END_CPP11
}
// sampler.cpp
cpp11::external_pointer<StochTree::ForestContainer> forest_container_cpp(int num_trees, int output_dimension, bool is_leaf_constant);
extern "C" SEXP _stochtree_forest_container_cpp(SEXP num_trees, SEXP output_dimension, SEXP is_leaf_constant) {
  BEGIN_CPP11
    return cpp11::as_sexp(forest_container_cpp(cpp11::as_cpp<cpp11::decay_t<int>>(num_trees), cpp11::as_cpp<cpp11::decay_t<int>>(output_dimension), cpp11::as_cpp<cpp11::decay_t<bool>>(is_leaf_constant)));
  END_CPP11
}
// sampler.cpp
int num_samples_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples);
extern "C" SEXP _stochtree_num_samples_forest_container_cpp(SEXP forest_samples) {
  BEGIN_CPP11
    return cpp11::as_sexp(num_samples_forest_container_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestContainer>>>(forest_samples)));
  END_CPP11
}
// sampler.cpp
void json_save_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, std::string json_filename);
extern "C" SEXP _stochtree_json_save_forest_container_cpp(SEXP forest_samples, SEXP json_filename) {
  BEGIN_CPP11
    json_save_forest_container_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestContainer>>>(forest_samples), cpp11::as_cpp<cpp11::decay_t<std::string>>(json_filename));
    return R_NilValue;
  END_CPP11
}
// sampler.cpp
void json_load_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, std::string json_filename);
extern "C" SEXP _stochtree_json_load_forest_container_cpp(SEXP forest_samples, SEXP json_filename) {
  BEGIN_CPP11
    json_load_forest_container_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestContainer>>>(forest_samples), cpp11::as_cpp<cpp11::decay_t<std::string>>(json_filename));
    return R_NilValue;
  END_CPP11
}
// sampler.cpp
int output_dimension_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples);
extern "C" SEXP _stochtree_output_dimension_forest_container_cpp(SEXP forest_samples) {
  BEGIN_CPP11
    return cpp11::as_sexp(output_dimension_forest_container_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestContainer>>>(forest_samples)));
  END_CPP11
}
// sampler.cpp
int is_leaf_constant_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples);
extern "C" SEXP _stochtree_is_leaf_constant_forest_container_cpp(SEXP forest_samples) {
  BEGIN_CPP11
    return cpp11::as_sexp(is_leaf_constant_forest_container_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestContainer>>>(forest_samples)));
  END_CPP11
}
// sampler.cpp
bool all_roots_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, int forest_num);
extern "C" SEXP _stochtree_all_roots_forest_container_cpp(SEXP forest_samples, SEXP forest_num) {
  BEGIN_CPP11
    return cpp11::as_sexp(all_roots_forest_container_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestContainer>>>(forest_samples), cpp11::as_cpp<cpp11::decay_t<int>>(forest_num)));
  END_CPP11
}
// sampler.cpp
void add_sample_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples);
extern "C" SEXP _stochtree_add_sample_forest_container_cpp(SEXP forest_samples) {
  BEGIN_CPP11
    add_sample_forest_container_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestContainer>>>(forest_samples));
    return R_NilValue;
  END_CPP11
}
// sampler.cpp
void set_leaf_value_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, double leaf_value);
extern "C" SEXP _stochtree_set_leaf_value_forest_container_cpp(SEXP forest_samples, SEXP leaf_value) {
  BEGIN_CPP11
    set_leaf_value_forest_container_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestContainer>>>(forest_samples), cpp11::as_cpp<cpp11::decay_t<double>>(leaf_value));
    return R_NilValue;
  END_CPP11
}
// sampler.cpp
void set_leaf_vector_forest_container_cpp(cpp11::external_pointer<StochTree::ForestContainer> forest_samples, cpp11::doubles leaf_vector);
extern "C" SEXP _stochtree_set_leaf_vector_forest_container_cpp(SEXP forest_samples, SEXP leaf_vector) {
  BEGIN_CPP11
    set_leaf_vector_forest_container_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestContainer>>>(forest_samples), cpp11::as_cpp<cpp11::decay_t<cpp11::doubles>>(leaf_vector));
    return R_NilValue;
  END_CPP11
}
// sampler.cpp
void update_residual_forest_container_cpp(cpp11::external_pointer<StochTree::ForestDataset> data, cpp11::external_pointer<StochTree::ColumnVector> residual, cpp11::external_pointer<StochTree::ForestContainer> forest_samples, cpp11::external_pointer<StochTree::ForestTracker> tracker, bool requires_basis, int forest_num, bool add);
extern "C" SEXP _stochtree_update_residual_forest_container_cpp(SEXP data, SEXP residual, SEXP forest_samples, SEXP tracker, SEXP requires_basis, SEXP forest_num, SEXP add) {
  BEGIN_CPP11
    update_residual_forest_container_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestDataset>>>(data), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ColumnVector>>>(residual), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestContainer>>>(forest_samples), cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestTracker>>>(tracker), cpp11::as_cpp<cpp11::decay_t<bool>>(requires_basis), cpp11::as_cpp<cpp11::decay_t<int>>(forest_num), cpp11::as_cpp<cpp11::decay_t<bool>>(add));
    return R_NilValue;
  END_CPP11
}
// sampler.cpp
cpp11::external_pointer<StochTree::TreePrior> tree_prior_cpp(double alpha, double beta, int min_samples_leaf);
extern "C" SEXP _stochtree_tree_prior_cpp(SEXP alpha, SEXP beta, SEXP min_samples_leaf) {
  BEGIN_CPP11
    return cpp11::as_sexp(tree_prior_cpp(cpp11::as_cpp<cpp11::decay_t<double>>(alpha), cpp11::as_cpp<cpp11::decay_t<double>>(beta), cpp11::as_cpp<cpp11::decay_t<int>>(min_samples_leaf)));
  END_CPP11
}
// sampler.cpp
cpp11::external_pointer<StochTree::ForestTracker> forest_tracker_cpp(cpp11::external_pointer<StochTree::ForestDataset> data, cpp11::integers feature_types, int num_trees, StochTree::data_size_t n);
extern "C" SEXP _stochtree_forest_tracker_cpp(SEXP data, SEXP feature_types, SEXP num_trees, SEXP n) {
  BEGIN_CPP11
    return cpp11::as_sexp(forest_tracker_cpp(cpp11::as_cpp<cpp11::decay_t<cpp11::external_pointer<StochTree::ForestDataset>>>(data), cpp11::as_cpp<cpp11::decay_t<cpp11::integers>>(feature_types), cpp11::as_cpp<cpp11::decay_t<int>>(num_trees), cpp11::as_cpp<cpp11::decay_t<StochTree::data_size_t>>(n)));
  END_CPP11
}

extern "C" {
static const R_CallMethodDef CallEntries[] = {
    {"_stochtree_add_sample_forest_container_cpp",                (DL_FUNC) &_stochtree_add_sample_forest_container_cpp,                 1},
    {"_stochtree_all_roots_forest_container_cpp",                 (DL_FUNC) &_stochtree_all_roots_forest_container_cpp,                  2},
    {"_stochtree_create_column_vector_cpp",                       (DL_FUNC) &_stochtree_create_column_vector_cpp,                        1},
    {"_stochtree_create_forest_dataset_cpp",                      (DL_FUNC) &_stochtree_create_forest_dataset_cpp,                       0},
    {"_stochtree_create_rfx_dataset_cpp",                         (DL_FUNC) &_stochtree_create_rfx_dataset_cpp,                          0},
    {"_stochtree_dataset_has_basis_cpp",                          (DL_FUNC) &_stochtree_dataset_has_basis_cpp,                           1},
    {"_stochtree_dataset_has_variance_weights_cpp",               (DL_FUNC) &_stochtree_dataset_has_variance_weights_cpp,                1},
    {"_stochtree_dataset_num_basis_cpp",                          (DL_FUNC) &_stochtree_dataset_num_basis_cpp,                           1},
    {"_stochtree_dataset_num_covariates_cpp",                     (DL_FUNC) &_stochtree_dataset_num_covariates_cpp,                      1},
    {"_stochtree_dataset_num_rows_cpp",                           (DL_FUNC) &_stochtree_dataset_num_rows_cpp,                            1},
    {"_stochtree_forest_container_cpp",                           (DL_FUNC) &_stochtree_forest_container_cpp,                            3},
    {"_stochtree_forest_dataset_add_basis_cpp",                   (DL_FUNC) &_stochtree_forest_dataset_add_basis_cpp,                    2},
    {"_stochtree_forest_dataset_add_covariates_cpp",              (DL_FUNC) &_stochtree_forest_dataset_add_covariates_cpp,               2},
    {"_stochtree_forest_dataset_add_weights_cpp",                 (DL_FUNC) &_stochtree_forest_dataset_add_weights_cpp,                  2},
    {"_stochtree_forest_dataset_update_basis_cpp",                (DL_FUNC) &_stochtree_forest_dataset_update_basis_cpp,                 2},
    {"_stochtree_forest_tracker_cpp",                             (DL_FUNC) &_stochtree_forest_tracker_cpp,                              4},
    {"_stochtree_is_leaf_constant_forest_container_cpp",          (DL_FUNC) &_stochtree_is_leaf_constant_forest_container_cpp,           1},
    {"_stochtree_json_load_forest_container_cpp",                 (DL_FUNC) &_stochtree_json_load_forest_container_cpp,                  2},
    {"_stochtree_json_save_forest_container_cpp",                 (DL_FUNC) &_stochtree_json_save_forest_container_cpp,                  2},
    {"_stochtree_num_samples_forest_container_cpp",               (DL_FUNC) &_stochtree_num_samples_forest_container_cpp,                1},
    {"_stochtree_output_dimension_forest_container_cpp",          (DL_FUNC) &_stochtree_output_dimension_forest_container_cpp,           1},
    {"_stochtree_predict_forest_cpp",                             (DL_FUNC) &_stochtree_predict_forest_cpp,                              2},
    {"_stochtree_predict_forest_raw_cpp",                         (DL_FUNC) &_stochtree_predict_forest_raw_cpp,                          2},
    {"_stochtree_predict_forest_raw_single_forest_cpp",           (DL_FUNC) &_stochtree_predict_forest_raw_single_forest_cpp,            3},
    {"_stochtree_rfx_container_cpp",                              (DL_FUNC) &_stochtree_rfx_container_cpp,                               2},
    {"_stochtree_rfx_container_num_components_cpp",               (DL_FUNC) &_stochtree_rfx_container_num_components_cpp,                1},
    {"_stochtree_rfx_container_num_groups_cpp",                   (DL_FUNC) &_stochtree_rfx_container_num_groups_cpp,                    1},
    {"_stochtree_rfx_container_num_samples_cpp",                  (DL_FUNC) &_stochtree_rfx_container_num_samples_cpp,                   1},
    {"_stochtree_rfx_container_predict_cpp",                      (DL_FUNC) &_stochtree_rfx_container_predict_cpp,                       3},
    {"_stochtree_rfx_dataset_add_basis_cpp",                      (DL_FUNC) &_stochtree_rfx_dataset_add_basis_cpp,                       2},
    {"_stochtree_rfx_dataset_add_group_labels_cpp",               (DL_FUNC) &_stochtree_rfx_dataset_add_group_labels_cpp,                2},
    {"_stochtree_rfx_dataset_add_weights_cpp",                    (DL_FUNC) &_stochtree_rfx_dataset_add_weights_cpp,                     2},
    {"_stochtree_rfx_dataset_has_basis_cpp",                      (DL_FUNC) &_stochtree_rfx_dataset_has_basis_cpp,                       1},
    {"_stochtree_rfx_dataset_has_group_labels_cpp",               (DL_FUNC) &_stochtree_rfx_dataset_has_group_labels_cpp,                1},
    {"_stochtree_rfx_dataset_has_variance_weights_cpp",           (DL_FUNC) &_stochtree_rfx_dataset_has_variance_weights_cpp,            1},
    {"_stochtree_rfx_dataset_num_rows_cpp",                       (DL_FUNC) &_stochtree_rfx_dataset_num_rows_cpp,                        1},
    {"_stochtree_rfx_label_mapper_cpp",                           (DL_FUNC) &_stochtree_rfx_label_mapper_cpp,                            1},
    {"_stochtree_rfx_model_cpp",                                  (DL_FUNC) &_stochtree_rfx_model_cpp,                                   2},
    {"_stochtree_rfx_model_sample_random_effects_cpp",            (DL_FUNC) &_stochtree_rfx_model_sample_random_effects_cpp,             7},
    {"_stochtree_rfx_model_set_group_parameter_covariance_cpp",   (DL_FUNC) &_stochtree_rfx_model_set_group_parameter_covariance_cpp,    2},
    {"_stochtree_rfx_model_set_group_parameters_cpp",             (DL_FUNC) &_stochtree_rfx_model_set_group_parameters_cpp,              2},
    {"_stochtree_rfx_model_set_variance_prior_scale_cpp",         (DL_FUNC) &_stochtree_rfx_model_set_variance_prior_scale_cpp,          2},
    {"_stochtree_rfx_model_set_variance_prior_shape_cpp",         (DL_FUNC) &_stochtree_rfx_model_set_variance_prior_shape_cpp,          2},
    {"_stochtree_rfx_model_set_working_parameter_covariance_cpp", (DL_FUNC) &_stochtree_rfx_model_set_working_parameter_covariance_cpp,  2},
    {"_stochtree_rfx_model_set_working_parameter_cpp",            (DL_FUNC) &_stochtree_rfx_model_set_working_parameter_cpp,             2},
    {"_stochtree_rfx_tracker_cpp",                                (DL_FUNC) &_stochtree_rfx_tracker_cpp,                                 1},
    {"_stochtree_rfx_tracker_get_unique_group_ids_cpp",           (DL_FUNC) &_stochtree_rfx_tracker_get_unique_group_ids_cpp,            1},
    {"_stochtree_rng_cpp",                                        (DL_FUNC) &_stochtree_rng_cpp,                                         1},
    {"_stochtree_sample_gfr_one_iteration_cpp",                   (DL_FUNC) &_stochtree_sample_gfr_one_iteration_cpp,                   13},
    {"_stochtree_sample_mcmc_one_iteration_cpp",                  (DL_FUNC) &_stochtree_sample_mcmc_one_iteration_cpp,                  13},
    {"_stochtree_sample_sigma2_one_iteration_cpp",                (DL_FUNC) &_stochtree_sample_sigma2_one_iteration_cpp,                 4},
    {"_stochtree_sample_tau_one_iteration_cpp",                   (DL_FUNC) &_stochtree_sample_tau_one_iteration_cpp,                    5},
    {"_stochtree_set_leaf_value_forest_container_cpp",            (DL_FUNC) &_stochtree_set_leaf_value_forest_container_cpp,             2},
    {"_stochtree_set_leaf_vector_forest_container_cpp",           (DL_FUNC) &_stochtree_set_leaf_vector_forest_container_cpp,            2},
    {"_stochtree_tree_prior_cpp",                                 (DL_FUNC) &_stochtree_tree_prior_cpp,                                  3},
    {"_stochtree_update_residual_forest_container_cpp",           (DL_FUNC) &_stochtree_update_residual_forest_container_cpp,            7},
    {NULL, NULL, 0}
};
}

extern "C" attribute_visible void R_init_stochtree(DllInfo* dll){
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
  R_forceSymbols(dll, TRUE);
}
