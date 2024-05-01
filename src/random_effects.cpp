#include <cpp11.hpp>
#include "stochtree_types.h"
#include <stochtree/container.h>
#include <stochtree/leaf_model.h>
#include <stochtree/meta.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/random_effects.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>
#include <functional>
#include <memory>
#include <vector>

[[cpp11::register]]
cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container_cpp(int num_components, int num_groups) {
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::RandomEffectsContainer> rfx_container_ptr_ = std::make_unique<StochTree::RandomEffectsContainer>(num_components, num_groups);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::RandomEffectsContainer>(rfx_container_ptr_.release());
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model_cpp(int num_components, int num_groups) {
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model_ptr_ = std::make_unique<StochTree::MultivariateRegressionRandomEffectsModel>(num_components, num_groups);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel>(rfx_model_ptr_.release());
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::RandomEffectsTracker> rfx_tracker_cpp(cpp11::integers group_labels) {
    // Convert group_labels to a std::vector<int32_t>
    std::vector<int32_t> group_labels_vec(group_labels.begin(), group_labels.end());
    
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::RandomEffectsTracker> rfx_tracker_ptr_ = std::make_unique<StochTree::RandomEffectsTracker>(group_labels_vec);
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::RandomEffectsTracker>(rfx_tracker_ptr_.release());
}

[[cpp11::register]]
cpp11::external_pointer<StochTree::LabelMapper> rfx_label_mapper_cpp(cpp11::external_pointer<StochTree::RandomEffectsTracker> rfx_tracker) {
    // Create smart pointer to newly allocated object
    std::unique_ptr<StochTree::LabelMapper> rfx_label_mapper_ptr_ = std::make_unique<StochTree::LabelMapper>(rfx_tracker->GetLabelMap());
    
    // Release management of the pointer to R session
    return cpp11::external_pointer<StochTree::LabelMapper>(rfx_label_mapper_ptr_.release());
}

[[cpp11::register]]
void rfx_model_sample_random_effects_cpp(cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model, cpp11::external_pointer<StochTree::RandomEffectsDataset> rfx_dataset, 
                                         cpp11::external_pointer<StochTree::ColumnVector> residual, cpp11::external_pointer<StochTree::RandomEffectsTracker> rfx_tracker, 
                                         cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container, double global_variance, cpp11::external_pointer<std::mt19937> rng) {
    rfx_model->SampleRandomEffects(*rfx_dataset, *residual, *rfx_tracker, global_variance, *rng);
    rfx_container->AddSample(*rfx_model);
}

[[cpp11::register]]
cpp11::writable::doubles rfx_container_predict_cpp(cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container, 
                                                   cpp11::external_pointer<StochTree::RandomEffectsDataset> rfx_dataset, 
                                                   cpp11::external_pointer<StochTree::LabelMapper> label_mapper) {
    int num_observations = rfx_dataset->NumObservations();
    int num_samples = rfx_container->NumSamples();
    std::vector<double> output(num_observations*num_samples);
    rfx_container->Predict(*rfx_dataset, *label_mapper, output);
    return output;
}

[[cpp11::register]]
int rfx_container_num_samples_cpp(cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container) {
    return rfx_container->NumSamples();
}

[[cpp11::register]]
int rfx_container_num_components_cpp(cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container) {
    return rfx_container->NumComponents();
}

[[cpp11::register]]
int rfx_container_num_groups_cpp(cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_container) {
    return rfx_container->NumGroups();
}

[[cpp11::register]]
void rfx_model_set_working_parameter_cpp(cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model, cpp11::doubles working_param_init) {
    Eigen::VectorXd working_param_eigen(working_param_init.size());
    for (int i = 0; i < working_param_init.size(); i++) {
        working_param_eigen(i) = working_param_init.at(i);
    }
    rfx_model->SetWorkingParameter(working_param_eigen);
}

[[cpp11::register]]
void rfx_model_set_group_parameters_cpp(cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model, cpp11::doubles_matrix<> group_params_init) {
    Eigen::MatrixXd group_params_eigen(group_params_init.nrow(), group_params_init.ncol());
    for (int i = 0; i < group_params_init.nrow(); i++) {
        for (int j = 0; j < group_params_init.ncol(); j++) {
            group_params_eigen(i,j) = group_params_init(i,j);
        }
    }
    rfx_model->SetGroupParameters(group_params_eigen);
}

[[cpp11::register]]
void rfx_model_set_working_parameter_covariance_cpp(cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model, cpp11::doubles_matrix<> working_param_cov_init) {
    Eigen::MatrixXd working_param_cov_eigen(working_param_cov_init.nrow(), working_param_cov_init.ncol());
    for (int i = 0; i < working_param_cov_init.nrow(); i++) {
        for (int j = 0; j < working_param_cov_init.ncol(); j++) {
            working_param_cov_eigen(i,j) = working_param_cov_init(i,j);
        }
    }
    rfx_model->SetWorkingParameterCovariance(working_param_cov_eigen);
}

[[cpp11::register]]
void rfx_model_set_group_parameter_covariance_cpp(cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model, cpp11::doubles_matrix<> group_param_cov_init) {
    Eigen::MatrixXd group_param_cov_eigen(group_param_cov_init.nrow(), group_param_cov_init.ncol());
    for (int i = 0; i < group_param_cov_init.nrow(); i++) {
        for (int j = 0; j < group_param_cov_init.ncol(); j++) {
            group_param_cov_eigen(i,j) = group_param_cov_init(i,j);
        }
    }
    rfx_model->SetGroupParameterCovariance(group_param_cov_eigen);
}

[[cpp11::register]]
void rfx_model_set_variance_prior_shape_cpp(cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model, double shape) {
    rfx_model->SetVariancePriorShape(shape);
}

[[cpp11::register]]
void rfx_model_set_variance_prior_scale_cpp(cpp11::external_pointer<StochTree::MultivariateRegressionRandomEffectsModel> rfx_model, double scale) {
    rfx_model->SetVariancePriorScale(scale);
}

[[cpp11::register]]
cpp11::writable::integers rfx_tracker_get_unique_group_ids_cpp(cpp11::external_pointer<StochTree::RandomEffectsTracker> rfx_tracker) {
    std::vector<int32_t> output = rfx_tracker->GetUniqueGroupIds();
    return output;
}
