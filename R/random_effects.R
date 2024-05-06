#' Class that wraps the "persistent" aspects of a C++ random effects model
#' (draws of the parameters and a map from the original label indices to the 
#' 0-indexed label numbers used to place group samples in memory (i.e. the 
#' first label is stored in column 0 of the sample matrix, the second label 
#' is store in column 1 of the sample matrix, etc...))
#'
#' @description
#' Coordinates various C++ random effects classes and persists those 
#' needed for prediction / serialization

RandomEffectSamples <- R6::R6Class(
    classname = "RandomEffectSamples",
    cloneable = FALSE,
    public = list(
        
        #' @field rfx_container_ptr External pointer to a C++ StochTree::RandomEffectsContainer class
        rfx_container_ptr = NULL,
        
        #' @field label_mapper_ptr External pointer to a C++ StochTree::LabelMapper class
        label_mapper_ptr = NULL,
        
        #' @field training_group_ids Unique vector of group IDs that were in the training dataset
        training_group_ids = NULL,
        
        #' @description
        #' Create a new RandomEffectSamples object.
        #' @param num_components Number of "components" or bases defining the random effects regression
        #' @param num_groups Number of random effects groups
        #' @param random_effects_tracker Object of type `RandomEffectsTracker`
        #' @return A new `RandomEffectSamples` object.
        initialize = function(num_components, num_groups, random_effects_tracker) {
            # Initialize
            self$rfx_container_ptr <- rfx_container_cpp(num_components, num_groups)
            self$label_mapper_ptr <- rfx_label_mapper_cpp(random_effects_tracker$rfx_tracker_ptr)
            self$training_group_ids <- rfx_tracker_get_unique_group_ids_cpp(random_effects_tracker$rfx_tracker_ptr)
        }, 
        
        #' @description
        #' Predict random effects for each observation implied by `rfx_group_ids` and `rfx_basis`. 
        #' If a random effects model is "intercept-only" the `rfx_basis` will be a vector of ones of size `length(rfx_group_ids)`.
        #' @param rfx_group_ids Indices of random effects groups in a prediction set
        #' @param rfx_basis Basis used for random effects prediction
        #' @return Matrix with as many rows as observations provided and as many columns as samples drawn of the model.
        predict = function(rfx_group_ids, rfx_basis) {
            num_observations = length(rfx_group_ids)
            num_samples = rfx_container_num_samples_cpp(self$rfx_container_ptr)
            num_components = rfx_container_num_components_cpp(self$rfx_container_ptr)
            num_groups = rfx_container_num_groups_cpp(self$rfx_container_ptr)
            stopifnot(sum(!(rfx_group_ids %in% self$training_group_ids)) == 0)
            stopifnot(ncol(rfx_basis) == num_components)
            rfx_dataset <- createRandomEffectsDataset(rfx_group_ids, rfx_basis)
            output <- rfx_container_predict_cpp(self$rfx_container_ptr, rfx_dataset$data_ptr, self$label_mapper_ptr)
            dim(output) <- c(num_observations, num_samples)
            return(output)
        }, 
        
        #' @description
        #' Extract the random effects parameters sampled. With the "redundant parameterization" 
        #' of Gelman et al (2008), this includes four parameters: alpha (the "working parameter" 
        #' shared across every group), xi (the "group parameter" sampled separately for each group), 
        #' beta (the product of alpha and xi, which corresponds to the overall group-level random effects), 
        #' and sigma (group-independent prior variance for each component of xi).
        #' @return List of arrays. The alpha array has dimension (`num_components`, `num_samples`) and is simply a vector if `num_components = 1`.
        #' The xi and beta arrays have dimension (`num_components`, `num_groups`, `num_samples`) and is simply a matrix if `num_components = 1`.
        #' The sigma array has dimension (`num_components`, `num_samples`) and is simply a vector if `num_components = 1`.
        extract_parameter_samples = function() {
            num_samples = rfx_container_num_samples_cpp(self$rfx_container_ptr)
            num_components = rfx_container_num_components_cpp(self$rfx_container_ptr)
            num_groups = rfx_container_num_groups_cpp(self$rfx_container_ptr)
            beta_samples <- rfx_container_get_beta_cpp(self$rfx_container_ptr)
            xi_samples <- rfx_container_get_xi_cpp(self$rfx_container_ptr)
            alpha_samples <- rfx_container_get_alpha_cpp(self$rfx_container_ptr)
            sigma_samples <- rfx_container_get_sigma_cpp(self$rfx_container_ptr)
            if (num_components == 1) {
                dim(beta_samples) <- c(num_groups, num_samples)
                dim(xi_samples) <- c(num_groups, num_samples)
            } else if (num_components > 1) {
                dim(beta_samples) <- c(num_components, num_groups, num_samples)
                dim(xi_samples) <- c(num_components, num_groups, num_samples)
                dim(alpha_samples) <- c(num_components, num_samples)
                dim(sigma_samples) <- c(num_components, num_samples)
            } else stop("Invalid random effects sample container, num_components is less than 1")
            
            output = list(
                "beta_samples" = beta_samples, 
                "xi_samples" = xi_samples, 
                "alpha_samples" = alpha_samples, 
                "sigma_samples" = sigma_samples
            )
            return(output)
        }
    )
)

#' Class that defines a "tracker" for random effects models, most notably  
#' storing the data indices available in each group for quicker posterior 
#' computation and sampling of random effects terms.
#'
#' @description
#' Stores a mapping from every observation to its group index, a mapping 
#' from group indices to the training sample observations available in that 
#' group, and predictions for each observation.

RandomEffectsTracker <- R6::R6Class(
    classname = "RandomEffectsTracker",
    cloneable = FALSE,
    public = list(
        
        #' @field rfx_tracker_ptr External pointer to a C++ StochTree::RandomEffectsTracker class
        rfx_tracker_ptr = NULL,
        
        #' @description
        #' Create a new RandomEffectsTracker object.
        #' @param rfx_group_indices Integer indices indicating groups used to define random effects
        #' @return A new `RandomEffectsTracker` object.
        initialize = function(rfx_group_indices) {
            # Initialize
            self$rfx_tracker_ptr <- rfx_tracker_cpp(rfx_group_indices)
        }
    )
)

#' The core "model" class for sampling random effects.
#'
#' @description
#' Stores current model state, prior parameters, and procedures for 
#' sampling from the conditional posterior of each parameter.

RandomEffectsModel <- R6::R6Class(
    classname = "RandomEffectsModel",
    cloneable = FALSE,
    public = list(
        
        #' @field rfx_model_ptr External pointer to a C++ StochTree::RandomEffectsModel class
        rfx_model_ptr = NULL,
        
        #' @field num_groups Number of groups in the random effects model
        num_groups = NULL,
        
        #' @field num_components Number of components (i.e. dimension of basis) in the random effects model
        num_components = NULL,
        
        #' @description
        #' Create a new RandomEffectsModel object.
        #' @param num_components Number of "components" or bases defining the random effects regression
        #' @param num_groups Number of random effects groups
        #' @return A new `RandomEffectsModel` object.
        initialize = function(num_components, num_groups) {
            # Initialize
            self$rfx_model_ptr <- rfx_model_cpp(num_components, num_groups)
            self$num_components <- num_components
            self$num_groups <- num_groups
        },
        
        #' @description
        #' Sample from random effects model.
        #' @param rfx_dataset Object of type `RandomEffectsDataset`
        #' @param residual Object of type `Outcome`
        #' @param rfx_tracker Object of type `RandomEffectsTracker`
        #' @param rfx_samples Object of type `RandomEffectSamples`
        #' @param global_variance Scalar global variance parameter
        #' @param rng Object of type `CppRNG`
        #' @return None
        sample_random_effect = function(rfx_dataset, residual, rfx_tracker, rfx_samples, global_variance, rng) {
            rfx_model_sample_random_effects_cpp(self$rfx_model_ptr, rfx_dataset$data_ptr, 
                                                residual$data_ptr, rfx_tracker$rfx_tracker_ptr, 
                                                rfx_samples$rfx_container_ptr, global_variance, rng$rng_ptr)
        },
        
        #' @description
        #' Set value for the "working parameter." This is typically 
        #' used for initialization, but could also be used to interrupt 
        #' or override the sampler.
        #' @param value Parameter input
        #' @return None
        set_working_parameter = function(value) {
            stopifnot(is.double(value))
            stopifnot(!is.matrix(value))
            stopifnot(length(value) == self$num_components)
            rfx_model_set_working_parameter_cpp(self$rfx_model_ptr, value)
        },
        
        #' @description
        #' Set value for the "group parameters." This is typically 
        #' used for initialization, but could also be used to interrupt 
        #' or override the sampler.
        #' @param value Parameter input
        #' @return None
        set_group_parameters = function(value) {
            stopifnot(is.double(value))
            stopifnot(is.matrix(value))
            stopifnot(nrow(value) == self$num_components)
            stopifnot(ncol(value) == self$num_groups)
            rfx_model_set_group_parameters_cpp(self$rfx_model_ptr, value)
        },
        
        #' @description
        #' Set value for the working parameter covariance. This is typically 
        #' used for initialization, but could also be used to interrupt 
        #' or override the sampler.
        #' @param value Parameter input
        #' @return None
        set_working_parameter_cov = function(value) {
            stopifnot(is.double(value))
            stopifnot(is.matrix(value))
            stopifnot(nrow(value) == self$num_components)
            stopifnot(ncol(value) == self$num_components)
            rfx_model_set_working_parameter_covariance_cpp(self$rfx_model_ptr, value)
        },
        
        #' @description
        #' Set value for the group parameter covariance. This is typically 
        #' used for initialization, but could also be used to interrupt 
        #' or override the sampler.
        #' @param value Parameter input
        #' @return None
        set_group_parameter_cov = function(value) {
            stopifnot(is.double(value))
            stopifnot(is.matrix(value))
            stopifnot(nrow(value) == self$num_components)
            stopifnot(ncol(value) == self$num_components)
            rfx_model_set_group_parameter_covariance_cpp(self$rfx_model_ptr, value)
        }, 
        
        #' @description
        #' Set shape parameter for the group parameter variance prior.
        #' @param value Parameter input
        #' @return None
        set_variance_prior_shape = function(value) {
            stopifnot(is.double(value))
            stopifnot(!is.matrix(value))
            stopifnot(length(value) == 1)
            rfx_model_set_variance_prior_shape_cpp(self$rfx_model_ptr, value)
        },
        
        #' @description
        #' Set shape parameter for the group parameter variance prior.
        #' @param value Parameter input
        #' @return None
        set_variance_prior_scale = function(value) {
            stopifnot(is.double(value))
            stopifnot(!is.matrix(value))
            stopifnot(length(value) == 1)
            rfx_model_set_variance_prior_scale_cpp(self$rfx_model_ptr, value)
        }
    )
)

#' Create a `RandomEffectSamples` object
#'
#' @param num_components Number of "components" or bases defining the random effects regression
#' @param num_groups Number of random effects groups
#' @param random_effects_tracker Object of type `RandomEffectsTracker`
#' @return `RandomEffectSamples` object
#' @export
createRandomEffectSamples <- function(num_components, num_groups, random_effects_tracker) {
    return(invisible((
        RandomEffectSamples$new(num_components, num_groups, random_effects_tracker)
    )))
}

#' Create a `RandomEffectsTracker` object
#'
#' @param rfx_group_indices Integer indices indicating groups used to define random effects
#' @return `RandomEffectsTracker` object
#' @export
createRandomEffectsTracker <- function(rfx_group_indices) {
    return(invisible((
        RandomEffectsTracker$new(rfx_group_indices)
    )))
}

#' Create a `RandomEffectsModel` object
#'
#' @param num_components Number of "components" or bases defining the random effects regression
#' @param num_groups Number of random effects groups
#' @return `RandomEffectsModel` object
#' @export
createRandomEffectsModel <- function(num_components, num_groups) {
    return(invisible((
        RandomEffectsModel$new(num_components, num_groups)
    )))
}
