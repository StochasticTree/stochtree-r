#' Class that wraps a C++ random effects term
#'
#' @description
#' Coordinates various C++ random effects classes and persists those 
#' needed for prediction / serialization

RandomEffectsTerm <- R6::R6Class(
    classname = "RandomEffectsTerm",
    cloneable = FALSE,
    public = list(
        
        #' @field rng_ptr External pointer to a C++ StochTree::RandomEffectsContainer class
        rfx_container_ptr = NULL,
        
        #' @field rng_ptr External pointer to a C++ StochTree::LabelMapper class
        label_mapper_ptr = NULL,
        
        #' @field rng_ptr External pointer to a C++ StochTree::RandomEffectsTracker class
        rfx_tracker_ptr = NULL,
        
        #' @field rng_ptr External pointer to a C++ StochTree::MultivariateRegressionRandomEffectsModel class
        rfx_model_ptr = NULL,
        
        #' @description
        #' Create a new RandomEffectsTerm object.
        #' @param rfx_group_indices Integer indices indicating groups used to define random effects
        #' @param rfx_basis Matrix of bases used to define the random effects regression
        #' @return A new `RandomEffectsTerm` object.
        initialize = function(rfx_group_indices, rfx_basis) {
            # Placeholder implementation
            self$rfx_container_ptr <- rfx_container_cpp()
            self$label_mapper_ptr <- label_mapper_cpp()
            self$rfx_tracker_ptr <- rfx_tracker_cpp()
            self$rfx_model_ptr <- rfx_model_cpp()
        }
    )
)

#' Create a random effects object
#'
#' @param rfx_group_indices Integer indices indicating groups used to define random effects
#' @param rfx_basis Matrix of bases used to define the random effects regression
#' @return `RandomEffectsTerm` object
#' @export
createRandomEffectsTerm <- function(rfx_group_indices, rfx_basis) {
    return(invisible((
        RandomEffectsTerm$new(rfx_group_indices, rfx_basis)
    )))
}
