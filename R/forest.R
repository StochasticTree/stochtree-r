#' Class that stores draws from an random ensemble of decision trees
#'
#' @description
#' Wrapper around a C++ container of tree ensembles

ForestSamples <- R6::R6Class(
    classname = "ForestSamples",
    cloneable = FALSE,
    public = list(
        
        #' @field forest_container_ptr External pointer to a C++ ForestContainer class
        forest_container_ptr = NULL,
        
        #' @description
        #' Create a new ForestContainer object.
        #' @param num_trees Number of trees
        #' @param output_dimension Dimensionality of the outcome model
        #' @param is_leaf_constant Whether leaf is constant
        #' @return A new `ForestContainer` object.
        initialize = function(num_trees, output_dimension=1, is_leaf_constant=F) {
            self$forest_container_ptr <- forest_container_cpp(num_trees, output_dimension, is_leaf_constant)
        }, 
        
        #' @description
        #' Predict every tree ensemble on every sample in `forest_dataset`
        #' @param forest_dataset `ForestDataset` R class
        #' @return matrix of predictions with as many rows as in forest_dataset 
        #' and as many columns as samples in the `ForestContainer`
        predict = function(forest_dataset) {
            stopifnot(!is.null(forest_dataset$data_ptr))
            return(predict_forest_cpp(self$forest_container_ptr, forest_dataset$data_ptr))
        }, 
        
        #' @description
        #' Predict "raw" leaf values (without being multiplied by basis) for every tree ensemble on every sample in `forest_dataset`
        #' @param forest_dataset `ForestDataset` R class
        #' @return Array of predictions for each observation in `forest_dataset` and 
        #' each sample in the `ForestSamples` class with each prediction having the 
        #' dimensionality of the forests' leaf model. In the case of a constant leaf model 
        #' or univariate leaf regression, this array is two-dimensional (number of observations, 
        #' number of forest samples). In the case of a multivariate leaf regression, 
        #' this array is three-dimension (number of observations, leaf model dimension, 
        #' number of samples).
        predict_raw = function(forest_dataset) {
            stopifnot(!is.null(forest_dataset$data_ptr))
            # Unpack dimensions
            output_dim <- output_dimension_forest_container_cpp(self$forest_container_ptr)
            num_samples <- num_samples_forest_container_cpp(self$forest_container_ptr)
            n <- dataset_num_rows_cpp(forest_dataset$data_ptr)
            
            # Predict leaf values from forest
            predictions <- predict_forest_raw_cpp(self$forest_container_ptr, forest_dataset$data_ptr)
            
            # Extract results
            if (output_dim > 1) {
                output <- aperm(array(predictions, c(output_dim, n, num_samples)), c(2,1,3))
            } else {
                output <- predictions
            }
            return(output)
        }, 
        
        #' @description
        #' Predict "raw" leaf values (without being multiplied by basis) for a specific forest on every sample in `forest_dataset`
        #' @param forest_dataset `ForestDataset` R class
        #' @param forest_num Index of the forest sample within the container
        #' @return matrix of predictions with as many rows as in forest_dataset 
        #' and as many columns as samples in the `ForestContainer`
        predict_raw_single_forest = function(forest_dataset, forest_num) {
            stopifnot(!is.null(forest_dataset$data_ptr))
            # Unpack dimensions
            output_dim <- output_dimension_forest_container_cpp(self$forest_container_ptr)
            n <- dataset_num_rows_cpp(forest_dataset$data_ptr)
            
            # Predict leaf values from forest
            output <- predict_forest_raw_single_forest_cpp(self$forest_container_ptr, forest_dataset$data_ptr, forest_num)
            return(output)
        }, 
        
        #' @description
        #' Return number of samples in a `ForestContainer` object
        #' @return Sample count
        num_samples = function() {
            return(num_samples_forest_container_cpp(self$forest_container_ptr))
        }
    )
)

#' Create a container of forest samples
#'
#' @param num_trees Number of trees
#' @param output_dimension Dimensionality of the outcome model
#' @param is_leaf_constant Whether leaf is constant
#'
#' @return `ForestSamples` object
#' @export
createForestContainer <- function(num_trees, output_dimension=1, is_leaf_constant=F) {
    return(invisible((
        ForestSamples$new(num_trees, output_dimension, is_leaf_constant)
    )))
}
