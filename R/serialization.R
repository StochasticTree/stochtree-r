#' Class that stores draws from an random ensemble of decision trees
#'
#' @description
#' Wrapper around a C++ container of tree ensembles

CppJson <- R6::R6Class(
    classname = "CppJson",
    cloneable = FALSE,
    public = list(
        
        #' @field json_ptr External pointer to a C++ nlohmann::json object
        json_ptr = NULL,
        
        #' @field num_forests Number of forests in the nlohmann::json object
        num_forests = NULL,
        
        #' @field forest_labels Names of forest objects in the overall nlohmann::json object
        forest_labels = NULL,
        
        #' @field num_rfx Number of random effects terms in the nlohman::json object
        num_rfx = NULL,
        
        #' @field rfx_container_labels Names of rfx container objects in the overall nlohmann::json object
        rfx_container_labels = NULL,
        
        #' @field rfx_mapper_labels Names of rfx label mapper objects in the overall nlohmann::json object
        rfx_mapper_labels = NULL,
        
        #' @field rfx_groupid_labels Names of rfx group id objects in the overall nlohmann::json object
        rfx_groupid_labels = NULL,
        
        #' @description
        #' Create a new CppJson object.
        #' @return A new `CppJson` object.
        initialize = function() {
            self$json_ptr <- init_json_cpp()
            self$num_forests <- 0
            self$forest_labels <- c()
            self$num_rfx <- 0
            self$rfx_container_labels <- c()
            self$rfx_mapper_labels <- c()
            self$rfx_groupid_labels <- c()
        }, 
        
        #' @description
        #' Convert a forest container to json and add to the current `CppJson` object
        #' @param forest_samples `ForestSamples` R class
        #' @return NULL
        add_forest = function(forest_samples) {
            forest_label <- json_add_forest_cpp(self$json_ptr, forest_samples$forest_container_ptr)
            self$num_forests <- self$num_forests + 1
            self$forest_labels <- c(self$forest_labels, forest_label)
        }, 
        
        #' @description
        #' Convert a random effects container to json and add to the current `CppJson` object
        #' @param rfx_samples `RandomEffectSamples` R class
        #' @return NULL
        add_random_effects = function(rfx_samples) {
            rfx_container_label <- json_add_rfx_container_cpp(self$json_ptr, rfx_samples$rfx_container_ptr)
            self$rfx_container_labels <- c(self$rfx_container_labels, rfx_container_label)
            rfx_mapper_label <- json_add_rfx_label_mapper_cpp(self$json_ptr, rfx_samples$label_mapper_ptr)
            self$rfx_mapper_labels <- c(self$rfx_mapper_labels, rfx_mapper_label)
            rfx_groupid_label <- json_add_rfx_groupids_cpp(self$json_ptr, rfx_samples$training_group_ids)
            self$rfx_groupid_labels <- c(self$rfx_groupid_labels, rfx_groupid_label)
            json_increment_rfx_count_cpp(self$json_ptr)
            self$num_rfx <- self$num_rfx + 1
        }, 
        
        #' @description
        #' Save a json object to file
        #' @param filename String of filepath, must end in ".json"
        #' @return NULL
        save_file = function(filename) {
            json_save_cpp(self$json_ptr, filename)
        }
    )
)

#' Create a C++ Json object
#'
#' @return `CppJson` object
#' @export
createCppJson <- function() {
    return(invisible((
        CppJson$new()
    )))
}
