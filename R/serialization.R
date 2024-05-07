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
        
        #' @description
        #' Create a new CppJson object.
        #' @return A new `CppJson` object.
        initialize = function() {
            self$json_ptr <- init_json_cpp()
        }, 
        
        #' @description
        #' Convert a forest container to json and add to the current `CppJson` object
        #' @param forest_samples `ForestSamples` R class
        #' @return NULL
        add_forest = function(forest_samples) {
            json_add_forest_cpp(self$json_ptr, forest_samples$forest_container_ptr)
        }, 
        
        #' @description
        #' Convert a random effects container to json and add to the current `CppJson` object
        #' @param rfx_samples `RandomEffectSamples` R class
        #' @return NULL
        add_random_effects = function(rfx_samples) {
            json_add_rfx_cpp(self$json_ptr, rfx_samples$rfx_container_ptr)
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
