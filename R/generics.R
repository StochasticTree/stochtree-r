#' Generic function for extracting random effect samples from a model object (BCF, BART, etc...)
#' 
#' @return List of random effect samples
#' @export
getRandomEffectSamples <- function(object, ...) UseMethod("getRandomEffectSamples")

#' Convert a model object (BCF, BART, etc...) to JSON
#' 
#' @return Object of type `CppJson` which can be saved to disk with the `save_file(filename)` 
#' method
#' @export
convertToJson <- function(object, ...) UseMethod("convertToJson")

#' Convert a model object (BCF, BART, etc...) to JSON and save it to a file 
#' with a json suffix named `filename`
#' 
#' @return NULL
#' @export
saveToJsonFile <- function(object, filename, ...) UseMethod("saveToJsonFile")
