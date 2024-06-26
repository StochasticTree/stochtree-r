% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/utils.R
\name{orderedCatInitializeAndPreprocess}
\alias{orderedCatInitializeAndPreprocess}
\title{Run some simple preprocessing of ordered categorical variables, converting
ordered levels to integers if necessary, and storing the unique levels of a
variable.}
\usage{
orderedCatInitializeAndPreprocess(x_input)
}
\arguments{
\item{x_input}{Vector of ordered categorical data. If the data is not already
stored as an ordered factor, it will be converted to one using the default
sort order.}
}
\value{
List containing a preprocessed vector of integer-converted ordered
categorical observations and the unique level of the original ordered
categorical feature.
}
\description{
Run some simple preprocessing of ordered categorical variables, converting
ordered levels to integers if necessary, and storing the unique levels of a
variable.
}
\examples{
x <- c("1. Strongly disagree", "3. Neither agree nor disagree", "2. Disagree", "4. Agree", "3. Neither agree nor disagree", "5. Strongly agree", "4. Agree")
preprocess_list <- orderedCatInitializeAndPreprocess(x)
x_preprocessed <- preprocess_list$x_preprocessed
}
