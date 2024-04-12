#' Run the BART algorithm for supervised learning
#'
#' @param X_train 
#' @param W_train 
#' @param y_train 
#' @param X_test 
#' @param W_test 
#' @param feature_types 
#' @param variable_weights 
#' @param tau_init 
#' @param alpha 
#' @param beta 
#' @param min_samples_leaf 
#' @param leaf_model 
#' @param nu 
#' @param lambda 
#' @param a_leaf 
#' @param b_leaf 
#' @param sigma2_init 
#' @param num_trees 
#' @param num_gfr 
#' @param num_burnin 
#' @param num_mcmc 
#' @param sample_sigma 
#' @param sample_tau 
#' @param random_seed 
#'
#' @return An instance of the `BARTModel` class, sampled according to the provided parameters
#' @export
BART <- function(X_train, y_train, W_train = NULL, X_test = NULL, W_test = NULL, 
                 feature_types = rep(0, ncol(X_train)), 
                 variable_weights = rep(1/ncol(X_train), ncol(X_train)), 
                 cutpoint_grid_size = 100, tau_init = NULL, alpha = 0.95, 
                 beta = 2.0, min_samples_leaf = 5, leaf_model = 0, 
                 nu = 3, lambda = NULL, a_leaf = 3, b_leaf = NULL, 
                 q = 0.9, sigma2_init = NULL, num_trees = 100, num_gfr = 5, 
                 num_burnin = 0, num_mcmc = 100, sample_sigma = T, 
                 sample_tau = T, random_seed = -1){
    # Convert all input data to matrices if not already converted
    if ((is.null(dim(X_train))) && (!is.null(X_train))) {
        X_train <- as.matrix(X_train)
    }
    if ((is.null(dim(W_train))) && (!is.null(W_train))) {
        W_train <- as.matrix(W_train)
    }
    if ((is.null(dim(X_test))) && (!is.null(X_test))) {
        X_test <- as.matrix(X_test)
    }
    if ((is.null(dim(W_test))) && (!is.null(W_test))) {
        W_test <- as.matrix(W_test)
    }
    
    # Convert y_train to numeric vector if not already converted
    if (!is.null(dim(y_train))) {
        y_train <- as.matrix(y_train)
    }
    
    # Determine whether a test set is provided
    has_test = !is.null(X_test)

    # Standardize outcome separately for test and train
    y_bar_train <- mean(y_train)
    y_std_train <- sd(y_train)
    resid_train <- (y_train-y_bar_train)/y_std_train

    # Calibrate priors for sigma^2 and tau
    sigma2hat <- (sigma(lm(resid_train ~ X_train)))^2
    quantile_cutoff <- 0.9
    if (is.null(lambda)) {
        lambda <- (sigma2hat*qgamma(1-quantile_cutoff,nu))/nu
    }
    if (is.null(sigma2_init)) sigma2_init <- sigma2hat
    if (is.null(b_leaf)) b_leaf <- var(resid_train)/(2*num_trees)
    if (is.null(tau_init)) tau_init <- var(resid_train)/(num_trees)
    current_leaf_scale <- as.matrix(tau_init)
    current_sigma2 <- sigma2_init
    
    # Unpack model type info
    if (leaf_model == 0) {
        output_dimension = 1
        is_leaf_constant = T
        leaf_regression = F
    } else if (leaf_model == 1) {
        output_dimension = 1
        is_leaf_constant = F
        leaf_regression = T
    } else if (leaf_model == 2) {
        stopifnot(!is.null(W_train))
        output_dimension = ncol(W_train)
        is_leaf_constant = F
        leaf_regression = T
        if (sample_tau) {
            stop("Sampling leaf scale not yet supported for multivariate leaf models")
        }
    }
    
    # Data
    if (leaf_regression) {
        forest_dataset_train <- createForestDataset(X_train, W_train)
        forest_dataset_test <- createForestDataset(X_test, W_test)
    } else {
        forest_dataset_train <- createForestDataset(X_train)
        forest_dataset_test <- createForestDataset(X_test)
    }
    outcome_train <- createOutcome(resid_train)
    
    # Random number generator (std::mt19937)
    if (is.null(random_seed)) random_seed = sample(1:10000,1,F)
    rng <- createRNG(random_seed)
    
    # Sampling data structures
    feature_types <- as.integer(feature_types)
    forest_model <- createForestModel(forest_dataset_train, feature_types, num_trees, nrow(X_train), alpha, beta, min_samples_leaf)
    
    # Container of forest samples
    forest_samples <- createForestContainer(num_trees, output_dimension, is_leaf_constant)

    # Container of variance parameter samples
    num_samples <- num_gfr + num_burnin + num_mcmc
    if (sample_sigma) global_var_samples <- rep(0, num_samples)
    if (sample_tau) leaf_scale_samples <- rep(0, num_samples)
    
    # Run GFR (warm start) if specified
    if (num_gfr > 0){
        for (i in 1:num_gfr) {
            forest_model$sample_one_iteration(
                forest_dataset_train, outcome_train, forest_samples, rng, feature_types, 
                leaf_model, current_leaf_scale, variable_weights, 
                current_sigma2, cutpoint_grid_size, gfr = T
            )
            if (sample_sigma) {
                global_var_samples[i] <- sample_sigma2_one_iteration(outcome_train, rng, nu, lambda)
                current_sigma2 <- global_var_samples[i]
            }
            if (sample_tau) {
                leaf_scale_samples[i] <- sample_tau_one_iteration(forest_samples, rng, a_leaf, b_leaf, i-1)
                current_leaf_scale <- as.matrix(leaf_scale_samples[i])
            }
        }
    }
    
    # Run MCMC
    for (i in (num_gfr+1):num_samples) {
        forest_model$sample_one_iteration(
            forest_dataset_train, outcome_train, forest_samples, rng, feature_types, 
            leaf_model, current_leaf_scale, variable_weights, 
            current_sigma2, cutpoint_grid_size, gfr = F
        )
        if (sample_sigma) {
            global_var_samples[i] <- sample_sigma2_one_iteration(outcome_train, rng, nu, lambda)
            current_sigma2 <- global_var_samples[i]
        }
        if (sample_tau) {
            leaf_scale_samples[i] <- sample_tau_one_iteration(forest_samples, rng, a_leaf, b_leaf, i-1)
            current_leaf_scale <- as.matrix(leaf_scale_samples[i])
        }
    }
    
    # Forest predictions
    yhat_train <- forest_samples$predict(forest_dataset_train)*y_std_train + y_bar_train
    if (has_test) yhat_test <- forest_samples$predict(forest_dataset_test)*y_std_train + y_bar_train
    
    # Global error variance
    if (sample_sigma) sigma2_samples <- global_var_samples*(y_std_train^2)
    
    # Global error variance
    if (sample_tau) tau_samples <- leaf_scale_samples
    
    # Return results as a list
    model_params <- list(
        "sigma2_init" = sigma2_init, 
        "nu" = nu,
        "lambda" = lambda, 
        "tau_init" = tau_init,
        "a" = a_leaf, 
        "b" = b_leaf,
        "outcome_mean" = y_bar_train,
        "outcome_scale" = y_std_train
    )
    result <- list(
        "forests" = forest_samples, 
        "model_params" = model_params, 
        "yhat_train" = yhat_train
    )
    if (has_test) result[["yhat_test"]] = yhat_test
    if (sample_sigma) result[["sigma2_samples"]] = sigma2_samples
    if (sample_tau) result[["tau_samples"]] = tau_samples
    
    return(result)
}
