#' Run the BART algorithm for supervised learning. 
#'
#' @param X_train Covariates used to split trees in the ensemble.
#' @param W_train (Optional) Bases used to define a regression model `y ~ W` in 
#' each leaf of each regression tree. By default, BART assumes constant leaf node 
#' parameters, implicitly regressing on a constant basis of ones (i.e. `y ~ 1`).
#' @param y_train Outcome to be modeled by the ensemble.
#' @param X_test (Optional) Test set of covariates used to define "out of sample" evaluation data.
#' @param W_test (Optional) Test set of bases used to define "out of sample" evaluation data. 
#' While a test set is optional, the structure of any provided test set must match that 
#' of the training set (i.e. if both X_train and W_train are provided, then a test set must 
#' consist of X_test and W_test with the same number of columns).
#' @param feature_types Vector of length `ncol(X_train)` indicating the "type" of each covariates 
#' (0 = numeric, 1 = ordered categorical, 2 = unordered categorical). Default: `rep(0,ncol(X_train))`.
#' @param variable_weights Vector of length `ncol(X_train)` indicating a "weight" placed on each 
#' variable for sampling purposes. Default: `rep(1/ncol(X_train),ncol(X_train))`.
#' @param cutpoint_grid_size Maximum size of the "grid" of potential cutpoints to consider. Default: 100.
#' @param tau_init Starting value of leaf node scale parameter. Calibrated internally as 1/num_trees if not set here.
#' @param alpha Prior probability of splitting for a tree of depth 0. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`.
#' @param beta Exponent that decreases split probabilities for nodes of depth > 0. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`.
#' @param min_samples_leaf Minimum allowable size of a leaf, in terms of training samples.
#' @param leaf_model Integer indicating leaf model, where 0 = constant with Gaussian prior, 1 = univariate regression with Gaussian prior, 2 = multivariate regression with Gaussian prior. W_train will be ignored if this is set to 0. Default: 0.
#' @param nu Shape parameter in the `IG(nu, nu*lambda)` global error variance model. Default: 3.
#' @param lambda Component of the scale parameter in the `IG(nu, nu*lambda)` global error variance prior. If not specified, this is calibrated as in Sparapani et al (2021).
#' @param a_leaf Shape parameter in the `IG(a_leaf, b_leaf)` leaf node parameter variance model. Default: 3.
#' @param b_leaf Scale parameter in the `IG(a_leaf, b_leaf)` leaf node parameter variance model. Calibrated internally as 0.5/num_trees if not set here.
#' @param q Quantile used to calibrated `lambda` as in Sparapani et al (2021). Default: 0.9.
#' @param sigma2_init Starting value of global variance parameter. Calibrated internally as in Sparapani et al (2021) if not set here.
#' @param num_trees Number of trees in the ensemble. Default: 100.
#' @param num_gfr Number of "warm-start" iterations run using the grow-from-root algorithm (He and Hahn, 2021). Default: 5.
#' @param num_burnin Number of "burn-in" iterations of the MCMC sampler. Default: 0.
#' @param num_mcmc Number of "retained" iterations of the MCMC sampler. Default: 100.
#' @param sample_sigma Whether or not to update the `sigma^2` global error variance parameter based on `IG(nu, nu*lambda)`. Default: T.
#' @param sample_tau Whether or not to update the `tau` leaf scale variance parameter based on `IG(a_leaf, b_leaf)`. Cannot be set to true if `leaf_model=2`. Default: T.
#' @param random_seed Integer parameterizing the C++ random number generator. If not specified, the C++ random number generator is seeded according to `std::random_device`.
#'
#' @return List of sampling outputs and a wrapper around the sampled forests (which can be used for in-memory prediction on new data, or serialized to JSON on disk).
#' @export
#'
#' @examples
#' n <- 100
#' p <- 5
#' X <- matrix(runif(n*p), ncol = p)
#' f_XW <- (
#'     ((0 <= X[,1]) & (0.25 > X[,1])) * (-7.5) + 
#'     ((0.25 <= X[,1]) & (0.5 > X[,1])) * (-2.5) + 
#'     ((0.5 <= X[,1]) & (0.75 > X[,1])) * (2.5) + 
#'     ((0.75 <= X[,1]) & (1 > X[,1])) * (7.5)
#' )
#' noise_sd <- 1
#' y <- f_XW + rnorm(n, 0, noise_sd)
#' test_set_pct <- 0.2
#' n_test <- round(test_set_pct*n)
#' n_train <- n - n_test
#' test_inds <- sort(sample(1:n, n_test, replace = F))
#' train_inds <- (1:n)[!((1:n) %in% test_inds)]
#' X_test <- X[test_inds,]
#' X_train <- X[train_inds,]
#' y_test <- y[test_inds]
#' y_train <- y[train_inds]
#' bart_model <- bart(X_train = X_train, y_train = y_train, X_test = X_test, leaf_model = 0)
bart <- function(X_train, y_train, W_train = NULL, X_test = NULL, W_test = NULL, 
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
    
    # Determine whether a basis vector is provided
    has_basis = !is.null(W_train)
    
    # Determine whether a test set is provided
    has_test = !is.null(X_test)

    # Standardize outcome separately for test and train
    y_bar_train <- mean(y_train)
    y_std_train <- sd(y_train)
    resid_train <- (y_train-y_bar_train)/y_std_train

    # Calibrate priors for sigma^2 and tau
    reg_basis <- cbind(W_train, X_train)
    sigma2hat <- (sigma(lm(resid_train~reg_basis)))^2
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


