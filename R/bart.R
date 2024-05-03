#' Run the BART algorithm for supervised learning. 
#'
#' @param X_train Covariates used to split trees in the ensemble.
#' @param y_train Outcome to be modeled by the ensemble.
#' @param W_train (Optional) Bases used to define a regression model `y ~ W` in 
#' each leaf of each regression tree. By default, BART assumes constant leaf node 
#' parameters, implicitly regressing on a constant basis of ones (i.e. `y ~ 1`).
#' @param group_ids_train (Optional) Group labels used for an additive random effects model.
#' @param rfx_basis_train (Optional) Basis for "random-slope" regression in an additive random effects model.
#' If `group_ids_train` is provided with a regression basis, an intercept-only random effects model 
#' will be estimated.
#' @param X_test (Optional) Test set of covariates used to define "out of sample" evaluation data.
#' @param W_test (Optional) Test set of bases used to define "out of sample" evaluation data. 
#' While a test set is optional, the structure of any provided test set must match that 
#' of the training set (i.e. if both X_train and W_train are provided, then a test set must 
#' consist of X_test and W_test with the same number of columns).
#' @param group_ids_test (Optional) Test set group labels used for an additive random effects model. 
#' We do not currently support (but plan to in the near future), test set evaluation for group labels
#' that were not in the training set.
#' @param rfx_basis_test (Optional) Test set basis for "random-slope" regression in additive random effects model.
#' @param feature_types Vector of length `ncol(X_train)` indicating the "type" of each covariates 
#' (0 = numeric, 1 = ordered categorical, 2 = unordered categorical). Default: `rep(0,ncol(X_train))`.
#' @param variable_weights Vector of length `ncol(X_train)` indicating a "weight" placed on each 
#' variable for sampling purposes. Default: `rep(1/ncol(X_train),ncol(X_train))`.
#' @param cutpoint_grid_size Maximum size of the "grid" of potential cutpoints to consider. Default: 100.
#' @param tau_init Starting value of leaf node scale parameter. Calibrated internally as `1/num_trees` if not set here.
#' @param alpha Prior probability of splitting for a tree of depth 0. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`.
#' @param beta Exponent that decreases split probabilities for nodes of depth > 0. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`.
#' @param min_samples_leaf Minimum allowable size of a leaf, in terms of training samples. Default: 5.
#' @param leaf_model Integer indicating leaf model, where 0 = constant with Gaussian prior, 1 = univariate regression with Gaussian prior, 2 = multivariate regression with Gaussian prior. W_train will be ignored if this is set to 0. Default: 0.
#' @param nu Shape parameter in the `IG(nu, nu*lambda)` global error variance model. Default: 3.
#' @param lambda Component of the scale parameter in the `IG(nu, nu*lambda)` global error variance prior. If not specified, this is calibrated as in Sparapani et al (2021).
#' @param a_leaf Shape parameter in the `IG(a_leaf, b_leaf)` leaf node parameter variance model. Default: 3.
#' @param b_leaf Scale parameter in the `IG(a_leaf, b_leaf)` leaf node parameter variance model. Calibrated internally as 0.5/num_trees if not set here.
#' @param q Quantile used to calibrated `lambda` as in Sparapani et al (2021). Default: 0.9.
#' @param sigma2_init Starting value of global variance parameter. Calibrated internally as in Sparapani et al (2021) if not set here.
#' @param num_trees Number of trees in the ensemble. Default: 200.
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
#' # plot(rowMeans(bart_model$yhat_test), y_test, xlab = "predicted", ylab = "actual")
#' # abline(0,1,col="red",lty=3,lwd=3)
bart <- function(X_train, y_train, W_train = NULL, group_ids_train = NULL, 
                 rfx_basis_train = NULL, X_test = NULL, W_test = NULL, 
                 group_ids_test = NULL, rfx_basis_test = NULL, 
                 feature_types = rep(0, ncol(X_train)), 
                 variable_weights = rep(1/ncol(X_train), ncol(X_train)), 
                 cutpoint_grid_size = 100, tau_init = NULL, alpha = 0.95, 
                 beta = 2.0, min_samples_leaf = 5, leaf_model = 0, 
                 nu = 3, lambda = NULL, a_leaf = 3, b_leaf = NULL, 
                 q = 0.9, sigma2_init = NULL, num_trees = 200, num_gfr = 5, 
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
    if ((is.null(dim(rfx_basis_train))) && (!is.null(rfx_basis_train))) {
        rfx_basis_train <- as.matrix(rfx_basis_train)
    }
    if ((is.null(dim(rfx_basis_test))) && (!is.null(rfx_basis_test))) {
        rfx_basis_test <- as.matrix(rfx_basis_test)
    }
    
    # Recode group IDs to integer vector (if passed as, for example, a vector of county names, etc...)
    has_rfx <- F
    has_rfx_test <- F
    if (!is.null(group_ids_train)) {
        group_ids_factor <- factor(group_ids_train)
        group_ids_train <- as.integer(group_ids_factor)
        has_rfx <- T
        if (!is.null(group_ids_test)) {
            group_ids_factor_test <- factor(group_ids_test, levels = levels(group_ids_factor))
            if (sum(is.na(group_ids_factor_test)) > 0) {
                stop("All random effect group labels provided in group_ids_test must be present in group_ids_train")
            }
            group_ids_test <- as.integer(group_ids_factor_test)
            has_rfx_test <- T
        }
    }
    
    # Data consistency checks
    if ((!is.null(X_test)) && (ncol(X_test) != ncol(X_train))) {
        stop("X_train and X_test must have the same number of columns")
    }
    if ((!is.null(W_test)) && (ncol(W_test) != ncol(W_train))) {
        stop("W_train and W_test must have the same number of columns")
    }
    if ((!is.null(W_train)) && (nrow(W_train) != nrow(X_train))) {
        stop("W_train and X_train must have the same number of rows")
    }
    if ((!is.null(W_test)) && (nrow(W_test) != nrow(X_test))) {
        stop("W_test and X_test must have the same number of rows")
    }
    if (nrow(X_train) != length(y_train)) {
        stop("X_train and y_train must have the same number of observations")
    }
    if ((!is.null(rfx_basis_test)) && (ncol(rfx_basis_test) != ncol(rfx_basis_train))) {
        stop("rfx_basis_train and rfx_basis_test must have the same number of columns")
    }
    if (!is.null(group_ids_train)) {
        if (!is.null(group_ids_test)) {
            if ((!is.null(rfx_basis_train)) && (is.null(rfx_basis_test))) {
                stop("rfx_basis_train is provided but rfx_basis_test is not provided")
            }
        }
    }
    
    # Fill in rfx basis as a vector of 1s (random intercept) if a basis not provided 
    if (has_rfx) {
        if (is.null(rfx_basis_train)) {
            rfx_basis_train <- matrix(rep(1,nrow(X_train)), nrow = nrow(X_train), ncol = 1)
        }
        num_rfx_groups <- length(unique(group_ids_train))
        num_rfx_components <- ncol(rfx_basis_train)
    }
    if (has_rfx_test) {
        if (is.null(rfx_basis_test)) {
            rfx_basis_test <- matrix(rep(1,nrow(X_test)), nrow = nrow(X_test), ncol = 1)
        }
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
        stopifnot(has_basis)
        stopifnot(ncol(W_train) == 1)
        output_dimension = 1
        is_leaf_constant = F
        leaf_regression = T
    } else if (leaf_model == 2) {
        stopifnot(has_basis)
        stopifnot(ncol(W_train) > 1)
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
        if (has_test) forest_dataset_test <- createForestDataset(X_test, W_test)
        requires_basis <- T
    } else {
        forest_dataset_train <- createForestDataset(X_train)
        if (has_test) forest_dataset_test <- createForestDataset(X_test)
        requires_basis <- F
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
    
    # Random effects prior parameters
    if (has_rfx) {
        if (num_rfx_components == 1) {
            alpha_init <- c(1)
        } else if (num_rfx_components > 1) {
            alpha_init <- c(1,rep(0,num_rfx_components-1))
        } else {
            stop("There must be at least 1 random effect component")
        }
        xi_init <- matrix(rep(alpha_init, num_rfx_groups),num_rfx_components,num_rfx_groups)
        sigma_alpha_init <- diag(1,num_rfx_components,num_rfx_components)
        sigma_xi_init <- diag(1,num_rfx_components,num_rfx_components)
        sigma_xi_shape <- 1
        sigma_xi_scale <- 1
    }

    # Random effects data structure and storage container
    if (has_rfx) {
        rfx_dataset_train <- createRandomEffectsDataset(group_ids_train, rfx_basis_train)
        rfx_tracker_train <- createRandomEffectsTracker(group_ids_train)
        rfx_model <- createRandomEffectsModel(num_rfx_components, num_rfx_groups)
        rfx_model$set_working_parameter(alpha_init)
        rfx_model$set_group_parameters(xi_init)
        rfx_model$set_working_parameter_cov(sigma_alpha_init)
        rfx_model$set_group_parameter_cov(sigma_xi_init)
        rfx_model$set_variance_prior_shape(sigma_xi_shape)
        rfx_model$set_variance_prior_scale(sigma_xi_scale)
        rfx_samples <- createRandomEffectSamples(num_rfx_components, num_rfx_groups, rfx_tracker_train)
    }

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
                current_sigma2, cutpoint_grid_size, gfr = T, pre_initialized = F
            )
            if (sample_sigma) {
                global_var_samples[i] <- sample_sigma2_one_iteration(outcome_train, rng, nu, lambda)
                current_sigma2 <- global_var_samples[i]
            }
            if (sample_tau) {
                leaf_scale_samples[i] <- sample_tau_one_iteration(forest_samples, rng, a_leaf, b_leaf, i-1)
                current_leaf_scale <- as.matrix(leaf_scale_samples[i])
            }
            if (has_rfx) {
                rfx_model$sample_random_effect(rfx_dataset_train, outcome_train, rfx_tracker_train, rfx_samples, current_sigma2, rng)
            }
        }
    }
    
    # Run MCMC
    if (num_burnin + num_mcmc > 0) {
        for (i in (num_gfr+1):num_samples) {
            forest_model$sample_one_iteration(
                forest_dataset_train, outcome_train, forest_samples, rng, feature_types, 
                leaf_model, current_leaf_scale, variable_weights, 
                current_sigma2, cutpoint_grid_size, gfr = F, pre_initialized = F
            )
            if (sample_sigma) {
                global_var_samples[i] <- sample_sigma2_one_iteration(outcome_train, rng, nu, lambda)
                current_sigma2 <- global_var_samples[i]
            }
            if (sample_tau) {
                leaf_scale_samples[i] <- sample_tau_one_iteration(forest_samples, rng, a_leaf, b_leaf, i-1)
                current_leaf_scale <- as.matrix(leaf_scale_samples[i])
            }
            if (has_rfx) {
                rfx_model$sample_random_effect(rfx_dataset_train, outcome_train, rfx_tracker_train, rfx_samples, current_sigma2, rng)
            }
        }
    }
    
    # Forest predictions
    yhat_train <- forest_samples$predict(forest_dataset_train)*y_std_train + y_bar_train
    if (has_test) yhat_test <- forest_samples$predict(forest_dataset_test)*y_std_train + y_bar_train
    
    # Random effects predictions
    if (has_rfx) {
        rfx_preds_train <- rfx_samples$predict(group_ids_train, rfx_basis_train)*y_std_train
        yhat_train <- yhat_train + rfx_preds_train
    }
    if ((has_rfx_test) && (has_test)) {
        rfx_preds_test <- rfx_samples$predict(group_ids_test, rfx_basis_test)*y_std_train
        yhat_test <- yhat_test + rfx_preds_test
    }
    
    # Global error variance
    if (sample_sigma) sigma2_samples <- global_var_samples*(y_std_train^2)
    
    # Leaf parameter variance
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
        "outcome_scale" = y_std_train, 
        "output_dimension" = output_dimension,
        "is_leaf_constant" = is_leaf_constant,
        "leaf_regression" = leaf_regression,
        "requires_basis" = requires_basis, 
        "num_covariates" = ncol(X_train), 
        "num_basis" = ifelse(is.null(W_train),0,ncol(W_train)), 
        "num_samples" = num_samples
    )
    result <- list(
        "forests" = forest_samples, 
        "model_params" = model_params, 
        "yhat_train" = yhat_train
    )
    if (has_test) result[["yhat_test"]] = yhat_test
    if (sample_sigma) result[["sigma2_samples"]] = sigma2_samples
    if (sample_tau) result[["tau_samples"]] = tau_samples
    if (has_rfx) {
        result[["rfx_samples"]] = rfx_samples
        result[["rfx_preds_train"]] = rfx_preds_train
    }
    if ((has_rfx_test) && (has_test)) result[["rfx_preds_test"]] = rfx_preds_test
    class(result) <- "bartmodel"
    
    # Clean up classes with external pointers to C++ data structures
    rm(forest_model)
    rm(forest_dataset_train)
    if (has_test) rm(forest_dataset_test)
    if (has_rfx) rm(rfx_dataset_train, rfx_tracker_train, rfx_model)
    rm(outcome_train)
    rm(rng)
    
    return(result)
}


#' Predict from a sampled BART model on new data
#'
#' @param bart Object of type `bart` containing draws of a regression forest and associated sampling outputs.
#' @param X_test Covariates used to determine tree leaf predictions for each observation.
#' @param W_test (Optional) Bases used for prediction (by e.g. dot product with leaf values). Default: `NULL`.
#'
#' @return Matrix of predictions with `nrow(X_test)` rows and `bart$num_samples` columns
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
#' bart_model <- bart(X_train = X_train, y_train = y_train, leaf_model = 0)
#' yhat_test <- predict(bart_model, X_test)
#' # plot(rowMeans(yhat_test), y_test, xlab = "predicted", ylab = "actual")
#' # abline(0,1,col="red",lty=3,lwd=3)
predict.bartmodel <- function(bart, X_test, W_test = NULL){
    # Convert all input data to matrices if not already converted
    if ((is.null(dim(X_test))) && (!is.null(X_test))) {
        X_test <- as.matrix(X_test)
    }
    if ((is.null(dim(W_test))) && (!is.null(W_test))) {
        W_test <- as.matrix(W_test)
    }
    
    # Data checks
    if ((bart$model_params$requires_basis) && (is.null(W_test))) {
        stop("Basis (W_test) must be provided for this model")
    }
    if ((!is.null(W_test)) && (nrow(X_test) != nrow(W_test))) {
        stop("X_test and W_test must have the same number of rows")
    }
    if (bart$model_params$num_covariates != ncol(X_test)) {
        stop("X_test and W_test must have the same number of rows")
    }
    
    # Create prediction dataset
    if (!is.null(W_test)) prediction_dataset <- createForestDataset(X_test, W_test)
    else prediction_dataset <- createForestDataset(X_test)
    
    # Compute and return predictions
    y_std <- bart$model_params$outcome_scale
    y_bar <- bart$model_params$outcome_mean
    result <- bart$forests$predict(prediction_dataset)*y_std + y_bar
    return(result)
}
