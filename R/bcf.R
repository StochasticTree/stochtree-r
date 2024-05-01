#' Run the Bayesian Causal Forest (BCF) algorithm for regularized causal effect estimation. 
#'
#' @param X_train Covariates used to split trees in the ensemble.
#' @param Z_train Vector of (continuous or binary) treatment assignments.
#' @param y_train Outcome to be modeled by the ensemble.
#' @param pi_train (Optional) Vector of propensity scores. If not provided, this will be estimated from the data.
#' @param X_test (Optional) Test set of covariates used to define "out of sample" evaluation data.
#' @param Z_test (Optional) Test set of (continuous or binary) treatment assignments.
#' @param pi_test (Optional) Vector of propensity scores. If not provided, this will be estimated from the data.
#' @param feature_types Vector of length `ncol(X_train)` indicating the "type" of each covariates 
#' (0 = numeric, 1 = ordered categorical, 2 = unordered categorical). Default: `rep(0,ncol(X_train))`.
#' @param cutpoint_grid_size Maximum size of the "grid" of potential cutpoints to consider. Default: 100.
#' @param sigma_leaf_mu Starting value of leaf node scale parameter for the prognostic forest. Calibrated internally as `2/num_trees_mu` if not set here.
#' @param sigma_leaf_tau Starting value of leaf node scale parameter for the treatment effect forest. Calibrated internally as `1/num_trees_tau` if not set here.
#' @param alpha_mu Prior probability of splitting for a tree of depth 0 for the prognostic forest. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Default: 0.95.
#' @param alpha_tau Prior probability of splitting for a tree of depth 0 for the treatment effect forest. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Default: 0.25.
#' @param beta_mu Exponent that decreases split probabilities for nodes of depth > 0 for the prognostic forest. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Default: 2.0.
#' @param beta_tau Exponent that decreases split probabilities for nodes of depth > 0 for the treatment effect forest. Tree split prior combines `alpha` and `beta` via `alpha*(1+node_depth)^-beta`. Default: 3.0.
#' @param min_samples_leaf_mu Minimum allowable size of a leaf, in terms of training samples, for the prognostic forest. Default: 5.
#' @param min_samples_leaf_tau Minimum allowable size of a leaf, in terms of training samples, for the treatment effect forest. Default: 5.
#' @param nu Shape parameter in the `IG(nu, nu*lambda)` global error variance model. Default: 3.
#' @param lambda Component of the scale parameter in the `IG(nu, nu*lambda)` global error variance prior. If not specified, this is calibrated as in Sparapani et al (2021).
#' @param a_leaf_mu Shape parameter in the `IG(a_leaf, b_leaf)` leaf node parameter variance model for the prognostic forest. Default: 3.
#' @param a_leaf_tau Shape parameter in the `IG(a_leaf, b_leaf)` leaf node parameter variance model for the treatment effect forest. Default: 3.
#' @param b_leaf_mu Scale parameter in the `IG(a_leaf, b_leaf)` leaf node parameter variance model for the prognostic forest. Calibrated internally as 0.5/num_trees if not set here.
#' @param b_leaf_tau Scale parameter in the `IG(a_leaf, b_leaf)` leaf node parameter variance model for the treatment effect forest. Calibrated internally as 0.5/num_trees if not set here.
#' @param q Quantile used to calibrated `lambda` as in Sparapani et al (2021). Default: 0.9.
#' @param sigma2 Starting value of global variance parameter. Calibrated internally as in Sparapani et al (2021) if not set here.
#' @param num_trees_mu Number of trees in the prognostic forest. Default: 200.
#' @param num_trees_tau Number of trees in the treatment effect forest. Default: 50.
#' @param num_gfr Number of "warm-start" iterations run using the grow-from-root algorithm (He and Hahn, 2021). Default: 5.
#' @param num_burnin Number of "burn-in" iterations of the MCMC sampler. Default: 0.
#' @param num_mcmc Number of "retained" iterations of the MCMC sampler. Default: 100.
#' @param sample_sigma_global Whether or not to update the `sigma^2` global error variance parameter based on `IG(nu, nu*lambda)`. Default: T.
#' @param sample_sigma_leaf_mu Whether or not to update the `sigma_leaf_mu` leaf scale variance parameter in the prognostic forest based on `IG(a_leaf_mu, b_leaf_mu)`. Default: T.
#' @param sample_sigma_leaf_tau Whether or not to update the `sigma_leaf_tau` leaf scale variance parameter in the treatment effect forest based on `IG(a_leaf_tau, b_leaf_tau)`. Default: T.
#' @param propensity_covariate Whether to include the propensity score as a covariate in either or both of the forests. Enter "none" for neither, "mu" for the prognostic forest, "tau" for the treatment forest, and "both" for both forests. If this is not "none" and a propensity score is not provided, it will be estimated from (`X_train`, `Z_train`) using `xgboost`. Default: "mu".
#' @param adaptive_coding Whether or not to use an "adaptive coding" scheme in which a binary treatment variable is not coded manually as (0,1) or (-1,1) but learned via parameters `b_0` and `b_1` that attach to the outcome model `[b_0 (1-Z) + b_1 Z] tau(X)`. This is ignored when Z is not binary. Default: T.
#' @param b_0 Initial value of the "control" group coding parameter. This is ignored when Z is not binary. Default: -0.5.
#' @param b_1 Initial value of the "treatment" group coding parameter. This is ignored when Z is not binary. Default: 0.5.
#' @param random_seed Integer parameterizing the C++ random number generator. If not specified, the C++ random number generator is seeded according to `std::random_device`.
#'
#' @return List of sampling outputs and a wrapper around the sampled forests (which can be used for in-memory prediction on new data, or serialized to JSON on disk).
#' @export
#'
#' @examples
#' n <- 500
#' x1 <- rnorm(n)
#' x2 <- rnorm(n)
#' x3 <- rnorm(n)
#' x4 <- as.numeric(rbinom(n,1,0.5))
#' x5 <- as.numeric(sample(1:3,n,replace=T))
#' X <- cbind(x1,x2,x3,x4,x5)
#' p <- ncol(X)
#' g <- function(x) {ifelse(x[,5]==1,2,ifelse(x[,5]==2,-1,4))}
#' mu1 <- function(x) {1+g(x)+x[,1]*x[,3]}
#' mu2 <- function(x) {1+g(x)+6*abs(x[,3]-1)}
#' tau1 <- function(x) {rep(3,nrow(x))}
#' tau2 <- function(x) {1+2*x[,2]*x[,4]}
#' mu_x <- mu1(X)
#' tau_x <- tau2(X)
#' pi_x <- 0.8*pnorm((3*mu_x/sd(mu_x)) - 0.5*X[,1]) + 0.05 + runif(n)/10
#' Z <- rbinom(n,1,pi_x)
#' E_XZ <- mu_x + Z*tau_x
#' snr <- 4
#' y <- E_XZ + rnorm(n, 0, 1)*(sd(E_XZ)/snr)
#' test_set_pct <- 0.2
#' n_test <- round(test_set_pct*n)
#' n_train <- n - n_test
#' test_inds <- sort(sample(1:n, n_test, replace = F))
#' train_inds <- (1:n)[!((1:n) %in% test_inds)]
#' X_test <- X[test_inds,]
#' X_train <- X[train_inds,]
#' pi_test <- pi_x[test_inds]
#' pi_train <- pi_x[train_inds]
#' Z_test <- Z[test_inds]
#' Z_train <- Z[train_inds]
#' y_test <- y[test_inds]
#' y_train <- y[train_inds]
#' mu_test <- mu_x[test_inds]
#' mu_train <- mu_x[train_inds]
#' tau_test <- tau_x[test_inds]
#' tau_train <- tau_x[train_inds]
#' bcf_model <- bcf(X_train = X_train, Z_train = Z_train, y_train = y_train, pi_train = pi_train, 
#'                  X_test = X_test, Z_test = Z_test, pi_test = pi_test)
#' # plot(rowMeans(bcf_model$mu_hat_test), mu_test, xlab = "predicted", ylab = "actual", main = "Prognostic function")
#' # abline(0,1,col="red",lty=3,lwd=3)
#' # plot(rowMeans(bcf_model$tau_hat_test), tau_test, xlab = "predicted", ylab = "actual", main = "Treatment effect")
#' # abline(0,1,col="red",lty=3,lwd=3)
bcf <- function(X_train, Z_train, y_train, pi_train = NULL, X_test = NULL, Z_test = NULL, pi_test = NULL, 
                feature_types = rep(0, ncol(X_train)), cutpoint_grid_size = 100, 
                sigma_leaf_mu = NULL, sigma_leaf_tau = NULL, alpha_mu = 0.95, alpha_tau = 0.25, 
                beta_mu = 2.0, beta_tau = 3.0, min_samples_leaf_mu = 5, min_samples_leaf_tau = 5, 
                nu = 3, lambda = NULL, a_leaf_mu = 3, a_leaf_tau = 3, b_leaf_mu = NULL, b_leaf_tau = NULL, 
                q = 0.9, sigma2 = NULL, num_trees_mu = 250, num_trees_tau = 50, num_gfr = 5, 
                num_burnin = 0, num_mcmc = 100, sample_sigma_global = T, sample_sigma_leaf_mu = T, 
                sample_sigma_leaf_tau = T, propensity_covariate = "mu", adaptive_coding = T,
                b_0 = -0.5, b_1 = 0.5, random_seed = -1) {
    # Convert all input data to matrices if not already converted
    if ((is.null(dim(X_train))) && (!is.null(X_train))) {
        X_train <- as.matrix(X_train)
    }
    if ((is.null(dim(Z_train))) && (!is.null(Z_train))) {
        Z_train <- as.matrix(as.numeric(Z_train))
    }
    if ((is.null(dim(pi_train))) && (!is.null(pi_train))) {
        pi_train <- as.matrix(pi_train)
    }
    if ((is.null(dim(X_test))) && (!is.null(X_test))) {
        X_test <- as.matrix(X_test)
    }
    if ((is.null(dim(Z_test))) && (!is.null(Z_test))) {
        Z_test <- as.matrix(as.numeric(Z_test))
    }
    if ((is.null(dim(pi_test))) && (!is.null(pi_test))) {
        pi_test <- as.matrix(pi_test)
    }
    
    # Data consistency checks
    if ((!is.null(X_test)) && (ncol(X_test) != ncol(X_train))) {
        stop("X_train and X_test must have the same number of columns")
    }
    if ((!is.null(Z_test)) && (ncol(Z_test) != ncol(Z_train))) {
        stop("Z_train and Z_test must have the same number of columns")
    }
    if ((!is.null(Z_train)) && (nrow(Z_train) != nrow(X_train))) {
        stop("Z_train and X_train must have the same number of rows")
    }
    if ((!is.null(pi_train)) && (nrow(pi_train) != nrow(X_train))) {
        stop("pi_train and X_train must have the same number of rows")
    }
    if ((!is.null(Z_test)) && (nrow(Z_test) != nrow(X_test))) {
        stop("Z_test and X_test must have the same number of rows")
    }
    if ((!is.null(pi_test)) && (nrow(pi_test) != nrow(X_test))) {
        stop("pi_test and X_test must have the same number of rows")
    }
    if (nrow(X_train) != length(y_train)) {
        stop("X_train and y_train must have the same number of observations")
    }

    # Determine whether a test set is provided
    has_test = !is.null(X_test)
    
    # Convert y_train to numeric vector if not already converted
    if (!is.null(dim(y_train))) {
        y_train <- as.matrix(y_train)
    }
    
    # Check whether treatment is binary
    binary_treatment <- length(unique(Z_train)) == 2
    
    # Adaptive coding will be ignored for continuous / ordered categorical treatments
    if ((!binary_treatment) && (adaptive_coding)) {
        adaptive_coding <- F
    }
    
    # Estimate if pre-estimated propensity score is not provided
    if ((is.null(pi_train)) && (propensity_covariate != "none")) {
        # Estimate using xgboost with some elementary hyperparameter tuning
        dtrain <- xgboost::xgb.DMatrix(X_train, label = Z_train)
        if (binary_treatment) {
            cv <- xgboost::xgb.cv(data = dtrain, nrounds = 100, nfold = 5, metrics = list("rmse","auc"), 
                                  max_depth = 3, eta = 1, objective = "binary:logistic")
            num_trees = c(20, 50, 100)
            etas <- c(0.01, 0.05, 0.1, 0.5)
            max_depth <- c(3, 6)
            eval_grid <- expand.grid(num_trees, etas, max_depth)
            evals <- rep(NA, nrow(eval_grid))
            for (i in 1:nrow(eval_grid)) {
                cv <- xgboost::xgb.cv(data = dtrain, nrounds = 100, nfold = 5, metrics = list("rmse","auc"), 
                                      max_depth = 3, eta = 0.1, objective = "binary:logistic", verbose = 0)
                evals[i] <- cv$evaluation_log[length(cv$evaluation_log),"test_auc_mean"]$test_auc_mean
            }
            best_params <- as.numeric(eval_grid[which.min(evals),])
            param <- list(max_depth = best_params[3], eta = best_params[2], verbose = 0, objective = "binary:logistic", eval_metric = "rmse")
            bst <- xgboost::xgb.train(param, dtrain, nrounds = best_params[1])
            pi_train <- as.matrix(predict(bst, X_train))
            if (has_test) pi_test <- as.matrix(predict(bst, X_test))
        } else {
            cv <- xgboost::xgb.cv(data = dtrain, nrounds = 100, nfold = 5, metrics = list("rmse"), 
                                  max_depth = 3, eta = 1, objective = "reg:squarederror")
            num_trees = c(20, 50, 100)
            etas <- c(0.01, 0.05, 0.1, 0.5)
            max_depth <- c(3, 6)
            eval_grid <- expand.grid(num_trees, etas, max_depth)
            evals <- rep(NA, nrow(eval_grid))
            for (i in 1:nrow(eval_grid)) {
                cv <- xgboost::xgb.cv(data = dtrain, nrounds = 100, nfold = 5, metrics = list("rmse","auc"), 
                                      max_depth = 3, eta = 0.1, objective = "reg:squarederror", verbose = 0)
                evals[i] <- cv$evaluation_log[length(cv$evaluation_log),"test_rmse_mean"]$test_rmse_mean
            }
            best_params <- as.numeric(eval_grid[which.min(evals),])
            param <- list(max_depth = best_params[3], eta = best_params[2], verbose = 0, objective = "reg:squarederror", eval_metric = "rmse")
            bst <- xgboost::xgb.train(param, dtrain, nrounds = best_params[1])
            pi_train <- as.matrix(predict(bst, X_train))
            if (has_test) pi_test <- as.matrix(predict(bst, X_test))
        }
    }

    if (has_test) {
        if (is.null(pi_test)) stop("Propensity score must be provided for the test set as well")
    }
    
    if (propensity_covariate == "mu") {
        feature_types_mu <- as.integer(c(feature_types,0))
        feature_types_tau <- as.integer(feature_types)
        X_train_mu <- cbind(X_train, pi_train)
        X_train_tau <- X_train
        if (has_test) {
            X_test_mu <- cbind(X_test, pi_test)
            X_test_tau <- X_test
        }
    } else if (propensity_covariate == "tau") {
        feature_types_mu <- as.integer(feature_types)
        feature_types_tau <- as.integer(c(feature_types,0))
        X_train_mu <- X_train
        X_train_tau <- cbind(X_train, pi_train)
        if (has_test) {
            X_test_mu <- X_test
            X_test_tau <- cbind(X_test, pi_test)
        }
    } else if (propensity_covariate == "both") {
        feature_types_mu <- as.integer(c(feature_types,0))
        feature_types_tau <- as.integer(c(feature_types,0))
        X_train_mu <- cbind(X_train, pi_train)
        X_train_tau <- cbind(X_train, pi_train)
        if (has_test) {
            X_test_mu <- cbind(X_test, pi_test)
            X_test_tau <- cbind(X_test, pi_test)
        }
    } else if (propensity_covariate == "none") {
        feature_types_mu <- as.integer(feature_types)
        feature_types_tau <- as.integer(feature_types)
        X_train_mu <- X_train
        X_train_tau <- X_train
        if (has_test) {
            X_test_mu <- X_test
            X_test_tau <- X_test
        }
    } else {
        stop("propensity_covariate must equal one of 'none', 'mu', 'tau', or 'both'")
    }
    
    # Set variable weights for the prognostic and treatment effect forests
    variable_weights_mu = rep(1/ncol(X_train_mu), ncol(X_train_mu))
    variable_weights_tau = rep(1/ncol(X_train_tau), ncol(X_train_tau))
    
    # Standardize outcome separately for test and train
    y_bar_train <- mean(y_train)
    y_std_train <- sd(y_train)
    resid_train <- (y_train-y_bar_train)/y_std_train
    
    # Calibrate priors for global sigma^2 and sigma_leaf_mu / sigma_leaf_tau
    reg_basis <- X_train
    sigma2hat <- mean(resid(lm(y~reg_basis))^2)
    quantile_cutoff <- 0.9
    if (is.null(lambda)) {
        lambda <- (sigma2hat*qgamma(1-quantile_cutoff,nu))/nu
    }
    if (is.null(sigma2)) sigma2 <- sigma2hat
    if (is.null(b_leaf_mu)) b_leaf_mu <- var(resid_train)/(num_trees_mu)
    if (is.null(b_leaf_tau)) b_leaf_tau <- var(resid_train)/(2*num_trees_tau)
    if (is.null(sigma_leaf_mu)) sigma_leaf_mu <- var(resid_train)/(num_trees_mu)
    if (is.null(sigma_leaf_tau)) sigma_leaf_tau <- var(resid_train)/(2*num_trees_tau)
    current_sigma2 <- sigma2
    current_leaf_scale_mu <- as.matrix(sigma_leaf_mu)
    current_leaf_scale_tau <- as.matrix(sigma_leaf_tau)
    
    # Container of variance parameter samples
    num_samples <- num_gfr + num_burnin + num_mcmc
    if (sample_sigma_global) global_var_samples <- rep(0, num_samples)
    if (sample_sigma_leaf_mu) leaf_scale_mu_samples <- rep(0, num_samples)
    if (sample_sigma_leaf_tau) leaf_scale_tau_samples <- rep(0, num_samples)

    # Prepare adaptive coding structure
    if ((!is.numeric(b_0)) || (!is.numeric(b_1)) || (length(b_0) > 1) || (length(b_1) > 1)) {
        stop("b_0 and b_1 must be single numeric values")
    }
    if (adaptive_coding) {
        b_0_samples <- rep(0, num_samples)
        b_1_samples <- rep(0, num_samples)
        current_b_0 <- b_0
        current_b_1 <- b_1
        tau_basis_train <- (1-Z_train)*current_b_0 + Z_train*current_b_1
        if (has_test) tau_basis_test <- (1-Z_test)*current_b_0 + Z_test*current_b_1
    } else {
        tau_basis_train <- Z_train
        if (has_test) tau_basis_test <- Z_test
    }
    
    # Data
    forest_dataset_mu_train <- createForestDataset(X_train_mu)
    forest_dataset_tau_train <- createForestDataset(X_train_tau, tau_basis_train)
    if (has_test) forest_dataset_mu_test <- createForestDataset(X_test_mu)
    if (has_test) forest_dataset_tau_test <- createForestDataset(X_test_tau, tau_basis_test)
    outcome_train <- createOutcome(resid_train)
    
    # Random number generator (std::mt19937)
    if (is.null(random_seed)) random_seed = sample(1:10000,1,F)
    rng <- createRNG(random_seed)
    
    # Sampling data structures
    forest_model_mu <- createForestModel(forest_dataset_mu_train, feature_types_mu, num_trees_mu, nrow(X_train_mu), alpha_mu, beta_mu, min_samples_leaf_mu)
    forest_model_tau <- createForestModel(forest_dataset_tau_train, feature_types_tau, num_trees_tau, nrow(X_train_tau), alpha_tau, beta_tau, min_samples_leaf_tau)
    
    # Container of forest samples
    forest_samples_mu <- createForestContainer(num_trees_mu, 1, T)
    forest_samples_tau <- createForestContainer(num_trees_tau, 1, F)
    
    # Initialize the leaves of each tree in the prognostic forest
    forest_samples_mu$set_root_leaves(0, mean(resid_train) / num_trees_mu)
    update_residual_forest_container_cpp(forest_dataset_mu_train$data_ptr, outcome_train$data_ptr, 
                                         forest_samples_mu$forest_container_ptr, forest_model_mu$tracker_ptr, 
                                         F, 0, F)
    
    # Initialize the leaves of each tree in the treatment effect forest
    forest_samples_tau$set_root_leaves(0, 0.)
    update_residual_forest_container_cpp(forest_dataset_tau_train$data_ptr, outcome_train$data_ptr, 
                                         forest_samples_tau$forest_container_ptr, forest_model_tau$tracker_ptr, 
                                         T, 0, F)
    
    # Run GFR (warm start) if specified
    if (num_gfr > 0){
        for (i in 1:num_gfr) {
            # Sample the prognostic forest
            forest_model_mu$sample_one_iteration(
                forest_dataset_mu_train, outcome_train, forest_samples_mu, rng, feature_types_mu, 
                0, current_leaf_scale_mu, variable_weights_mu, 
                current_sigma2, cutpoint_grid_size, gfr = T, pre_initialized = T
            )
            
            # Sample variance parameters (if requested)
            if (sample_sigma_global) {
                global_var_samples[i] <- sample_sigma2_one_iteration(outcome_train, rng, nu, lambda)
                current_sigma2 <- global_var_samples[i]
            }
            if (sample_sigma_leaf_mu) {
                leaf_scale_mu_samples[i] <- sample_tau_one_iteration(forest_samples_mu, rng, a_leaf_mu, b_leaf_mu, i-1)
                current_leaf_scale_mu <- as.matrix(leaf_scale_mu_samples[i])
            }
            
            # Sample the treatment forest
            forest_model_tau$sample_one_iteration(
                forest_dataset_tau_train, outcome_train, forest_samples_tau, rng, feature_types_tau, 
                1, current_leaf_scale_tau, variable_weights_tau, 
                current_sigma2, cutpoint_grid_size, gfr = T, pre_initialized = T
            )
            
            # Sample coding parameters (if requested)
            if (adaptive_coding) {
                # Estimate mu(X) and tau(X) and compute y - mu(X)
                mu_x_raw_train <- forest_samples_mu$predict_raw_single_forest(forest_dataset_mu_train, i-1)
                tau_x_raw_train <- forest_samples_tau$predict_raw_single_forest(forest_dataset_tau_train, i-1)
                partial_resid_mu_train <- resid_train - mu_x_raw_train
                
                # Compute sufficient statistics for regression of y - mu(X) on [tau(X)(1-Z), tau(X)Z]
                s_tt0 <- sum(tau_x_raw_train*tau_x_raw_train*(Z_train==0))
                s_tt1 <- sum(tau_x_raw_train*tau_x_raw_train*(Z_train==1))
                s_ty0 <- sum(tau_x_raw_train*partial_resid_mu_train*(Z_train==0))
                s_ty1 <- sum(tau_x_raw_train*partial_resid_mu_train*(Z_train==1))
                
                # Sample b0 (coefficient on tau(X)(1-Z)) and b1 (coefficient on tau(X)Z)
                current_b_0 <- rnorm(1, (s_ty0/(s_tt0 + 2*current_sigma2)), sqrt(current_sigma2/(s_tt0 + 2*current_sigma2)))
                current_b_1 <- rnorm(1, (s_ty1/(s_tt1 + 2*current_sigma2)), sqrt(current_sigma2/(s_tt1 + 2*current_sigma2)))
                
                # Update basis for the leaf regression
                tau_basis_train <- (1-Z_train)*current_b_0 + Z_train*current_b_1
                forest_dataset_tau_train$update_basis(tau_basis_train)
                b_0_samples[i] <- current_b_0
                b_1_samples[i] <- current_b_1
                if (has_test) {
                    tau_basis_test <- (1-Z_test)*current_b_0 + Z_test*current_b_1
                    forest_dataset_tau_test$update_basis(tau_basis_test)
                }
                
                # TODO Update leaf predictions and residual
            }
            
            # Sample variance parameters (if requested)
            if (sample_sigma_global) {
                global_var_samples[i] <- sample_sigma2_one_iteration(outcome_train, rng, nu, lambda)
                current_sigma2 <- global_var_samples[i]
            }
            if (sample_sigma_leaf_tau) {
                leaf_scale_tau_samples[i] <- sample_tau_one_iteration(forest_samples_tau, rng, a_leaf_tau, b_leaf_tau, i-1)
                current_leaf_scale_tau <- as.matrix(leaf_scale_tau_samples[i])
            }
        }
    }
    
    # Run MCMC
    if (num_burnin + num_mcmc > 0) {
        for (i in (num_gfr+1):num_samples) {
            # Sample the prognostic forest
            forest_model_mu$sample_one_iteration(
                forest_dataset_mu_train, outcome_train, forest_samples_mu, rng, feature_types_mu, 
                0, current_leaf_scale_mu, variable_weights_mu, 
                current_sigma2, cutpoint_grid_size, gfr = F, pre_initialized = T
            )
            
            # Sample variance parameters (if requested)
            if (sample_sigma_global) {
                global_var_samples[i] <- sample_sigma2_one_iteration(outcome_train, rng, nu, lambda)
                current_sigma2 <- global_var_samples[i]
            }
            if (sample_sigma_leaf_mu) {
                leaf_scale_mu_samples[i] <- sample_tau_one_iteration(forest_samples_mu, rng, a_leaf_mu, b_leaf_mu, i-1)
                current_leaf_scale_mu <- as.matrix(leaf_scale_mu_samples[i])
            }
            
            # Sample the treatment forest
            forest_model_tau$sample_one_iteration(
                forest_dataset_tau_train, outcome_train, forest_samples_tau, rng, feature_types_tau, 
                1, current_leaf_scale_tau, variable_weights_tau, 
                current_sigma2, cutpoint_grid_size, gfr = F, pre_initialized = T
            )
            
            # Sample coding parameters (if requested)
            if (adaptive_coding) {
                mu_x_raw_train <- forest_samples_mu$predict_raw_single_forest(forest_dataset_mu_train, i-1)
                tau_x_raw_train <- forest_samples_tau$predict_raw_single_forest(forest_dataset_tau_train, i-1)
                s_tt0 <- sum(tau_x_raw_train*tau_x_raw_train*(Z_train==0))
                s_tt1 <- sum(tau_x_raw_train*tau_x_raw_train*(Z_train==1))
                partial_resid_mu_train <- resid_train - mu_x_raw_train
                s_ty0 <- sum(tau_x_raw_train*partial_resid_mu_train*(Z_train==0))
                s_ty1 <- sum(tau_x_raw_train*partial_resid_mu_train*(Z_train==1))
                current_b_0 <- rnorm(1, (s_ty0/(s_tt0 + 2*current_sigma2)), sqrt(current_sigma2/(s_tt0 + 2*current_sigma2)))
                current_b_1 <- rnorm(1, (s_ty1/(s_tt1 + 2*current_sigma2)), sqrt(current_sigma2/(s_tt1 + 2*current_sigma2)))
                tau_basis_train <- (1-Z_train)*current_b_0 + Z_train*current_b_1
                forest_dataset_tau_train$update_basis(tau_basis_train)
                b_0_samples[i] <- current_b_0
                b_1_samples[i] <- current_b_1
                if (has_test) {
                    tau_basis_test <- (1-Z_test)*current_b_0 + Z_test*current_b_1
                    forest_dataset_tau_test$update_basis(tau_basis_test)
                }
            }
            
            # Sample variance parameters (if requested)
            if (sample_sigma_global) {
                global_var_samples[i] <- sample_sigma2_one_iteration(outcome_train, rng, nu, lambda)
                current_sigma2 <- global_var_samples[i]
            }
            if (sample_sigma_leaf_tau) {
                leaf_scale_tau_samples[i] <- sample_tau_one_iteration(forest_samples_tau, rng, a_leaf_tau, b_leaf_tau, i-1)
                current_leaf_scale_tau <- as.matrix(leaf_scale_tau_samples[i])
            }
        }
    }
    
    # Forest predictions
    mu_hat_train <- forest_samples_mu$predict(forest_dataset_mu_train)*y_std_train + y_bar_train
    # tau_hat_train <- forest_samples_tau$predict_raw(forest_dataset_tau_train)*y_std_train
    if (adaptive_coding) {
        tau_hat_train_raw <- forest_samples_tau$predict_raw(forest_dataset_tau_train)
        tau_hat_train <- t(t(tau_hat_train_raw) * (b_1_samples - b_0_samples))*y_std_train
    } else {
        tau_hat_train <- forest_samples_tau$predict_raw(forest_dataset_tau_train)*y_std_train
    }
    y_hat_train <- mu_hat_train + tau_hat_train * as.numeric(Z_train)
    if (has_test) {
        mu_hat_test <- forest_samples_mu$predict(forest_dataset_mu_test)*y_std_train + y_bar_train
        # tau_hat_test <- forest_samples_tau$predict(forest_dataset_tau_test)*y_std_train
        if (adaptive_coding) {
            tau_hat_test_raw <- forest_samples_tau$predict_raw(forest_dataset_tau_test)
            tau_hat_test <- t(t(tau_hat_test_raw) * (b_1_samples - b_0_samples))*y_std_train
        } else {
            tau_hat_test <- forest_samples_tau$predict_raw(forest_dataset_tau_test)*y_std_train
        }
        y_hat_test <- mu_hat_test + tau_hat_test * as.numeric(Z_test)
    }
    
    # Global error variance
    if (sample_sigma_global) sigma2_samples <- global_var_samples*(y_std_train^2)
    
    # Leaf parameter variance for prognostic forest
    if (sample_sigma_leaf_mu) sigma_leaf_mu_samples <- leaf_scale_mu_samples
    
    # Leaf parameter variance for treatment effect forest
    if (sample_sigma_leaf_tau) sigma_leaf_tau_samples <- leaf_scale_tau_samples
    
    # Return results as a list
    model_params <- list(
        "initial_sigma2" = sigma2, 
        "initial_sigma_leaf_mu" = sigma_leaf_mu,
        "initial_sigma_leaf_tau" = sigma_leaf_tau,
        "initial_b_0" = b_0,
        "initial_b_1" = b_1,
        "nu" = nu,
        "lambda" = lambda, 
        "a_leaf_mu" = a_leaf_mu, 
        "b_leaf_mu" = b_leaf_mu,
        "a_leaf_tau" = a_leaf_tau, 
        "b_leaf_tau" = b_leaf_tau,
        "outcome_mean" = y_bar_train,
        "outcome_scale" = y_std_train, 
        "num_covariates" = ncol(X_train),
        "num_prognostic_covariates" = ncol(X_train_mu),
        "num_treatment_covariates" = ncol(X_train_tau),
        "treatment_dim" = ncol(Z_train), 
        "propensity_covariate" = propensity_covariate, 
        "binary_treatment" = binary_treatment, 
        "adaptive_coding" = adaptive_coding, 
        "num_samples" = num_samples
    )
    result <- list(
        "forests_mu" = forest_samples_mu, 
        "forests_tau" = forest_samples_tau, 
        "model_params" = model_params, 
        "mu_hat_train" = mu_hat_train, 
        "tau_hat_train" = tau_hat_train, 
        "y_hat_train" = y_hat_train
    )
    if (has_test) result[["mu_hat_test"]] = mu_hat_test
    if (has_test) result[["tau_hat_test"]] = tau_hat_test
    if (has_test) result[["y_hat_test"]] = y_hat_test
    if (sample_sigma_global) result[["sigma2_samples"]] = sigma2_samples
    if (sample_sigma_leaf_mu) result[["sigma_leaf_mu_samples"]] = sigma_leaf_mu_samples
    if (sample_sigma_leaf_tau) result[["sigma_leaf_tau_samples"]] = sigma_leaf_tau_samples
    if (adaptive_coding) {
        result[["b_0_samples"]] = b_0_samples
        result[["b_1_samples"]] = b_1_samples
    }
    class(result) <- "bcf"
    
    return(result)
}

#' Predict from a sampled BCF model on new data
#'
#' @param bcf Object of type `bcf` containing draws of a Bayesian causal forest model and associated sampling outputs.
#' @param X_test Covariates used to determine tree leaf predictions for each observation.
#' @param Z_test Treatments used for prediction.
#' @param pi_test (Optional) Propensities used for prediction. Default: `NULL`.
#'
#' @return List of three `nrow(X_test)` by `bcf$num_samples` matrices: prognostic function estimates, treatment effect estimates and outcome predictions.
#' @export
#'
#' @examples
#' n <- 500
#' x1 <- rnorm(n)
#' x2 <- rnorm(n)
#' x3 <- rnorm(n)
#' x4 <- as.numeric(rbinom(n,1,0.5))
#' x5 <- as.numeric(sample(1:3,n,replace=T))
#' X <- cbind(x1,x2,x3,x4,x5)
#' p <- ncol(X)
#' g <- function(x) {ifelse(x[,5]==1,2,ifelse(x[,5]==2,-1,4))}
#' mu1 <- function(x) {1+g(x)+x[,1]*x[,3]}
#' mu2 <- function(x) {1+g(x)+6*abs(x[,3]-1)}
#' tau1 <- function(x) {rep(3,nrow(x))}
#' tau2 <- function(x) {1+2*x[,2]*x[,4]}
#' mu_x <- mu1(X)
#' tau_x <- tau2(X)
#' pi_x <- 0.8*pnorm((3*mu_x/sd(mu_x)) - 0.5*X[,1]) + 0.05 + runif(n)/10
#' Z <- rbinom(n,1,pi_x)
#' E_XZ <- mu_x + Z*tau_x
#' snr <- 4
#' y <- E_XZ + rnorm(n, 0, 1)*(sd(E_XZ)/snr)
#' test_set_pct <- 0.2
#' n_test <- round(test_set_pct*n)
#' n_train <- n - n_test
#' test_inds <- sort(sample(1:n, n_test, replace = F))
#' train_inds <- (1:n)[!((1:n) %in% test_inds)]
#' X_test <- X[test_inds,]
#' X_train <- X[train_inds,]
#' pi_test <- pi_x[test_inds]
#' pi_train <- pi_x[train_inds]
#' Z_test <- Z[test_inds]
#' Z_train <- Z[train_inds]
#' y_test <- y[test_inds]
#' y_train <- y[train_inds]
#' mu_test <- mu_x[test_inds]
#' mu_train <- mu_x[train_inds]
#' tau_test <- tau_x[test_inds]
#' tau_train <- tau_x[train_inds]
#' bcf_model <- bcf(X_train = X_train, Z_train = Z_train, y_train = y_train, pi_train = pi_train)
#' preds <- predict(bcf_model, X_test, Z_test, pi_test)
#' # plot(rowMeans(preds$mu_hat), mu_test, xlab = "predicted", ylab = "actual", main = "Prognostic function")
#' # abline(0,1,col="red",lty=3,lwd=3)
#' # plot(rowMeans(preds$tau_hat), tau_test, xlab = "predicted", ylab = "actual", main = "Treatment effect")
#' # abline(0,1,col="red",lty=3,lwd=3)
predict.bcf <- function(bcf, X_test, Z_test, pi_test = NULL){
    # Convert all input data to matrices if not already converted
    if ((is.null(dim(X_test))) && (!is.null(X_test))) {
        X_test <- as.matrix(X_test)
    }
    if ((is.null(dim(Z_test))) && (!is.null(Z_test))) {
        Z_test <- as.matrix(as.numeric(Z_test))
    }
    if ((is.null(dim(pi_test))) && (!is.null(pi_test))) {
        pi_test <- as.matrix(pi_test)
    }
    
    # Data checks
    if ((bcf$model_params$propensity_covariate != "none") && (is.null(pi_test))) {
        stop("pi_test must be provided for this model")
    }
    if (nrow(X_test) != nrow(Z_test)) {
        stop("X_test and Z_test must have the same number of rows")
    }
    if (bcf$model_params$num_covariates != ncol(X_test)) {
        stop("X_test and must have the same number of columns as the covariates used to train the model")
    }
    
    # Add propensities to any covariate set
    if (bcf$model_params$propensity_covariate == "both") {
        X_test_mu <- cbind(X_test, pi_test)
        X_test_tau <- cbind(X_test, pi_test)
    } else if (bcf$model_params$propensity_covariate == "mu") {
        X_test_mu <- cbind(X_test, pi_test)
        X_test_tau <- X_test
    } else if (bcf$model_params$propensity_covariate == "tau") {
        X_test_mu <- X_test
        X_test_tau <- cbind(X_test, pi_test)
    }
    
    # Create prediction datasets
    prediction_dataset_mu <- createForestDataset(X_test_mu)
    prediction_dataset_tau <- createForestDataset(X_test_tau, Z_test)

    # Compute and return predictions
    y_std <- bcf$model_params$outcome_scale
    y_bar <- bcf$model_params$outcome_mean
    mu_hat_test <- bcf$forests_mu$predict(prediction_dataset_mu)*y_std + y_bar
    if (bcf$model_params$adaptive_coding) {
        tau_hat_test_raw <- bcf$forests_tau$predict_raw(prediction_dataset_tau)
        tau_hat_test <- t(t(tau_hat_test_raw) * (bcf$b_1_samples - bcf$b_0_samples))*y_std
    } else {
        tau_hat_test <- bcf$forests_tau$predict_raw(forest_dataset_tau_test)*y_std
    }
    y_hat_test <- mu_hat_test + tau_hat_test * as.numeric(Z_test)
    
    result <- list(
        "mu_hat" = mu_hat_test, 
        "tau_hat" = tau_hat_test, 
        "y_hat" = y_hat_test
    )
    return(result)
}
