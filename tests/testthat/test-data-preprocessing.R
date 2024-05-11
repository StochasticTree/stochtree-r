test_that("Preprocessing of all-numeric covariate dataset works", {
    cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
    cov_mat <- matrix(c(
        1,2,3,4,5,
        5,4,3,2,1,
        6,7,8,9,10
    ), ncol = 3, byrow = F)
    preprocess_list <- createForestCovariates(cov_df)
    expect_equal(preprocess_list$X, cov_mat)
    expect_equal(preprocess_list$feature_types, rep(0,3))
    expect_equal(preprocess_list$num_numeric_vars, 3)
    expect_equal(preprocess_list$num_ordered_cat_vars, 0)
    expect_equal(preprocess_list$num_unordered_cat_vars, 0)
    expect_equal(preprocess_list$numeric_vars, c("x1","x2","x3"))
})

test_that("Preprocessing of all-unordered-categorical covariate dataset works", {
    cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
    cov_mat <- matrix(c(
        1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,
        0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,
        0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,
        0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,
        0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0
    ), nrow = 5, byrow = T)
    preprocess_list <- createForestCovariates(cov_df, unordered_cat_vars = c("x1","x2","x3"))
    expect_equal(preprocess_list$X, cov_mat)
    expect_equal(preprocess_list$feature_types, rep(1,18))
    expect_equal(preprocess_list$num_numeric_vars, 0)
    expect_equal(preprocess_list$num_ordered_cat_vars, 0)
    expect_equal(preprocess_list$num_unordered_cat_vars, 3)
    expect_equal(preprocess_list$unordered_cat_vars, c("x1","x2","x3"))
    expect_equal(preprocess_list$unordered_unique_levels, 
                 list(x1=c("1","2","3","4","5"), 
                      x2=c("1","2","3","4","5"), 
                      x3=c("6","7","8","9","10"))
    )
})

test_that("Preprocessing of all-ordered-categorical covariate dataset works", {
    cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
    cov_mat <- matrix(c(
        1,2,3,4,5,
        5,4,3,2,1,
        1,2,3,4,5
    ), ncol = 3, byrow = F)
    preprocess_list <- createForestCovariates(cov_df, ordered_cat_vars = c("x1","x2","x3"))
    expect_equal(preprocess_list$X, cov_mat)
    expect_equal(preprocess_list$feature_types, rep(1,3))
    expect_equal(preprocess_list$num_numeric_vars, 0)
    expect_equal(preprocess_list$num_ordered_cat_vars, 3)
    expect_equal(preprocess_list$num_unordered_cat_vars, 0)
    expect_equal(preprocess_list$ordered_cat_vars, c("x1","x2","x3"))
    expect_equal(preprocess_list$ordered_unique_levels, 
                 list(x1=c("1","2","3","4","5"), 
                      x2=c("1","2","3","4","5"), 
                      x3=c("6","7","8","9","10"))
    )
})

test_that("Preprocessing of mixed-covariate dataset works", {
    cov_df <- data.frame(x1 = 1:5, x2 = 5:1, x3 = 6:10)
    cov_mat <- matrix(c(
        1,5,1,0,0,0,0,0,
        2,4,0,1,0,0,0,0,
        3,3,0,0,1,0,0,0,
        4,2,0,0,0,1,0,0,
        5,1,0,0,0,0,1,0
    ), nrow = 5, byrow = T)
    preprocess_list <- createForestCovariates(cov_df, ordered_cat_vars = c("x2"), unordered_cat_vars = "x3")
    expect_equal(preprocess_list$X, cov_mat)
    expect_equal(preprocess_list$feature_types, c(0, rep(1,7)))
    expect_equal(preprocess_list$num_numeric_vars, 1)
    expect_equal(preprocess_list$num_ordered_cat_vars, 1)
    expect_equal(preprocess_list$num_unordered_cat_vars, 1)
    expect_equal(preprocess_list$ordered_cat_vars, c("x2"))
    expect_equal(preprocess_list$unordered_cat_vars, c("x3"))
    expect_equal(preprocess_list$ordered_unique_levels, list(x2=c("1","2","3","4","5")))
    expect_equal(preprocess_list$unordered_unique_levels, list(x3=c("6","7","8","9","10")))
})
