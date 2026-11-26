#load packages
library(tidyverse)
library(fpp3)
library(glmnet)
library(ranger)
library(xgboost)
library(fredr)
library(pROC)

# Part 2 - Penalized Regression: Forecasting Unemployment with 121 Predictors

### 2.1. The Data

fm <- read_csv("data/fredmd_transformed.csv", show_col_types = FALSE)
dim(fm)

fm_u <- fm |>
  mutate(y_next = lead(UNRATE)) |>     # next month's change in the unemployment rate
  filter(!is.na(y_next))

X <- fm_u |> select(-date, -y_next) |> as.matrix()   # predictors: all series this month
y <- fm_u$y_next                                     # target: next month's change

train <- fm_u$date < as.Date("1979-12-01")                                     # 1960-1979
val   <- fm_u$date >= as.Date("1979-12-01") & fm_u$date < as.Date("1989-12-01") # 1980-1989

X_train <- X[train, ]
y_train <- y[train]
X_val   <- X[val, ]
y_val   <- y[val]

### 2.1b. Mean of the Target Variable over the Training Sample and Validation Sample

mean(y_train)
mean(y_val)

###Yes, the means are close to zero. 
###They tell us that level of the unemployment rate over each sample is relatively stable

### 2.2. Choose λ on a Validation Sample

grid <- 10^seq(2, -3, length.out = 60)

fit_lasso <- glmnet(X_train, y_train, alpha = 1, lambda = grid)
fit_ridge <- glmnet(X_train, y_train, alpha = 0, lambda = grid)

validation <- bind_rows(
  tibble(method = "Lasso", lambda = grid,
         rmse = sqrt(colMeans((y_val - predict(fit_lasso, X_val))^2))),
  tibble(method = "Ridge", lambda = grid,
         rmse = sqrt(colMeans((y_val - predict(fit_ridge, X_val))^2))))

ggplot(validation, aes(lambda, rmse, color = method)) +
  geom_line() +
  scale_x_log10() +
  labs(title = "Validation RMSE, 1980-1989",
       x = "lambda (log scale)", y = "RMSE", color = NULL) +
  theme_minimal()

lam_lasso <- grid[which.min(validation$rmse[validation$method == "Lasso"])]
lam_ridge <- grid[which.min(validation$rmse[validation$method == "Ridge"])]
c(lasso = lam_lasso, ridge = lam_ridge)

###For the Lasso penalty regression, the λ chosen is 0.01263848.
###For the Ridge penalty regression, the λ chosen is 0.28683168.
###When λ is very large, that means all slope parameters are shrunk to zero. Therefore, the model would fail to include more useful information and thus becomes underfitting, resulting in a very high validation RMSE
###When λ is very small, it means that there's nearly no penalty and we are almost back to OLS. Thus, the model includes too much noise with 121 predictors and is thus overfitting, again resulting in a high validation RMSE.

### 2.3. Which Variables Does the Lasso Select?

first <- fm_u$date < as.Date("1989-12-01")
b <- coef(glmnet(X[first, ], y[first], alpha = 1, lambda = grid), s = lam_lasso)

kept <- tibble(series = rownames(b), beta = b[, 1]) |>
  filter(series != "(Intercept)", beta != 0) |>
  arrange(desc(abs(beta)))

kept
nrow(kept)
 
###Lasso keeps 19 series.
###The first five series with the largest parameters in absolute value are: USTPU, NDMANEMP, IPMAT, HOUSTW, CUSR0000SAS. 
###USTPU is the employment in trade, transportation and utilities. The series has been logged and then taken the first difference, which taking the growth rate.
###NDMANEMP is the employment in nondurable goods. The series has also been logged and then taken the first difference, which taking the growth rate.
###IPMAT is the production in materials. The series has also been logged and then taken the first difference, which taking the growth rate.
###HOUSTW is the Housing starts, West. The series has been only logged.
###CUSR0000SAS is the consumer price index for services. The series has been logged and then taken the second difference, which taking the change in the growth rate.
###The signs generally make sense. The target variable Civilian Unemployment Rate should have a negative association with the growth rate of the employment series and production series, and a positive association with the change in growth rate of the CPI for services.
###UNRATE itself is not among the kept series.

### 2.4. The Horse Race

origins <- which(fm_u$date >= as.Date("1989-12-01"))
own <- X[, "UNRATE"]           # this month's change in unemployment, for the AR(1)

e_ar <- e_ridge <- e_lasso <- n_kept <- numeric(length(origins))
for (j in seq_along(origins)) {
  T <- origins[j]
  tr <- 1:(T - 1)               # rows in this training sample
  e_ar[j] <- y[T] - sum(coef(lm(y[tr] ~ own[tr])) * c(1, own[T]))
  f_r <- glmnet(X[tr, ], y[tr], alpha = 0, lambda = grid)
  f_l <- glmnet(X[tr, ], y[tr], alpha = 1, lambda = grid)
  e_ridge[j] <- y[T] - predict(f_r, X[T, , drop = FALSE], s = lam_ridge)
  e_lasso[j] <- y[T] - predict(f_l, X[T, , drop = FALSE], s = lam_lasso)
  n_kept[j] <- sum(coef(f_l, s = lam_lasso)[-1] != 0)
}

unrate_results <- tibble(
  Model = c("AR(1)", "Ridge", "Lasso"),
  RMSE = sqrt(c(mean(e_ar^2), mean(e_ridge^2), mean(e_lasso^2))),
  MAE = c(mean(abs(e_ar)), mean(abs(e_ridge)), mean(abs(e_lasso))))
unrate_results

### 2.4a. RMSE and MAE of the three models

###There are 360 forecasts in total in this horse race. 
unrate_results
###The RMSE of the AR(1) model is 0.152, of the Ridge model is 0.144, and of the Lasso model is 0.136. 
###The MAE of the AR(1) model is 0.116, of the Ridge model is 0.111, and of the Lasso model is 0.107. 
###The Lasso model forecasts the best with lower RMSE and MAE than the other two alternatives. It reduces the RMSE relative to AR(1) by 10.5%. 


### 2.4b. DM Test with squared error loss

forecast::dm.test(e_ar, e_lasso, h = 1, power = 2)

###The result of the DM test shows that the Lasso model forecasts significantly better than the AR(1) model at the 5% level. 
###The p-value is around 0.0002, which is much less than 0.05. Therefore, we reject the null hypothesis that the two models have equal accuracy and conclude that the Lasso model has superior forecasting performance compared to the AR(1) model.

### 2.4c. Plot n-kept over time

tibble(date = fm_u$date[origins], n_kept = n_kept) |>
  ggplot(aes(date, n_kept)) +
  geom_point() +
  labs(title = "Number of Predictors Kept by the Lasso",
       x = "Forecast origin", y = "Number of predictors kept") +
  theme_minimal()

###Lasso does not deep about the same numbers of predictors over time. The number of predictors kept by the Lasso model fluctuates between 10 and 20, with an average of around 15. 
###This indicates that the Lasso model is not stable, in other words, really adaptively selecting relevant predictors for forecasting unemployment over time.

### 2.5. Sparse vs. Dense: Housing Starts

penalized_race <- function(target) {
  d <- fm |> mutate(y_next = lead(.data[[target]])) |> filter(!is.na(y_next))
  X <- d |> select(-date, -y_next) |> as.matrix()
  y <- d$y_next
  own <- X[, target]
  train <- d$date < as.Date("1979-12-01")
  val   <- d$date >= as.Date("1979-12-01") & d$date < as.Date("1989-12-01")
  lam <- c(ridge = 0, lasso = 1) |>            # choose lambda on the validation sample
    map_dbl(function(a) {
      f <- glmnet(X[train, ], y[train], alpha = a, lambda = grid)
      grid[which.min(colMeans((y[val] - predict(f, X[val, ]))^2))]
    })
  origins <- which(d$date >= as.Date("1989-12-01"))
  e_ar <- e_ridge <- e_lasso <- numeric(length(origins))
  for (j in seq_along(origins)) {
    T <- origins[j]
    tr <- 1:(T - 1)
    e_ar[j] <- y[T] - sum(coef(lm(y[tr] ~ own[tr])) * c(1, own[T]))
    f_r <- glmnet(X[tr, ], y[tr], alpha = 0, lambda = grid)
    f_l <- glmnet(X[tr, ], y[tr], alpha = 1, lambda = grid)
    e_ridge[j] <- y[T] - predict(f_r, X[T, , drop = FALSE], s = lam["ridge"])
    e_lasso[j] <- y[T] - predict(f_l, X[T, , drop = FALSE], s = lam["lasso"])
  }
  tibble(target,
         RMSE_AR = sqrt(mean(e_ar^2)),
         RMSE_ridge = sqrt(mean(e_ridge^2)),
         RMSE_lasso = sqrt(mean(e_lasso^2)))
}

penalized_race("UNRATE")    # should match your results in 2.4
penalized_race("HOUST")

###The lasso has got a smaller RMSE than the ridge for housing starts.
###Since forecasting the housing starts in logs should be a sparse problem given that it can be a very persistent series and thus the value of this month can be the best single predictor of next month's. 
###With the ridge penalty, the model is forced to include all 121 predictors and only shrink them in the direction of zero without eliminating them. This may introduce to much estimation error and noise into the model and the consequent overfitting problem will largely reduce forecasting accuracy.




















