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


# Part 3 - Trees, Forests, and Boosting: Peak Electricity Demand

### 3.1a.
elec <- vic_elec |>
  index_by(Date = as_date(Time)) |>
  summarise(Peak = max(Demand) / 1e3,          # peak half-hourly demand (GW)
            Temp = max(Temperature),           # daily max temperature (C)
            Holiday = any(Holiday)) |>
  mutate(Day = wday(Date, label = TRUE),       # day of the week
         Lag1 = lag(Peak, 1),                  # yesterday's peak
         Lag7 = lag(Peak, 7),                  # peak one week ago
         TempLag1 = lag(Temp, 1)) |>           # yesterday's max temperature
  filter(!is.na(Lag7)) |>
  as_tibble()

### 3.1b. Scatterplot of Peak against Temp

elec |>
  ggplot(aes(Temp, Peak)) +
  geom_point(alpha=0.5) +
  labs(title = "Peak Electricity Demand vs. Temperature",
       x = "Daily Maximum Temperature (C)", y = "Peak Demand (GWh)") +
  theme_minimal()

###There seems to be a quadratic relationship between peak electricity demand and temperature with a U-shaped pattern. 
###It is shaped this way because when the temperature is very low, people will use more electricity for heating, and when the temperature is very high, people will use more electricity for cooling. Therefore, the peak electricity demand is higher at both ends of the temperature spectrum and lower in the middle.

### 3.2. Tune Boosting on a Validation Sample

features <- c("Day", "Holiday", "Temp", "Lag1", "Lag7")

fit_xgb <- function(d, depth, eta, rounds) {
  xgboost(x = d[features], y = d$Peak, nrounds = rounds, max_depth = depth,
          learning_rate = eta, nthreads = 1, seed = 422)
}

tr <- filter(elec, year(Date) == 2012)
va <- filter(elec, year(Date) == 2013)

xgb_grid <- expand_grid(depth = c(2, 4, 6), eta = c(0.05, 0.3), rounds = c(100, 500))
xgb_grid$val_rmse <- NA
for (i in 1:nrow(xgb_grid)) {
  m <- fit_xgb(tr, xgb_grid$depth[i], xgb_grid$eta[i], xgb_grid$rounds[i])
  xgb_grid$val_rmse[i] <- sqrt(mean((va$Peak - predict(m, va[features]))^2))
}
xgb_grid <- arrange(xgb_grid, val_rmse)
xgb_grid
best <- xgb_grid[1, ]

###There are 12 combinations of tuning parameters that the code given above tried. 
###The chosen combination is the one with the lowest validation RMSE, which is depth = 2, eta (learning rate) = 0.3, and rounds (number of trees) = 100. 
###In this example, it seems like deeper trees generally do worse than the shallow ones.

### 3.3. The Horse Race

test_dates <- elec$Date[year(elec$Date) == 2014]
refit_dates <- test_dates[seq(1, length(test_dates), by = 28)]

elec_fc <- NULL
for (k in seq_along(refit_dates)) {
  start <- refit_dates[k]
  end <- if (k < length(refit_dates)) refit_dates[k + 1] - 1 else max(elec$Date)
  train <- filter(elec, Date < start)
  test <- filter(elec, Date >= start, Date <= end)
  m_lin  <- lm(Peak ~ Day + Holiday + Temp + Lag1 + Lag7, data = train)
  m_quad <- lm(Peak ~ Day + Holiday + Temp + I(Temp^2) + Lag1 + Lag7, data = train)
  m_rf   <- ranger(Peak ~ Day + Holiday + Temp + Lag1 + Lag7, data = train, seed = 422)
  m_xgb  <- fit_xgb(train, best$depth, best$eta, best$rounds)
  elec_fc <- bind_rows(elec_fc, tibble(
    Date = test$Date, Actual = test$Peak,
    srw = test$Lag7,
    linear = predict(m_lin, test),
    quadratic = predict(m_quad, test),
    forest = predict(m_rf, test)$predictions,
    boosting = predict(m_xgb, test[features])))
}

elec_accuracy <- function(fc) {
  fc |>
    pivot_longer(-c(Date, Actual), names_to = "Model", values_to = "Forecast") |>
    group_by(Model) |>
    summarise(RMSE = sqrt(mean((Actual - Forecast)^2)),
              MAE = mean(abs(Actual - Forecast))) |>
    arrange(RMSE)
}
elec_accuracy(elec_fc)

### 3.3a. Forecast evaluation results

elec_accuracy(elec_fc)

###For RMSE, the ranking from lowest (best) to highest (worst) is: boosting (0.318), quadratic (0.337), forest (0.339), linear (0.489), and seaasonal random walk (0.862).
###For MAE, the ranking is the same with an exception that the quadratic model has a same MAE as the forest model (0.226).
###For total daily demand in class, we found that the quadratic model was the best, but boosting has a similar level performance with a only a slightly higher RMSE. Here, boosting is the best model for forecasting peak electricity demand which is more critical for the opeartion of the electricity system.

### 3.3b. 

max(elec$Peak[year(elec$Date) <= 2013])
elec_fc |> filter(Actual > max(elec$Peak[year(elec$Date) <= 2013]))

###In the comparison between the actual and forecasts and boosting with the regression with temperature squared, we can see that the two tress forecast models did not manage to predict/ capture the extreme peak in 2014.
###This is because a tree's forecast is an average of training values of y, so it can never go above the largest or below the smallest y in the training sapmle.
###Since the extreme peak in 2014 is much higher than any peak in the training sample, and thus the models are not able to extrapolate beyond the range of the training data.
###This could cause big issues for a grid operator because they want to ensure that there is enough capacity to meet demand and avoid blackouts. However, excepting all future demands will be in the range of previous ones could be unrealistic. Thus, with only forecasts produced by the tree forecasts, they may not be able to plan for the necessary capacity and could face risks like lackouts or other issues during periods of high demand. 

### 3.4. Feature Engineering: Yesterday's Temperature

features2 <- c(features, "TempLag1")       # add yesterday's temperature

elec_fc2 <- NULL
for (k in seq_along(refit_dates)) {
  start <- refit_dates[k]
  end <- if (k < length(refit_dates)) refit_dates[k + 1] - 1 else max(elec$Date)
  train <- filter(elec, Date < start)
  test <- filter(elec, Date >= start, Date <= end)
  m_quad2 <- lm(Peak ~ Day + Holiday + Temp + I(Temp^2) + TempLag1 + I(TempLag1^2) +
                  Lag1 + Lag7, data = train)
  m_xgb2 <- xgboost(x = train[features2], y = train$Peak, nrounds = best$rounds,
                    max_depth = best$depth, learning_rate = best$eta,
                    nthreads = 1, seed = 422)
  elec_fc2 <- bind_rows(elec_fc2, tibble(
    Date = test$Date, Actual = test$Peak,
    quadratic_templag = predict(m_quad2, test),
    boosting_templag = predict(m_xgb2, test[features2])))
}
elec_accuracy(elec_fc2)

###The RMSE of the quadratic model reduced from 0.337 to 0.272 (19%). The RMSE of the boosting model reduced from 0.318 to 0.300 (5.7%).
###The OLS regression with temperature squared and yesterday's temperature squared is now the best overall.
###This indicates that building better features can be more important than choosing a fancier model. Again, we need to keep the balance of using as much useful information as possible without including too much noise and errors in to our models.



