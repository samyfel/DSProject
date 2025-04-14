# Cryptocurrency Forecasting in R
# Custom implementation of time series models without using specialized packages

# Install and load required libraries
if (!require(quantmod)) install.packages("quantmod")
if (!require(zoo)) install.packages("zoo")
if (!require(ggplot2)) install.packages("ggplot2")
library(quantmod)  # For retrieving financial data
library(zoo)       # For time series manipulation
library(ggplot2)   # For visualization

# Set seed for reproducibility
set.seed(123)

# Function to implement AR model from scratch
ar_model <- function(data, lags) {
  n <- length(data)
  X <- matrix(0, nrow = n - lags, ncol = lags + 1)
  X[, 1] <- 1  # Intercept
  
  # Create lagged values for predictors
  for (i in 1:lags) {
    X[, i + 1] <- data[(lags - i + 1):(n - i)]
  }
  
  # Response variable
  y <- data[(lags + 1):n]
  
  # Fit linear regression model
  model <- lm(y ~ X - 1)
  
  # Return coefficients and model
  return(list(
    coefficients = coef(model),
    model = model
  ))
}

# Function to predict using AR model
ar_predict <- function(model, data, lags, h) {
  n <- length(data)
  coef <- model$coefficients
  predictions <- numeric(h)
  
  # Create initial lagged values
  lag_values <- data[(n - lags + 1):n]
  
  # Generate h-step ahead forecasts
  for (i in 1:h) {
    # Calculate prediction
    pred <- coef[1]  # Intercept
    for (j in 1:lags) {
      pred <- pred + coef[j + 1] * lag_values[lags - j + 1]
    }
    
    # Update lag values
    lag_values <- c(lag_values[-1], pred)
    predictions[i] <- pred
  }
  
  return(predictions)
}

# Function to implement MA model from scratch
ma_model <- function(data, order) {
  n <- length(data)
  
  # Initialize with random coefficients
  coefficients <- rnorm(order + 1, 0, 0.1)
  errors <- numeric(n)
  
  # Optimization function
  ma_obj <- function(params) {
    theta0 <- params[1]
    thetas <- params[-1]
    
    # Calculate errors
    e <- numeric(n)
    yhat <- numeric(n)
    
    for (t in (order+1):n) {
      yhat[t] <- theta0
      for (j in 1:order) {
        if (t-j > 0) {
          yhat[t] <- yhat[t] + thetas[j] * e[t-j]
        }
      }
      e[t] <- data[t] - yhat[t]
    }
    
    # Return sum of squared errors
    return(sum(e^2, na.rm = TRUE))
  }
  
  # Optimize parameters
  opt <- optim(coefficients, ma_obj, method = "BFGS")
  
  # Calculate final errors
  theta0 <- opt$par[1]
  thetas <- opt$par[-1]
  
  e <- numeric(n)
  yhat <- numeric(n)
  
  for (t in (order+1):n) {
    yhat[t] <- theta0
    for (j in 1:order) {
      if (t-j > 0) {
        yhat[t] <- yhat[t] + thetas[j] * e[t-j]
      }
    }
    e[t] <- data[t] - yhat[t]
  }
  
  return(list(
    coefficients = opt$par,
    errors = e,
    fitted = yhat
  ))
}

# Function to predict using MA model
ma_predict <- function(model, h) {
  coef <- model$coefficients
  errors <- model$errors
  n <- length(errors)
  
  predictions <- numeric(h)
  order <- length(coef) - 1
  
  # Generate h-step ahead forecasts
  for (i in 1:h) {
    pred <- coef[1]  # Constant term
    
    for (j in 1:min(i-1, order)) {
      if (i-j <= h) {
        # For future errors, use zero
        pred <- pred + coef[j+1] * 0
      } else {
        # For historical errors, use the calculated ones
        pred <- pred + coef[j+1] * errors[n-(i-j)+1]
      }
    }
    
    predictions[i] <- pred
  }
  
  return(predictions)
}

# Function to implement ARMA model from scratch
arma_model <- function(data, ar_order, ma_order) {
  n <- length(data)
  
  # Initialize with random coefficients
  coefficients <- rnorm(ar_order + ma_order + 1, 0, 0.1)
  errors <- numeric(n)
  
  # Optimization function
  arma_obj <- function(params) {
    mu <- params[1]
    ar_coef <- params[2:(ar_order+1)]
    ma_coef <- params[(ar_order+2):(ar_order+ma_order+1)]
    
    # Calculate errors
    e <- numeric(n)
    yhat <- numeric(n)
    
    for (t in (max(ar_order, ma_order)+1):n) {
      yhat[t] <- mu
      
      # AR component
      for (j in 1:ar_order) {
        if (t-j > 0) {
          yhat[t] <- yhat[t] + ar_coef[j] * data[t-j]
        }
      }
      
      # MA component
      for (j in 1:ma_order) {
        if (t-j > 0) {
          yhat[t] <- yhat[t] + ma_coef[j] * e[t-j]
        }
      }
      
      e[t] <- data[t] - yhat[t]
    }
    
    # Return sum of squared errors
    return(sum(e^2, na.rm = TRUE))
  }
  
  # Optimize parameters
  opt <- optim(coefficients, arma_obj, method = "BFGS")
  
  # Calculate final errors and fitted values
  mu <- opt$par[1]
  ar_coef <- opt$par[2:(ar_order+1)]
  ma_coef <- opt$par[(ar_order+2):(ar_order+ma_order+1)]
  
  e <- numeric(n)
  yhat <- numeric(n)
  
  for (t in (max(ar_order, ma_order)+1):n) {
    yhat[t] <- mu
    
    # AR component
    for (j in 1:ar_order) {
      if (t-j > 0) {
        yhat[t] <- yhat[t] + ar_coef[j] * data[t-j]
      }
    }
    
    # MA component
    for (j in 1:ma_order) {
      if (t-j > 0) {
        yhat[t] <- yhat[t] + ma_coef[j] * e[t-j]
      }
    }
    
    e[t] <- data[t] - yhat[t]
  }
  
  return(list(
    coefficients = opt$par,
    ar_coef = ar_coef,
    ma_coef = ma_coef,
    constant = mu,
    errors = e,
    fitted = yhat
  ))
}

# Function to predict using ARMA model
arma_predict <- function(model, data, ar_order, ma_order, h) {
  n <- length(data)
  mu <- model$constant
  ar_coef <- model$ar_coef
  ma_coef <- model$ma_coef
  errors <- model$errors[!is.na(model$errors)]  # Remove NAs from errors
  
  predictions <- numeric(h)
  hist_data <- tail(data, ar_order)  # Last ar_order observations
  hist_errors <- tail(errors, ma_order)  # Last ma_order errors
  
  # Generate h-step ahead forecasts
  for (i in 1:h) {
    pred <- mu
    
    # AR component
    for (j in 1:ar_order) {
      if (i-j <= 0) {
        # Use historical data
        pred <- pred + ar_coef[j] * hist_data[ar_order - j + 1]
      } else {
        # Use forecasted values
        pred <- pred + ar_coef[j] * predictions[i-j]
      }
    }
    
    # MA component - note that future errors are assumed to be 0
    for (j in 1:ma_order) {
      if (i-j <= 0 && j <= length(hist_errors)) {
        # Use historical errors
        pred <- pred + ma_coef[j] * hist_errors[ma_order - j + 1]
      } else {
        # Future errors are 0
        pred <- pred + ma_coef[j] * 0
      }
    }
    
    predictions[i] <- pred
  }
  
  return(predictions)
}

# Function to compute differencing for ARIMA
diff_series <- function(data, d) {
  if (d == 0) return(data)
  
  result <- data
  for (i in 1:d) {
    result <- diff(result)
  }
  return(result)
}

# Function to reverse differencing for ARIMA predictions
undiff_series <- function(diff_data, orig_data, d) {
  if (d == 0) return(diff_data)
  
  result <- diff_data
  for (i in 1:d) {
    n <- length(result)
    last_val <- tail(orig_data, n+1)[1]
    result <- c(last_val, last_val + cumsum(result))[-1]
  }
  return(result)
}

# Function to implement ARIMA model from scratch
arima_model <- function(data, p, d, q) {
  # Apply differencing
  diff_data <- diff_series(data, d)
  
  # Fit ARMA model to differenced data
  arma_fit <- arma_model(diff_data, p, q)
  
  return(list(
    arma_model = arma_fit,
    p = p,
    d = d,
    q = q,
    diff_data = diff_data,
    original_data = data
  ))
}

# Function to predict using ARIMA model
arima_predict <- function(model, h) {
  # Get ARMA predictions on differenced data
  diff_preds <- arma_predict(model$arma_model, model$diff_data, model$p, model$q, h)
  
  # Convert back to original scale
  undiff_preds <- undiff_series(diff_preds, model$original_data, model$d)
  
  return(undiff_preds)
}

# Implementation of ADF test from scratch
adf_test <- function(data, max_lag = 1) {
  n <- length(data)
  y <- diff(data)
  x <- data[-n]
  
  # Create matrix for lagged differences
  X <- matrix(1, nrow = length(y), ncol = max_lag + 2)
  X[, 2] <- x
  
  for (i in 1:max_lag) {
    X[, i+2] <- c(rep(NA, i), diff(data, lag = i)[-(1:i)])
  }
  
  # Remove rows with NAs
  valid_rows <- complete.cases(X)
  X <- X[valid_rows, ]
  y <- y[valid_rows]
  
  # Fit regression model
  model <- lm(y ~ X - 1)
  
  # Calculate ADF statistic
  coef <- coef(model)
  se <- summary(model)$coefficients[2, 2]
  adf_stat <- (coef[2] - 1) / se
  
  # Critical values (approximated from standard tables)
  critical_values <- c(
    "1%" = -3.43,
    "5%" = -2.86,
    "10%" = -2.57
  )
  
  # Calculate p-value (approximate)
  p_value <- 0.01
  if (adf_stat > critical_values["1%"]) {
    p_value <- 0.01
  } else if (adf_stat > critical_values["5%"]) {
    p_value <- 0.05
  } else if (adf_stat > critical_values["10%"]) {
    p_value <- 0.1
  } else {
    p_value <- 0.2
  }
  
  return(list(
    statistic = adf_stat,
    p_value = p_value,
    critical_values = critical_values
  ))
}

# Function to implement GARCH(1,1) model from scratch
garch_model <- function(returns) {
  n <- length(returns)
  
  # Initial parameter values
  omega <- var(returns) * 0.1
  alpha <- 0.1
  beta <- 0.8
  params <- c(omega, alpha, beta)
  
  # Optimization function
  garch_obj <- function(params) {
    omega <- params[1]
    alpha <- params[2]
    beta <- params[3]
    
    # Parameter constraints
    if (omega <= 0 || alpha < 0 || beta < 0 || alpha + beta >= 1) {
      return(1e10)
    }
    
    # Initialize conditional variance
    sigma2 <- numeric(n)
    sigma2[1] <- var(returns)
    
    # Calculate log-likelihood
    log_lik <- 0
    
    for (t in 2:n) {
      sigma2[t] <- omega + alpha * returns[t-1]^2 + beta * sigma2[t-1]
      log_lik <- log_lik + (log(sigma2[t]) + returns[t]^2/sigma2[t])
    }
    
    return(log_lik)
  }
  
  # Optimize parameters
  opt <- optim(params, garch_obj, method = "L-BFGS-B", 
               lower = c(1e-6, 1e-6, 1e-6), 
               upper = c(Inf, 1-1e-6, 1-1e-6))
  
  omega <- opt$par[1]
  alpha <- opt$par[2]
  beta <- opt$par[3]
  
  # Calculate final variances
  sigma2 <- numeric(n)
  sigma2[1] <- var(returns)
  
  for (t in 2:n) {
    sigma2[t] <- omega + alpha * returns[t-1]^2 + beta * sigma2[t-1]
  }
  
  return(list(
    omega = omega,
    alpha = alpha,
    beta = beta,
    variance = sigma2,
    log_likelihood = -opt$value
  ))
}

# Function to forecast volatility using GARCH model
garch_forecast <- function(model, returns, h) {
  n <- length(returns)
  
  # Get GARCH parameters
  omega <- model$omega
  alpha <- model$alpha
  beta <- model$beta
  
  # Last estimated variance
  last_var <- tail(model$variance, 1)
  
  # Forecast future variances
  forecast_var <- numeric(h)
  forecast_var[1] <- omega + alpha * tail(returns, 1)^2 + beta * last_var
  
  for (t in 2:h) {
    forecast_var[t] <- omega + (alpha + beta) * forecast_var[t-1]
  }
  
  return(list(
    variance = forecast_var,
    volatility = sqrt(forecast_var)
  ))
}

# Function to calculate MAE and RMSE
calc_metrics <- function(actual, predicted) {
  mae <- mean(abs(actual - predicted))
  rmse <- sqrt(mean((actual - predicted)^2))
  
  return(list(mae = mae, rmse = rmse))
}

# Main function to run forecasting for a cryptocurrency
run_forecasting <- function(ticker, name) {
  cat(paste0("\n", rep("=", 50), "\n"))
  cat(paste("Forecasting for", name, "(", ticker, ")\n"))
  cat(paste0(rep("=", 50), "\n"))
  
  # Fetch data using quantmod
  cat("\nDownloading data...\n")
  # Convert Yahoo Finance ticker format
  ticker_yf <- gsub("-", "\\-", ticker)
  data <- getSymbols(ticker_yf, from = "2020-01-01", to = "2023-12-31", source = "yahoo", auto.assign = FALSE)
  
  # Extract the close prices directly from the returned data
  close_data <- data[, 4]  # Close prices
  
  cat(paste("\nData size:", length(close_data), "\n"))
  cat("\nFirst few rows:\n")
  print(head(close_data))
  
  # Calculate returns for GARCH modeling
  returns <- diff(log(close_data))[-1]
  
  # Split into training and test sets (80% train, 20% test)
  n <- length(close_data)
  train_size <- floor(n * 0.8)
  train_data <- as.numeric(close_data[1:train_size])
  test_data <- as.numeric(close_data[(train_size+1):n])
  
  # Split returns data
  n_returns <- length(returns)
  returns_train_size <- floor(n_returns * 0.8)
  returns_train <- as.numeric(returns[1:returns_train_size])
  returns_test <- as.numeric(returns[(returns_train_size+1):n_returns])
  
  cat(paste("\nTrain data size:", length(train_data)))
  cat(paste("\nTest data size:", length(test_data), "\n"))
  
  # Plot the data
  png(paste0(gsub("-", "", ticker), "_price.png"), width = 800, height = 600)
  plot(as.numeric(close_data), type = "l", main = paste(name, "Close Price"),
       xlab = "Days", ylab = "Price (USD)")
  abline(v = train_size, col = "red", lty = 2)
  legend("topright", legend = "Train/Test Split", col = "red", lty = 2)
  dev.off()
  
  # Check stationarity with ADF test
  adf_result <- adf_test(train_data)
  cat("\nADF Test Results:\n")
  cat(paste("ADF Statistic:", round(adf_result$statistic, 4), "\n"))
  cat(paste("p-value:", adf_result$p_value, "\n"))
  cat("Critical Values:\n")
  print(adf_result$critical_values)
  
  if (adf_result$p_value <= 0.05) {
    cat("The time series is stationary (reject H0)\n")
    d_order <- 0
  } else {
    cat("The time series is not stationary (fail to reject H0)\n")
    cat("Using differencing order d=1 (common for financial time series)\n")
    d_order <- 1
  }
  
  # 1. AutoRegressive (AR) Model
  cat("\n--- AR Model ---\n")
  ar_fit <- ar_model(train_data, lags = 2)
  cat("\nAR(2) Model Coefficients:\n")
  print(ar_fit$coefficients)
  
  # Get AR predictions
  ar_pred <- ar_predict(ar_fit, train_data, lags = 2, h = length(test_data))
  
  # Calculate AR metrics
  ar_metrics <- calc_metrics(test_data, ar_pred)
  
  cat(paste("\nAR Model - MAE:", round(ar_metrics$mae, 2)))
  cat(paste("\nAR Model - RMSE:", round(ar_metrics$rmse, 2), "\n"))
  
  # Plot AR predictions
  png(paste0(gsub("-", "", ticker), "_ar_predictions.png"), width = 800, height = 600)
  plot(test_data, type = "l", main = paste(name, "- AR(2) Model Predictions"),
       xlab = "Days", ylab = "Price (USD)")
  lines(ar_pred, col = "blue", lty = 2)
  legend("topright", legend = c("Actual Close", "AR(2) Predicted Close"),
         col = c("black", "blue"), lty = c(1, 2))
  dev.off()
  
  # 2. Moving Average (MA) Model
  cat("\n--- MA Model ---\n")
  ma_fit <- ma_model(train_data, order = 2)
  cat("\nMA(2) Model Coefficients:\n")
  print(ma_fit$coefficients)
  
  # Get MA predictions
  ma_pred <- ma_predict(ma_fit, h = length(test_data))
  
  # Calculate MA metrics
  ma_metrics <- calc_metrics(test_data, ma_pred)
  
  cat(paste("\nMA Model - MAE:", round(ma_metrics$mae, 2)))
  cat(paste("\nMA Model - RMSE:", round(ma_metrics$rmse, 2), "\n"))
  
  # Plot MA predictions
  png(paste0(gsub("-", "", ticker), "_ma_predictions.png"), width = 800, height = 600)
  plot(test_data, type = "l", main = paste(name, "- MA(2) Model Predictions"),
       xlab = "Days", ylab = "Price (USD)")
  lines(ma_pred, col = "red", lty = 2)
  legend("topright", legend = c("Actual Close", "MA(2) Predicted Close"),
         col = c("black", "red"), lty = c(1, 2))
  dev.off()
  
  # 3. ARMA Model
  cat("\n--- ARMA Model ---\n")
  arma_fit <- arma_model(train_data, ar_order = 2, ma_order = 2)
  cat("\nARMA(2,2) Model Coefficients:\n")
  print(arma_fit$coefficients)
  
  # Get ARMA predictions
  arma_pred <- arma_predict(arma_fit, train_data, ar_order = 2, ma_order = 2, h = length(test_data))
  
  # Calculate ARMA metrics
  arma_metrics <- calc_metrics(test_data, arma_pred)
  
  cat(paste("\nARMA Model - MAE:", round(arma_metrics$mae, 2)))
  cat(paste("\nARMA Model - RMSE:", round(arma_metrics$rmse, 2), "\n"))
  
  # Plot ARMA predictions
  png(paste0(gsub("-", "", ticker), "_arma_predictions.png"), width = 800, height = 600)
  plot(test_data, type = "l", main = paste(name, "- ARMA(2,2) Model Predictions"),
       xlab = "Days", ylab = "Price (USD)")
  lines(arma_pred, col = "green", lty = 2)
  legend("topright", legend = c("Actual Close", "ARMA(2,2) Predicted Close"),
         col = c("black", "green"), lty = c(1, 2))
  dev.off()
  
  # 4. ARIMA Model
  cat("\n--- ARIMA Model ---\n")
  
  # Manual fitting process - try different p, d, q combinations
  best_aic <- Inf
  best_order <- NULL
  
  for (p in 0:2) {
    for (q in 0:2) {
      tryCatch({
        # Fit ARIMA model
        temp_model <- arima_model(train_data, p, d_order, q)
        
        # Calculate AIC (approximated)
        n_eff <- length(temp_model$diff_data)
        k <- p + q + 1
        residuals <- temp_model$arma_model$errors[!is.na(temp_model$arma_model$errors)]
        aic <- 2 * k + n_eff * log(sum(residuals^2, na.rm = TRUE) / n_eff)
        
        if (aic < best_aic) {
          best_aic <- aic
          best_order <- c(p, d_order, q)
          cat(paste("New best AIC:", round(best_aic, 2), "with order", 
                    paste("(", p, ",", d_order, ",", q, ")", sep = "")), "\n")
        }
      }, error = function(e) {
        # Skip failed models
      })
    }
  }
  
  cat(paste("\nBest ARIMA order:", paste(best_order, collapse = ",")))
  
  # Fit best ARIMA model
  arima_fit <- arima_model(train_data, best_order[1], best_order[2], best_order[3])
  
  # Get ARIMA predictions
  arima_pred <- arima_predict(arima_fit, h = length(test_data))
  
  # Calculate ARIMA metrics
  arima_metrics <- calc_metrics(test_data, arima_pred)
  
  cat(paste("\nARIMA Model - MAE:", round(arima_metrics$mae, 2)))
  cat(paste("\nARIMA Model - RMSE:", round(arima_metrics$rmse, 2), "\n"))
  
  # Plot ARIMA predictions
  png(paste0(gsub("-", "", ticker), "_arima_predictions.png"), width = 800, height = 600)
  plot(test_data, type = "l", 
       main = paste(name, "- ARIMA(", best_order[1], ",", best_order[2], ",", best_order[3], ") Model Predictions", sep = ""),
       xlab = "Days", ylab = "Price (USD)")
  lines(arima_pred, col = "purple", lty = 2)
  legend("topright", 
         legend = c("Actual Close", 
                    paste("ARIMA(", best_order[1], ",", best_order[2], ",", best_order[3], ") Predicted Close", sep = "")),
         col = c("black", "purple"), lty = c(1, 2))
  dev.off()
  
  # 5. GARCH Model for Volatility
  cat("\n--- GARCH Model ---\n")
  
  # Fit GARCH(1,1) model
  garch_fit <- garch_model(returns_train)
  cat("\nGARCH(1,1) Model Parameters:\n")
  cat(paste("omega:", round(garch_fit$omega, 6), "\n"))
  cat(paste("alpha:", round(garch_fit$alpha, 6), "\n"))
  cat(paste("beta:", round(garch_fit$beta, 6), "\n"))
  
  # Forecast volatility
  garch_forecast_result <- garch_forecast(garch_fit, returns_train, h = length(returns_test))
  
  # Plot GARCH volatility forecast
  png(paste0(gsub("-", "", ticker), "_garch_volatility.png"), width = 800, height = 600)
  plot(returns_test, type = "l", main = paste(name, "- GARCH(1,1) Volatility Forecast"),
       xlab = "Days", ylab = "Returns / Volatility", col = "grey", ylim = range(c(returns_test, garch_forecast_result$volatility)))
  lines(garch_forecast_result$volatility, col = "red", lwd = 2)
  legend("topright", legend = c("Actual Returns", "GARCH(1,1) Volatility Forecast"),
         col = c("grey", "red"), lty = c(1, 1), lwd = c(1, 2))
  dev.off()
  
  # Compare all models
  png(paste0(gsub("-", "", ticker), "_model_comparison.png"), width = 1000, height = 600)
  plot(test_data, type = "l", main = paste(name, "- Model Comparison"),
       xlab = "Days", ylab = "Price (USD)", lwd = 2)
  lines(ar_pred, col = "blue", lty = 2)
  lines(ma_pred, col = "red", lty = 3)
  lines(arma_pred, col = "green", lty = 4)
  lines(arima_pred, col = "purple", lty = 5)
  legend("topright", 
         legend = c("Actual Close", "AR(2) Prediction", "MA(2) Prediction", 
                    "ARMA(2,2) Prediction", 
                    paste("ARIMA(", best_order[1], ",", best_order[2], ",", best_order[3], ") Prediction", sep = "")),
         col = c("black", "blue", "red", "green", "purple"), 
         lty = c(1, 2, 3, 4, 5))
  dev.off()
  
  # Return the metrics for comparison
  return(list(
    asset = name,
    ticker = ticker,
    ar_mae = ar_metrics$mae,
    ar_rmse = ar_metrics$rmse,
    ma_mae = ma_metrics$mae,
    ma_rmse = ma_metrics$rmse,
    arma_mae = arma_metrics$mae,
    arma_rmse = arma_metrics$rmse,
    arima_mae = arima_metrics$mae,
    arima_rmse = arima_metrics$rmse,
    arima_order = best_order
  ))
}

# Run forecasting for Bitcoin and Ethereum
cat("Starting cryptocurrency forecasting...\n")
bitcoin_results <- run_forecasting("BTC-USD", "Bitcoin")
ethereum_results <- run_forecasting("ETH-USD", "Ethereum")

# Compare results
cat("\n\n")
cat(rep("=", 50), "\n")
cat("MODEL COMPARISON SUMMARY\n")
cat(rep("=", 50), "\n")

cat("\nMAE Comparison (lower is better):\n")
cat(sprintf("%-10s %-10s %-10s %-10s %-10s\n", "Asset", "AR(2)", "MA(2)", "ARMA(2,2)", "ARIMA"))
cat(sprintf("%-10s %-10.2f %-10.2f %-10.2f %-10.2f\n", 
            bitcoin_results$asset, bitcoin_results$ar_mae, bitcoin_results$ma_mae, 
            bitcoin_results$arma_mae, bitcoin_results$arima_mae))
cat(sprintf("%-10s %-10.2f %-10.2f %-10.2f %-10.2f\n", 
            ethereum_results$asset, ethereum_results$ar_mae, ethereum_results$ma_mae, 
            ethereum_results$arma_mae, ethereum_results$arima_mae))

cat("\nRMSE Comparison (lower is better):\n")
cat(sprintf("%-10s %-10s %-10s %-10s %-10s\n", "Asset", "AR(2)", "MA(2)", "ARMA(2,2)", "ARIMA"))
cat(sprintf("%-10s %-10.2f %-10.2f %-10.2f %-10.2f\n", 
            bitcoin_results$asset, bitcoin_results$ar_rmse, bitcoin_results$ma_rmse, 
            bitcoin_results$arma_rmse, bitcoin_results$arima_rmse))
cat(sprintf("%-10s %-10.2f %-10.2f %-10.2f %-10.2f\n", 
            ethereum_results$asset, ethereum_results$ar_rmse, ethereum_results$ma_rmse, 
            ethereum_results$arma_rmse, ethereum_results$arima_rmse))

cat("\nBest ARIMA Models:\n")
cat(sprintf("%s: ARIMA(%s)\n", bitcoin_results$asset, paste(bitcoin_results$arima_order, collapse = ",")))
cat(sprintf("%s: ARIMA(%s)\n", ethereum_results$asset, paste(ethereum_results$arima_order, collapse = ",")))

# Determine best model for each asset
get_best_model <- function(results) {
  # Create model names
  arima_name <- paste0("ARIMA(", paste(results$arima_order, collapse = ","), ")")
  
  # Create vector of MAE values
  mae_values <- c(results$ar_mae, results$ma_mae, results$arma_mae, results$arima_mae)
  
  # Create names for the vector
  names(mae_values) <- c("AR(2)", "MA(2)", "ARMA(2,2)", arima_name)
  
  # Return the name of the model with minimum MAE
  return(names(which.min(mae_values)))
}

cat("\nBest Models (based on lowest MAE):\n")
cat(sprintf("%s: %s\n", bitcoin_results$asset, get_best_model(bitcoin_results)))
cat(sprintf("%s: %s\n", ethereum_results$asset, get_best_model(ethereum_results)))

cat("\nDiscussion:\n")
cat("The forecasting results demonstrate the varying effectiveness of different time series models\n")
cat("in predicting cryptocurrency prices. Cryptocurrencies are known for their high volatility,\n")
cat("which poses challenges for traditional forecasting methods.\n")
cat("\nARIMA models extend ARMA by incorporating differencing to handle non-stationarity,\n")
cat("which is common in cryptocurrency price data. The optimal differencing order is determined\n")
cat("through the ADF test and manual inspection.\n")
cat("\nGARCH models specifically target volatility forecasting, capturing the tendency of crypto\n")
cat("markets to experience periods of high volatility followed by relative calm (volatility clustering).\n")
cat("\nWhile price level forecasting provides directional insights, volatility forecasting with GARCH\n")
cat("is particularly valuable for risk management and options pricing in cryptocurrency markets.")