library(quantmod)
library(rugarch)
library(tseries)
library(parallel)
library(lattice)

# getting S&P 500 data
sp500 <- getSymbols('^GSPC', src = "yahoo", from = '2003-01-01', to = '2023-12-31', auto.assign = FALSE)
plot(sp500$GSPC.Close)

# check for missing values
print(sum(is.na(sp500)))

# compute daily log returns
sp500daily_log <- dailyReturn(sp500$GSPC.Close, type = 'log')
print(sum(is.na(sp500daily_log)))
plot(sp500daily_log)

window_length <- 200 #rolling window
forecast_length <- length(sp500daily_log) - window_length
forecasts <- vector(mode = 'character', length = forecast_length)

# fit ARIMA model function
fit_best_arima <- function(data) {
  best_aic <- Inf
  best_arima <- NULL
  for (p in 0:5) {
    for (q in 0:5) {
      if (p == 0 && q == 0) {
        next
      }
      arimaFit = tryCatch(arima(data, order = c(p, 0, q)), error = function(err) NULL, warning = function(err) NULL)
      if (!is.null(arimaFit)) {
        current_aic <- AIC(arimaFit)
        if (current_aic < best_aic) {
          best_aic <- current_aic
          best_arima <- arimaFit
        }
      }
    }
  }
  return(best_arima)
}

# fit GARCH model function
fit_garch_model <- function(data, arima_model) {
  if (is.null(arima_model)) {
    warning("ARIMA model is NULL, skipping GARCH fitting.")
    return(NULL)
  }
  spec <- ugarchspec(
    variance.model = list(garchOrder = c(1, 1)),
    mean.model = list(armaOrder = c(arima_model$arma[1], arima_model$arma[3]), include.mean = TRUE),
    distribution.model = "sged"
  )
  fit = tryCatch(
    ugarchfit(spec, data, solver = 'hybrid'),
    error = function(e) { warning("GARCH fitting failed: ", e$message); NULL },
    warning = function(w) { warning("GARCH fitting warning: ", w$message); NULL }
  )
  return(fit)
}

# create forecast function
create_forecast <- function(garch_fit, data) {
  if (is.null(garch_fit)) {
    return(paste(index(data[length(data)]), "no convergence", sep = ","))
  }
  fore = ugarchforecast(garch_fit, n.ahead = 1)
  predicted_value = fore@forecast$seriesFor
  return(paste(index(data[length(data)]), ifelse(predicted_value[1] < 0, -1, 1), sep = ","))
}

# create cluster for running loop
cl <- makeCluster(detectCores() - 1)  # use all cores but one
clusterEvalQ(cl, {
  library(quantmod)
  library(rugarch)
  library(tseries)
})
clusterExport(cl, c("sp500daily_log", "window_length", "fit_best_arima", "fit_garch_model", "create_forecast"))

# loop for signals
results <- parLapply(cl, 1:forecast_length, function(d) {
  sp_returns_offset = sp500daily_log[d:(d + window_length - 1)]
  final_arima = fit_best_arima(sp_returns_offset)
  final_garch = fit_garch_model(sp_returns_offset, final_arima)
  create_forecast(final_garch, sp_returns_offset)
})
stopCluster(cl)
results <- as.character(results)
split_strings <- strsplit(results, ",")
output <- data.frame(
  Date = sapply(split_strings, `[`, 1),
  Value = as.numeric(sapply(split_strings, `[`, 2))
)

#Shift forecasts to next day
output$Date <- as.Date(output$Date)
xts_forecasts <- xts(output$Value, order.by = output$Date)
shifted_xts_forecasts <- lag(xts_forecasts, k = 1)
shifted_xts_forecasts <- na.fill(shifted_xts_forecasts, fill = 1)

# Create the ARIMA+GARCH returns
spIntersect = merge( shifted_xts_forecasts[,1], sp500daily_log, all=F )
spArimaGarchReturns = spIntersect[,1] * spIntersect[,2]

# Create the backtests for ARIMA+GARCH and Buy & Hold
spArimaGarchCurve = log( cumprod( 1 + spArimaGarchReturns ) )
spBuyHoldCurve = log( cumprod( 1 + spIntersect[,2] ) )
spCombinedCurve = merge( spArimaGarchCurve, spBuyHoldCurve, all=F )



# Plot the equity curves
xyplot( 
  spCombinedCurve,
  superpose=T,
  col=c("darkgreen", "darkblue"),
  lwd=2,
  key=list( 
    text=list(
      c("ARIMA+GARCH", "Buy & Hold")
    ),
    lines=list(
      lwd=2, col=c("darkgreen", "darkblue")
    )
  )
)

