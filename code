## Libraries
library(tidyverse)
library(dplyr)
library(mice)
library(ggplot2)
library(ggthemes)
library(RColorBrewer)
library(rworldmap)
library(tseries)
library(grid)
library(gridExtra)
library(fpp2)
library(sp)
library(Kendall)
require(pracma)
library(randtests)
# Temperatures change in various locations of world map between time range 1850 - 2012

world_Map <- fortify(map_data("world"), region = "region")
wmap <- ggplot() + 
  geom_map(data = world_Map, map = world_Map, aes(x = long, y = lat, map_id = region, group = group),fill = "white", color = "black", size = 0.2)


# Scanning Data from CSV file
cities_data<-read.csv("GlobalLandTemperaturesByCity.csv")
dim(cities_data) # Data is quite Large in dimension
head(cities_data)

# Since the data is Large we decided to trim the data and take the data of year 1850 onwards 
# also we are removing the missing data rows in our analysis.
cities_data$date <- as.Date(cities_data$dt)
cities_data$year <- as.numeric(format(cities_data$date,'%Y'))
cities_data <-as.data.frame(cities_data %>% filter(year>=1850))
cities_data <-na.omit(cities_data)
dim(cities_data)

# To visualize the world map we need to format the data in a certain way
cities_data$month<-as.numeric(format(cities_data$date,'%m'))
cities_data$Longitude<-as.character(cities_data$Longitude)
cities_data$Latitude<-as.character(cities_data$Latitude)


Numeric_Coordinates<-function(x)
{
  extension <- substr(x,nchar(x),nchar(x))
  y <- substr(x,1,nchar(x)-1)
  value <- as.numeric(char2dms(paste(strsplit(y,'[.]')[[1]][1],'d',strsplit(y,'[.]')[[1]][1],"'",extension, sep = "")))
  return(value)
}

Initial <- cities_data %>% filter(year==1850)
Initial$long <- sapply(Initial$Longitude,Numeric_Coordinates) 
Initial$lat <- sapply(Initial$Latitude,Numeric_Coordinates)
Initial <- as.data.frame(Initial %>% group_by(Country, City) %>% select(AverageTemperature, City ,lat, long, Country) %>% summarise(avg_temp_initial = mean(AverageTemperature), long = mean(long), lat = mean(lat)))

Final <- as.data.frame(cities_data %>% filter(year==2012))
Final <- as.data.frame(Final %>% group_by(Country,City) %>% select(AverageTemperature,City) %>% summarise(avg_temp_final = mean(AverageTemperature)))

Plotdata <- as.data.frame(merge(Initial, Final, by=c('Country','City')))

wmap + 
  geom_point(data = Plotdata,aes(x=long, y=lat, size=avg_temp_final - avg_temp_initial,color=avg_temp_final - avg_temp_initial),alpha=.3) + 
  theme_fivethirtyeight() + 
  ggtitle('Temperature Change from 1850 to 2012') + theme(axis.text = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + labs(size = '') + 
  scale_color_gradientn(name ='Degrees Celsius',colors=rev(brewer.pal(10,'Spectral'))) + scale_size(guide = 'none')



#### Inference/Interpretation

# The darker the color of a country, the more increase in temperature was recorded.
# Western countries like US, Russia & some European Countries show red color indicating heavy change in temperature but 
# India has not shown any drastic change when the temperatures were compared.

rm(list = ls()) # Computation done so far did not used in future inference, so we will not need that datasets anymore.

# To print the mean and median
mean_median1 = function(data)
{
  k <-  data %>%
    summarize(average = mean(data$AverageTemperature,na.rm = TRUE), median = median(data$AverageTemperature, na.rm= TRUE))
  cat("Statistics for Average Temperature:\n")
  k
  
}

mean_median2 = function(data)
{
  l <-  data %>%
    summarize(average = mean(data$AverageTemperatureUncertainty,na.rm = TRUE), median = median(data$AverageTemperatureUncertainty, na.rm= TRUE))
  cat("Statistics for Average Temperature Uncertainity:\n")
  l
}


# To check if any NA value exists
check_na = function(data,col){
  print(paste("Col name: ", col))
  sum(is.na(data[col]))
}


## Loading of new dataset and Data pre-processing
country_data = read.csv("GlobalLandTemperaturesByCountry.csv")
dim(country_data)
head(country_data)


data_INDIA = country_data %>% filter(Country == "India")  %>% 
  separate(col = dt, into = c("Year", "Month", "Day"), convert = TRUE) -> cData
rownames(data_INDIA) = 1:nrow(data_INDIA)

## Data of India are only extracted
## splitting the date column (dt) into Year,Month and Day

dim(data_INDIA)
summary(data_INDIA)
str(data_INDIA)


## checking if any values exist in the columns

check_na(data_INDIA, "AverageTemperature")
check_na(data_INDIA, "Year")

mean_median1(data_INDIA)
mean_median2(data_INDIA)

## Data only in and after 1850. Range 1850-2013
data_INDIA = data_INDIA %>% filter(Year >= 1850)

check_na(data_INDIA, "AverageTemperature")
check_na(data_INDIA, "Year")
check_na(data_INDIA, "Month")
mean_median1(data_INDIA)
mean_median2(data_INDIA)

## Imputating the missing temperature values with median

imp_model <- mice(as.data.frame(data_INDIA[,c(4,5)]), method = "pmm")
data_imputed <- complete(imp_model)
mean_median1(data_imputed)
mean_median2(data_imputed)
data_INDIA[, c(4,5)] <- data_imputed


#### Now the column dt is seperated into year, month and date and the rows with na values in 
# Average Temperature Column are imputated with the method pmm (Predictive Mean Matching)

## this stores the rows where year >= 2000
India_2000 = data_INDIA %>% filter(Year >=2000)
str(India_2000)
mean_median1(India_2000)
mean_median2(India_2000)

# Converting the data India_2000 into ts data frame
Y = ts(India_2000$AverageTemperature,start = c(2000,1),frequency = 12)

## Storing yearly average Temperature
Ind_avgYr = data_INDIA %>% group_by(Year) %>% summarise(Temperature  = mean(AverageTemperature)) 

## Storing monthly average Temperature
Ind_avgMonth = data_INDIA %>% group_by(Month) %>% summarise(Temperature  = mean(AverageTemperature)) 

## Storing the maximum temperature for each year
Ind_maxYr1 = data_INDIA %>% group_by(Year) %>% summarise(Temperature  = max(AverageTemperature)) 

## Storing the minimum temperature for each year
Ind_minYr1 = data_INDIA %>% group_by(Year) %>% summarise(Temperature = min(AverageTemperature)) 

Ind_maxYr = Ind_maxYr1[!(Ind_maxYr1$Year=='1862' | Ind_maxYr1$Year=='1863'| Ind_maxYr1$Year=='1864'),]
Ind_minYr = Ind_minYr1[!(Ind_minYr1$Year=='1862' | Ind_minYr1$Year=='1863'| Ind_minYr1$Year=='1864'),]

check_na(data_INDIA, "AverageTemperature")
check_na(data_INDIA, "Year")

print(paste("Max avg temp: ",max(data_INDIA$AverageTemperature)))
print(paste("Min avg temp: ",min(data_INDIA$AverageTemperature)))

# Converting the data into ts dataset
tsdata_full = ts(data_INDIA$AverageTemperature, frequency = 12)
## Now we decompose a time series into seasonal, trend and irregular components using moving averages.
decdata_full1 = decompose(tsdata_full, "multiplicative")
decdata_full2 = decompose(tsdata_full, "additive")

#Trend Estimation 
MannKendall(tsdata_full) #tau = 0.062, 2-sided pvalue =3.8028e-05 #Trend present
MannKendall(diff(log(tsdata_full))) #detrended --- Order 1
#Trend is present

# Testing Stationarity
adf.test(tsdata_full) # Our data is stationary

# Testing for Seasonality
friedman.test(tsdata_full)

# Randomness Estimation
turning.point.test(tsdata_full) #p-value < 2.2e-16 # Non Random

 ## Plots and Analysis 

plot(decdata_full1) ## Checking trends, seasonality and randomness of the data. Trend can clearly be seen
plot(decdata_full2)
# One can easily see that error term in multiplicative model has mean 0 (approx.) and that of additive model is 1(approx.)
# So We are taking additive model for further analysis 
decdata_full = decdata_full1

### Here we can see the trend clearly but seasonality is not visible properly hence 
# further investigation is required for seasonality. Clearly, Trend shows that the temperature is increasing 
plot(data_INDIA$Year,data_INDIA$AverageTemperature, xlab = "Year", ylab = "Average Temperature", type = "l", ylim = c(15.27,31.329), main = "Average Monthy Temperature vs Year", col = "red") #Monthly Avg Temp
## In the above plot no trend or seasonality is visible because of many data points

### Let us see how our time series looks loke if we remove the trend (By first order differencing)
# Here also all the data points are plotted hence no seasonality or trend is observed.
plot(10 *diff(log(data_INDIA$AverageTemperature)), xlab = "year", ylab = "Differencing of log(Average Temperature)", type="l",lwd=1, ylim=c(-6.2,6.2), main = "Avg Temperature Plot", col = "blue")

# Following plot ensures the healthy seasonality component in the data
plot(India_2000$Year,India_2000$AverageTemperature, xlab = "Year", ylab = "Average Temperature", type = "l", ylim = c(16,32), main = "Average Monthy Temperature vs Year (Small Data)")


### Here the avg temperature of all the years are plotted and the trends becomes clearly visible.
qplot(Year, Temperature, data = Ind_avgYr, main = "Average Temperature in India (1850-2013) by year", geom = c("point","smooth"))+ aes(colour = Temperature) + scale_color_gradient(low = "purple", high = "green") ## Temperature Increasing trend can be seen here


### Here the avg temperature of all the months are plotted and the seasonality becomes clearly visible.
qplot(Month, Temperature, data = Ind_avgMonth, main = "Average Temperature in India (1850-2013) by month", geom = c("point","smooth"))+ aes(colour = Temperature) + scale_color_gradient(low = "purple", high = "green") ## monthly seasonality where temp increases till May which is its peek and then decreases


# Monthly seasonality where temp increases till May which is its peek and then decreases
ggseasonplot(window(Y,2000,2013)) + ggtitle("Seasonality [2000-2013]") 
# Above plot Confirming seasonality for years from 2000 to 2013


### Seasonality remains same for all years (Seen Above) from 2000 to 2013 where temp increases till may and then decreases
plot(Ind_avgYr$Year, Ind_avgYr$Temperature, xlab = "Year", ylab = "Average Temperature", type = "l",main = "Avg Yearly Temperature vs Year") #Yearly Avg Temp ## Increasing trend



### Here the Maximum/Minimum temperature of all the years are plotted and the trends becomes clearly visible which is showing an increasing trend.
plot(Ind_maxYr$Year,Ind_maxYr$Temperature,xlab = "Year", ylab = "Maximum Temperature", type = "l", main = "Maximum Yearly Temperature vs Year") #Yearly Max Temp

plot(Ind_minYr$Year,Ind_minYr$Temperature, xlab = "Year", ylab="Min Temp", type = "l", main = "Minimum Yearly Temperature vs Year") #Yearly Min Temp

qplot(Year, Temperature, data=Ind_maxYr, main = "Maximum Yearly Temperature in India (1850-2013)", geom = c("point","smooth"))+ aes(colour = Temperature) + scale_color_gradient(low = "purple", high = "green")
qplot(Year, Temperature, data=Ind_minYr, main="Minimum yearly Temperature in India (1850-2013)",geom=c("point","smooth"))+ aes(colour = Temperature) + scale_color_gradient(low="purple", high="green")

### Now the yearly maximum and minimum temperatures are plotted and even they show same trend which means that our inference on the avg temperature above was correct.
# 40 yr splits of years

b_1850_1890 = data_INDIA %>%
  filter(Year >= 1850 & Year<1890)

b_1890_1930 = data_INDIA %>%
  filter(Year >=1890 & Year<1930)

b_1930_1970 = data_INDIA %>%
  filter(Year >=1930 & Year<1970)

b_1970_2013 = data_INDIA %>%
  filter(Year >=1970 & Year<2013)



plot(b_1850_1890$AverageTemperature,xlab="Year",ylab="Average Temp",type = "l",main = "Avg Monthy Temperature between 1850 and 1890")

plot(b_1890_1930$AverageTemperature,xlab="Year",ylab="Average Temp",type = "l",main = "Avg Monthy Temperature between 1890 and 1930")

plot(b_1930_1970$AverageTemperature,xlab="Year",ylab="Average Temp",type = "l",main = "Avg Monthy Temperature between 1930 and 1970")

plot(b_1970_2013$AverageTemperature,xlab="Year",ylab="Average Temp",type = "l",main = "Avg Monthy Temperature between 1970 and 2013")

## We can observe from the 40 yr splits that same pattern is being observed. The temperature increases to a peak which can be considered as the month of May and then decreases. This continues.

# The temperature range gradually increases as the year reaches 1930.
# This is also seen by seeing range of the graph
# Initially the lower is 15 but then it changes to 20 indicating rise in temperature.

# Seeing boxplots on different yrs
Limit_Year = data_INDIA %>% filter(Year==1850 | Year==1890 | Year==1930 | Year==1970 | Year==2013)


Limit_Year$Year <- as.factor(Limit_Year$Year)
qplot(x =  Year, y = AverageTemperature, data = Limit_Year) + ggtitle("Average Temperature Comparison for 40 Year Intervals")+geom_boxplot(fill="cyan")
## Box plots show temperature steadily has increased

# Above 5 years data was taken as a difference of 40 years 
# It is observable that that both the range and median increases
# This further confirms the trend

# Taking particular yearr - 2000 (say)

temp_2000 = data_INDIA %>% filter(Year == 2000)

plot(temp_2000$Month,temp_2000$AverageTemperature,xlab="months",ylab="Average Temp",type = "l",main = "Avg Monthy Temperature of 2000", ylim = c(min(temp_2000$AverageTemperature), max(temp_2000$AverageTemperature)))


### Here seasonality is checked on the year 2000 and same thing is observed.

### BoxPlots

## Yearly
quantile(data_INDIA$AverageTemperature, probs = c(0,0.25,0.5,0.75,1))
boxplot(data_INDIA$AverageTemperature ~ data_INDIA$Year, main = "Average Temperature",
        ylab = "Avg Temp", xlab = "Years", las = 1)

boxplot(India_2000$AverageTemperature ~ India_2000$Year, main = "Average Temperature from 2000 to 2012",
        ylab = "Avg Temperature", xlab = "Years", las = 1, pch = 16, col = "cyan")

### Boxplots are made on Yearly Average Temperatures and the median increases over the years



## Monthly
boxplot(data_INDIA$AverageTemperature ~ data_INDIA$Month, main = "Average Temperature",
        ylab = "Avg Temp", xlab = "Months", las = 1, pch = 16, col = "cyan") ## again may has the hight temperature range



### Monthly boxplots

# Yearly - Differencing for converting to stationary time series
avg_time = ts(Ind_avgYr$Temperature,start = min(Ind_avgYr$Year), end=max(Ind_avgYr$Year), frequency = 1)
MannKendall(diff(log(avg_time))) #Detrended
adf.test(diff(log(avg_time))) # After removing trend, our data is stationary.
turning.point.test(((avg_time))) # Non randomness
plot(avg_time, xlab = "Years", ylab = "Average Temperature", main = "Non-Stationary Time Series") ## Non stationary time series as mean and variance does not remain same for any 2 periods
plot(log(avg_time),type = "l", xlab = "Years", ylab = "log(Average Temperature)")
plot((diff(log(avg_time))),type = "l", xlab = "Years",ylab = "Time Series Produced by Differncing" , main = "Stationary Time Series") ## converted to stationary time series

## The initial time series is not stationary meaning the mean and variance any 2 time periods is not same.
## We convert it into stationary time series by differencing the log values.


## We will use Yearly average temperature to make the model and forecast values
## ARIMA Model 

## AR I  MA
## q  d  p

# p = acf
# q = pacf

acf(avg_time) ## As non stationary time series, all lines are above the blue limit line
pacf(avg_time)


## Most of the lines between the line after converting to stationary
acf(diff(log(avg_time))) 
pacf(diff(log(avg_time)))
# both plots are tail off
auto.arima((avg_time)) # We can check with auto.arima is giving same model i.e ARIMA(0,1,1) 

### For non stationary time series-
# All the lines in acf graph are above the blue line

### For stationary time series-
# Most of the lines fit between the blue lines in both acf and padf graph
monthly_avg_time = ts(data_INDIA$AverageTemperature,start = min(data_INDIA$Year), end=max(data_INDIA$Year), frequency = 12)
plot(log(monthly_avg_time),type = "l", xlab = "Years", ylab = "log(Average Temperature)")  ## Non stationary time series as mean and variance does not remain same for any 2 periods
plot((diff(log(monthly_avg_time))),type = "l", xlab = "Years",ylab = "Time Series Produced by Differncing" , main = "Stationary Time Series") ## converted to stationary time series

## Model
model1 = auto.arima(log(avg_time), ic = "aic", trace = T)
model1     

acf(monthly_avg_time)# All the lines in acf graph are above the blue line
pacf(monthly_avg_time)

MannKendall((diff(((monthly_avg_time))))) #detrend
adf.test((diff((monthly_avg_time)))) # Stationary after trend removal
turning.point.test(diff(monthly_avg_time)) # Non-random
acf((diff(log(monthly_avg_time)))) 
pacf(diff(log(monthly_avg_time)))
model2 = auto.arima(diff(monthly_avg_time), ic = "aic", trace = T) # ARIMA(2,1,0)
auto.arima(monthly_avg_time)
auto.arima(Ind_avgYr$Temperature, ic = "aic", trace= T) 
plot.ts(model1$residuals)


# Here using the auto.arima() function, the best model obtained is ARIMA(0,1,1). 
# The 1 in center denotes the seasonality observed. It has the least AIC and BIC Values and maximum log liklihood confirming it to be most suitable model.

# The residuals when plotted by acf and pacf show very less auto-correlation between them.

## Forecast next 10 years


forecast1 = forecast(model1, level=c(95),h = 10)
plot(forecast1) ## the trend continues as the avg temperature continues to increase on yearly basis
print(forecast1)
forecast_data = as.data.frame(forecast1)
print(paste("Avg Temperature in 2020: ",forecast_data$`Point Forecast`[7]))
print(paste("Avg Temperature in 2021: ",forecast_data$`Point Forecast`[8]))

print(paste("Avg Temperature in 2022: ",forecast_data$`Point Forecast`[9]))

### The forecasts although not 100% accurate, are pretty close to real values.
### The forecast for 2021 only varies by 1 degree.
### The forecasted values in plot satisfy the trend that the temperature increases steadily.

## Validation


Box.test(model1$residuals, lag=5, type="Ljung-Box")

Box.test(model1$residuals, lag=10, type="Ljung-Box")

Box.test(model1$residuals, lag=15, type="Ljung-Box")

## p-values here are all above 0.05 hence good model
