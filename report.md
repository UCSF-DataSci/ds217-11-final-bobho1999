# Chicago Beach Weather Sensors Analysis
## Executive Summary

This analysis examines weather sensor data from Chicago beaches along Lake Michigan, covering 196,271 hourly measurements from April 2015 to December 2025 across three weather stations. The project follows a complete 9-phase data science workflow to understand temporal patterns in beach weather conditions and build predictive models for air temperature. Key findings include strong seasonal temperature patterns, significant daily cycles, and successful prediction models. The XGBoost model emerged as the best performer, with a test R² of 0.8102 and RMSE of 4.37°C, demonstrating that air temperature can be predicted with good accuracy from temporal features, rolling windows of predictor variables, and weather variables.

## Phase-by-Phase Findings

### Phase 1-2: Exploration

Initial exploration revealed a dataset of **196,271 records** with 18 columns including temperature measurements (air and wet bulb), wind speed and direction, humidity, precipitation, barometric pressure, solar radiation, and sensor metadata. The data spans from April 25, 2015 to December 2, 2025, with measurements from three different weather stations: 63rd Street Weather Station, Foster Weather Station, and Oak Street Weather Station.

**Key Data Quality Issues Identified:**
- 75 missing values in Air Temperature (0.04%)
- 75,926 missing values in Wet Bulb Temperature (38.68%) - significant portion of data
- Missing values in Wet Bulb Temperature, Rain Intensity, Total Rain, Precipitation Type, and Heading (same 75,926 records) all from Foster Weather Station
- 146 missing values in Barometric Pressure
- 13,425 negative values in Solar Radiation
- Some outliers in Air Temperature, Wet Bulb Temperature, Humidity, Wind Speed, Barometric Pressure, and Battery Life measurements
- Data collected at hourly intervals with some gaps

Initial visualizations showed:
- Air temperature ranging from approximately -20°C to 35°C
- Clear seasonal patterns visible in temperature data
- 

![Figure 1: Initial Data Exploration](output/q1_visualizations.png)
*Figure 1: Initial exploration visualizations showing distributions of air temperature, air temperature time series.*

### Phase 3: Data Cleaning

Data cleaning addressed missing values, outliers, and data type validation. After examining the dataset, it seems like all missing rainfall-related measurements, such as Rain Intensity, Total Rain, and Precipitation Type, occur exclusively at the Foster Weather Station, while the other two stations (Oak Street and 63rd Street) contain nearly complete records. Because these beach weather stations are located along the same Chicago lakefront and experience highly correlated weather patterns, I imputed Foster’s missing rainfall-related values, as well as Wet Bulb Temperature, using the hourly mean value (or mode for Precipitation type and Heading) from the other two stations. This approach provides a physically reasonable estimate of rainfall conditions at Foster during each timestamp while avoiding unrealistic assumptions produced by simple constant filling. For hours when neither of the other stations recorded a usable value, I applied a forward fill as a fallback, which preserves temporal continuity without injecting future rainfall events into earlier time periods. This two-step strategy balances meteorological realism with data completeness and ensures that modeling is not biased by systematic under-reporting or artificial rain events at the Foster station.

**Cleaning Results:**
- Rows before cleaning: **196,271**
- Missing values: Mean/mode-imputed and forward-filled
  - Wet Bulb Temperature: 75,926 missing → 0 missing (large gap, likely sensor-specific)
  - Rain Intensity: 75,926 missing → 0 missing (large gap, likely sensor-specific)
  - Total Rain: 75,926 missing → 0 missing (large gap, likely sensor-specific)
  - Precipitation Type: 75,926 missing → 0 missing (large gap, likely sensor-specific)
  - Heading: 75,926 missing → 0 missing (large gap, likely sensor-specific)
- Missing values: Forward-filled and median-imputed
  - Air Temperature: 75 missing → 0 missing
  - Barometric Pressure: 146 missing → 0 missing
- Outliers: Capped using IQR method (3×IQR bounds)
  - Air Temperature: 97 outliers capped (bounds: [-21.5, 47.3])
  - Wet Bulb Temperature: 144 outliers capped (bounds: [-20.6, 41.5])
  - Humidity: 185 outliers capped (bounds: [22.5, 114.5])
  - Battery Life: 6 outliers capped (bounds: [7.1, 19.9])
- Outliers: Capped using domain knowledge
  - Wind Speed: 5 outliers capped (bounds: [0, 103.3])
  - Max Wind: 6 outliers capped (bounds: [0, 103.3])
  - Barometric Pressure: 7 outliers capped (bounds: [870, 1083.8])
  - Solar Radiation: 13,425 outliers capped (bounds: [0, Inf])
- Duplicates: Removed (0 duplicates found)
- Data types: Validated and converted as needed
- Rows after cleaning: **182,516** (13,755 rows removed due to outliers)

The cleaning process maintained more than 90% of the full dataset size while improving data quality. The large number of missing values in Wet Bulb Temperature and rainfall-related variables (38.68%) suggests that some sensors may not be available at Foster stations, but mean/mode imputation and forward-fill ensured we could still use those features in analysis.

### Phase 4: Data Wrangling

Datetime parsing and temporal feature extraction were critical for time series analysis. The `Measurement Timestamp` column was parsed from the format "MM/DD/YYYY HH:MM:SS AM/PM" and set as the DataFrame index, enabling time-based operations.

**Temporal Features Extracted:**
- `hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0=Monday, 6=Sunday)
- `month`: Month of year (1-12)
- `year`: Year
- `day_name`: Day name (Monday-Sunday)
- `is_weekend`: Binary indicator (1 if Saturday/Sunday)

The dataset covers approximately 10.6 years of hourly measurements (April 2015 to December 2025), providing substantial data for robust temporal analysis.

### Phase 5: Feature Engineering

Feature engineering created derived variables and rolling window statistics to capture relationships and temporal dependencies. To avoid data leakage, no features were derived from the target variable `Air Temperature`. Similarly, only rolling windows of predictor variables were created. Creating rolling windows of the target variable (e.g., `air_temp_rolling_7h` when predicting Air Temperature) would cause data leakage. The rolling window features of predictor variables capture temporal dependencies essential for time series prediction.

**Derived Features:**
- `pressure_diff_1h`: Pressure change
- `wind_dir_delta`: Wind direction change using circular difference
- `wind_u`: Vectorized wind components (wind speed × cos(wind direction))
- `wind_v`: Vectorized wind components (wind speed × sin(wind direction))

**Rolling Window Features:**
- `wet_temp_rolling_7h`: 7-hour rolling mean of wet bulb temperature
- `wet_temp_rolling_24h`: 24-hour rolling mean of wet bulb temperature
- `rain_intensity_rolling_7h`: 7-hour rolling mean of rain intensity
- `rain_intensity_rolling_24h`: 24-hour rolling mean of rain intensity
- `humidity_rolling_7h`: 7-hour rolling mean of humidity
- `humidity_rolling_24h`: 24-hour rolling mean of humidity
- `pressure_rolling_7h`: 7-hour rolling mean of barometric pressure
- `pressure_rolling_24h`: 24-hour rolling mean of barometric pressure

**Categorical Features:**
- `wind_category`: Wind direction bins (North, East, South, West)
- `pressure_trend`: Pressure trend bins (rising, steady, falling)

### Phase 6: Pattern Analysis

Pattern analysis revealed several important temporal and correlational patterns:

**Temporal Trends:**
- Clear seasonal patterns: Air temperatures peak in summer months and reach minima in winter
- Monthly air temperature range: -5.0°C to 25.3°C
- Strong seasonal variation typical of Chicago's climate

**Daily Patterns:**
- Strong diurnal cycle in air temperature (warmer during day, cooler at night)
- Peak air temperature typically occurs around hour 15-16 (3-4 PM)
- Minimum air temperature typically occurs around hour 4-5 (4-5 AM)
- This pattern reflects solar heating and cooling cycles

**Correlations:**
- Air Temperature vs Wet Bulb Temperature: 0.98 (strong positive correlation as the two measurements are associated)
- Air Temperature vs Total Rain: 0.45 (moderate positive correlation - rainy days tend to be hotter)
- Air Temperature vs Barometric Pressure: -0.25 (moderate negative correlation - higher pressure tend to be cooler)
- Air Temperature vs Wind Speed: -0.18 (moderate negative correlation - windier days tend to be cooler)
- Air Temperature vs Humidity: 0.01 (very weak positive correlation)

![Figure 2: Pattern Analysis](output/q5_patterns.png)
*Figure 2: Advanced pattern analysis showing monthly temperature trends, seasonal patterns by month, daily patterns by hour, and correlation heatmap of key variables.*

### Phase 7: Modeling Preparation

Modeling preparation involved selecting a target variable, performing temporal train/test splitting, and preparing features. Air temperature was chosen as the target variable, as it's a key indicator of beach conditions and shows predictable patterns.

**Temporal Train/Test Split:**
- Split method: Temporal (80/20 split by time)
- Training set: **146,012 samples** (earlier data: April 2015 to ~May 2023)
- Test set: **36,504 samples** (later data: ~May 2023 to December 2025)
- Rationale: Time series data requires temporal splitting to avoid data leakage and ensure realistic evaluation

**Feature Preparation:**
- Features selected (excluding target, non-numeric columns, and features derived from target)
- Excluded features with >0.95 correlation to target (e.g., Wet Bulb Temperature with 0.978 correlation)
- Categorical variables (Station Name) one-hot encoded
- All features standardized and missing values handled
- Infinite values replaced with NaN then filled with median
- No data leakage: future data excluded from training set, and features derived from target excluded
- Total dataset: **182,516 rows** before split

### Phase 8: Modeling

Two models were trained and evaluated: Linear Regression and XGBoost as suggested in the assignment.

**Model Performance:**

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | 0.2573 | 8.64°C | 7.12°C |
| XGBoost | 0.8102 | 4.37°C | 3.36°C |

**Key Findings:**
- Linear Regression achieved moderate performance (R² = 0.2573), indicating that linear relationships alone are insufficient for accurate temperature prediction
- XGBoost achieved strong performance (R² = 0.8102), demonstrating the importance of non-linear modeling and gradient boosting methods
- XGBoost significantly outperforms Linear Regression, with RMSE of 4.37°C compared to 8.64°C

**Feature Importance (XGBoost):**
Top 5 features by importance:
1. `month` (62.25% importance) - the most important, capturing seasonal patterns
2. `Barometric Pressure` (6.95% importance)
3. `Total Rain` (5.88% importance)
4. `wind_u` (4.19% importance)
5. `Humidity` (3.72% importance)

The month feature dominates feature importance, accounting for 62.25% of total importance. This makes intuitive sense - seasonal patterns are the strongest predictor of air temperature. Temporal features (month) and weather variables (rain, pressure, humidity, wind) are more important than rolling windows of predictor variables. The top 5 features account for 82.99% of total importance.

![Figure 3: Model Performance](output/q8_final_visualizations.png)
*Figure 3: Final visualizations showing model performance comparison, and predictions vs actual values, feature importance for the best-performing XGBoost model.*

### Phase 9: Results

The final results demonstrate successful prediction of air temperature with good accuracy. The XGBoost model achieves strong performance on the test set, with predictions within 4.37°C on average.

**Summary of Key Findings:**
1. **Model Performance:** XGBoost achieves R² = 0.8102, indicating that 81.02% of variance in air temperature can be explained by the features
2. **Feature Importance:** The month feature is overwhelmingly the most important predictor (62.25% importance), highlighting the critical role of seasonal patterns
3. **Temporal Patterns:** Strong seasonal and daily patterns are critical for accurate prediction
4. **Data Quality:** Cleaning process maintained more than 90% of the dataset while improving reliability
5. **Data Leakage Avoidance:** By not deriving features from the target variable and excluding features highly correlated (> 0.95) with target variable, we achieved realistic and generalizable model performance

The residuals plot shows relatively uniform distribution around zero, suggesting the model performs reasonably well across the full temperature range. The predictions vs actual scatter plot shows points distributed around the perfect prediction line with some scatter, indicating good but not perfect accuracy - which is realistic for weather prediction.

## Visualizations

![Figure 1: Initial Data Exploration](output/q1_visualizations.png)
*Figure 1: Initial exploration showing distributions and time series of target variable 'Air Temperature'.*

![Figure 2: Pattern Analysis](output/q5_patterns.png)
*Figure 2: Advanced pattern analysis revealing temporal trends, seasonal patterns, daily cycles, and correlations.*

![Figure 3: Model Performance](output/q8_final_visualizations.png)
*Figure 3: Final results showing model comparison, prediction accuracy, feature importance.*

## Model Results

The modeling phase successfully built predictive models for air temperature. The performance metrics demonstrate that XGBoost performs well, while Linear Regression shows that linear relationships alone are insufficient for this task.

**Performance Interpretation:**
- **R² Score:** Measures proportion of variance explained. XGBoost's R² of 0.8102 means the model explains 81.02% of variance in air temperature - a strong but realistic result.
- **RMSE (Root Mean Squared Error):** Average prediction error in original units. XGBoost's RMSE of 4.37°C means predictions are typically within 4.37°C of actual values - reasonable for weather prediction.
- **MAE (Mean Absolute Error):** Average absolute prediction error. XGBoost's MAE of 3.36°C indicates good predictive accuracy.

**Model Selection:** XGBoost is selected as the best model due to:
1. Highest R² score (0.8102)
2. Lowest RMSE (4.37°C)
3. Lowest MAE (3.36°C)
4. Good generalization (train R² = 0.9145, test R² = 0.8102 - some overfitting but reasonable)

**Feature Importance Insights:**
The feature importance analysis reveals that:
- The month feature is overwhelmingly the most important predictor (62.25% importance), suggesting  seasonal patterns are the strongest predictor of air temperature
- Weather variables (Total Rain, Barometric Pressure, Humidity, Solar Radiation) and vectorized wind variables are important but secondary to temporal patterns
- Rolling windows of predictor variables (humidity, pressure, rain intensity) contribute but are less important than seasonal features
- Temporal features (month, year) are far more important than static weather variables
- Station location has minimal impact (encoded station features have very low importance)

**Note on Data Leakage Avoidance:** By not deriving  features from the target variable and excluding highly correlated features (Wet Bulb Temperature), we achieved realistic model performance. This demonstrates the importance of careful feature selection to avoid circular logic.

## Time Series Patterns

The analysis revealed several important temporal patterns:

**Long-term Trends:**
- Stable  long-term trend throughout the 2015–2025 period, showing no major upward or downward trends in average temperature despite some year to year variation
- Strong annual temperature cycle with air temperatures rising each summer and falling each winter in a consistent pattern across all years
- A small flattening appears around 2020–2021 likely due to missing sensor data rather than a real climatic effect

**Seasonal Patterns:**
- **Monthly:** Clear seasonal cycle with temperatures peaking in summer months (June-August) and reaching minima in winter months (December-February)
- Monthly air temperature range: -5.0°C to 25.3°C
- **Daily:** Strong diurnal cycle with temperatures peaking in afternoon (3-4 PM, hour 15-16) and reaching minima in early morning (4-5 AM, hour 4-5)
- Daily patterns are consistent across different day of the wekk, though amplitude varies

**Temporal Relationships:**
- Air temperature shows strong seasonal patterns with month being the most important predictor
- Total rain shows moderate positive correlation with air temperature (0.47)
- Solar radiation shows moderate positive correlation with temperature (0.29)
- Barometric pressure shows moderate negative correlation with temperature (-0.25)
- Rolling windows of predictor variables (rain intensity, humidity, pressure) capture temporal dependencies

**Anomalies:**
- Large gap in Wet Bulb Temperature and rainfall-related data (75,926 missing values, 38.68% of dataset) appears exclusively in Foster station data
- The length of gap matches the length of data from Foster station, which likely indicates certain sensors were not operational at the station throughout the measurement period
- Missing sensor value in 2020 and 2021 identified (gaps in time series)
- No major anomalies in temporal patterns beyond expected seasonal variation

These temporal patterns are critical for accurate prediction, as evidenced by the high importance of temporal features (especially rolling windows) in the model.

## Limitations & Next Steps

**Limitations:**

1. **Data Quality:**
   - Large number of missing values in Wet Bulb Temperature, rainfall-related features (38.68%) required imputation, which may introduce bias
   - Sensor dropouts create gaps in time series that could affect pattern detection
   - Outlier capping may have removed some valid extreme events
   - Only 3 weather stations - limited spatial coverage

2. **Model Limitations:**
   - Linear Regression's moderate performance (R² = 0.2575) indicates that linear relationships are insufficient for this task
   - XGBoost shows some overfitting (train R² = 0.9145 vs test R² = 0.8102), though this is reasonable
   - Model relies heavily on seasonal features (month = 62.25% importance), which limits predictive power for same-season predictions
   - Model trained on historical data may not generalize to future climate conditions
   - RMSE of 4.37°C, while reasonable, may not be sufficient for applications requiring high precision

3. **Feature Engineering:**
   - Some potentially useful features may not have been created (e.g., lag features, interaction terms)
   - Rolling window sizes (7h, 24h) were chosen somewhat arbitrarily
   - Avoided deriving features from target variable Air Temperature to avoid data leakage
   - External data (e.g., weather forecasts, lake conditions) not incorporated

4. **Scope:**
   - Analysis focused on air temperature prediction; other targets (e.g., wind speed, precipitation) not explored
   - Only one target variable analyzed; multi-target modeling could provide additional insights
   - Spatial relationships between stations not analyzed

**Next Steps:**

1. **Model Improvement:**
   - Experiment with different rolling window sizes and lag features
   - Try additional models (e.g., XGBoost, Gradient Boosting) to potentially improve performance
   - Incorporate external data sources (weather forecasts, lake level data)
   - Try ensemble methods combining multiple models
   - Validate model on truly out-of-sample data (future dates)
   - Address overfitting in XGBoost (train/test gap suggests some overfitting)

2. **Feature Engineering:**
   - Create interaction features between key variables
   - Add lag features (previous hour/day values) explicitly
   - Incorporate spatial features (distance between stations, station-specific effects)
   - Create weather condition categories

3. **Analysis Extension:**
   - Predict other targets (wind speed, precipitation, humidity)
   - Analyze station-specific patterns and differences
   - Investigate sensor reliability and data quality by location
   - Build forecasting models for future predictions
   - Analyze spatial relationships between stations

4. **Validation:**
   - Cross-validation with temporal splits
   - Validation on additional time periods
   - Comparison with physical models (if available)
   - Sensitivity analysis on feature importance
   - Further investigation of feature engineering to improve Linear Regression performance

5. **Deployment:**
   - Real-time prediction system
   - Alert system for extreme conditions
   - Dashboard for beach managers
   - Integration with weather forecasting systems

## Conclusion

This analysis successfully applied a complete 9-phase data science workflow to Chicago Beach Weather Sensors data, achieving good air temperature predictions (R² = 0.7684, RMSE = 4.87°C). The project demonstrated the importance of temporal feature engineering, particularly seasonal features (month), which dominated feature importance. Key insights include strong seasonal and daily patterns, the critical role of temporal features in prediction, and the superior performance of ensemble tree-based models over linear models. The analysis demonstrates proper data leakage avoidance by excluding features derived from the target variable, resulting in realistic and generalizable model performance. This provides a solid foundation for beach condition monitoring and prediction systems.

