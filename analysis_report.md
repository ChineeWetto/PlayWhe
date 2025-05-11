# Play Whe Lottery Data Analysis Report

Analysis date: 2025-03-30 20:22:24

Data range: 2016-09-05 to 2025-03-31
Total number of draws analyzed: 7532

## Number Distribution Analysis

This analysis examines the distribution of winning numbers in the Play Whe lottery to determine if certain numbers appear more frequently than others. A uniform distribution would suggest a fair lottery system, while significant deviations might indicate patterns that could be exploited for predictions.

### Key Findings

- The chi-square test (p-value: 0.1741) suggests that the distribution of winning numbers is uniform, consistent with a fair lottery system.
- Number 14 appears most frequently (240 times), which is 14.71% above expected frequency.
- Number 10 appears least frequently (174 times), which is 16.83% below expected frequency.

### Statistical Results

- chi_square: 42.6925
- p_value: 0.1741
- most_frequent_number: 14 (240 occurrences)
- least_frequent_number: 10 (174 occurrences)
- max_positive_deviation: 14.71% (Number 14)
- max_negative_deviation: -16.83% (Number 10)

### Recommendations

- Frequency-based prediction may not be effective due to uniform distribution
- Focus on other patterns such as time-based or sequential patterns
- Consider a balanced approach that doesn't heavily weight frequency

### Visualizations

![number_distribution.png](number_distribution.png)

![number_deviation.png](number_deviation.png)

## Time-Based Pattern Analysis

This analysis examines whether winning numbers show patterns based on draw times (morning, midday, afternoon, evening) or days of the week. Such patterns could indicate time-dependent biases in the lottery system or drawing process.

### Key Findings

- No significant time-based patterns were found in the winning numbers.
- No significant day-based patterns were found in the winning numbers.

### Statistical Results

- time_p_values: {'10:30AM': '0.5656', '1:00PM': '0.7342', '4:00PM': '0.1428', '6:30PM': '0.2905', '7:00PM': '0.6429'}
- day_p_values: {'Monday': '0.1741'}

### Recommendations

- Time-based and day-based predictions may not be effective due to uniform distributions
- Focus on other patterns such as frequency-based or sequential patterns

### Visualizations

![time_distribution_1030AM.png](time_distribution_1030AM.png)

![time_distribution_100PM.png](time_distribution_100PM.png)

![time_distribution_400PM.png](time_distribution_400PM.png)

![time_distribution_630PM.png](time_distribution_630PM.png)

![time_distribution_700PM.png](time_distribution_700PM.png)

![day_distribution_Monday.png](day_distribution_Monday.png)

## Sequential Pattern Analysis

This analysis examines whether there are patterns in the sequence of winning numbers. It looks for correlations between consecutive draws, common transitions between numbers, and other sequential dependencies that could inform prediction strategies.

### Key Findings

- Consecutive repeats occur 2.74% of the time, which is close to the expected 2.78% for a random sequence.
- The most common transition is from 11 to 24, occurring 14 times.
- Significant autocorrelation was found at lags: 5, 17.
- This suggests that past winning numbers may have predictive value for future draws.

### Statistical Results

- consecutive_repeats: 206
- repeat_percentage: 2.74%
- expected_repeat_percentage: 2.78%
- top_transitions: ['11 → 24 (14 times)', '7 → 11 (14 times)', '25 → 18 (13 times)', '11 → 5 (13 times)', '1 → 18 (13 times)', '24 → 16 (13 times)', '8 → 9 (13 times)', '11 → 20 (12 times)', '14 → 24 (12 times)', '26 → 13 (12 times)']

### Recommendations

- Develop a sequential prediction model that considers previous winning numbers
- Pay special attention to transitions with high frequency, such as 11 → 24
- Consider implementing a Markov chain model based on the transition matrix

### Visualizations

![transition_matrix.png](transition_matrix.png)

![autocorrelation.png](autocorrelation.png)

## Hot/Cold Number Analysis

This analysis identifies 'hot' numbers (those drawn frequently in recent draws) and 'cold' numbers (those drawn rarely in recent draws). These patterns can be useful for prediction strategies that assume hot numbers will continue to appear or that cold numbers are 'due' to appear.

### Key Findings

- In the last 30 draws, the hottest numbers are: 24, 19, 11, 16, 5.
- In the last 30 draws, the coldest numbers are: 2, 6, 8, 9, 14.
- In the last 50 draws, the hottest numbers are: 19, 15, 11, 18, 24.
- In the last 50 draws, the coldest numbers are: 2, 6, 9, 14, 29.
- In the last 100 draws, the hottest numbers are: 11, 8, 18, 15, 19.
- In the last 100 draws, the coldest numbers are: 2, 14, 7, 9, 26.
- In the last 200 draws, the hottest numbers are: 11, 29, 28, 5, 18.
- In the last 200 draws, the coldest numbers are: 6, 14, 23, 30, 32.

### Statistical Results

- hot_numbers_100: {'11': 9, '8': 5, '18': 5, '15': 5, '19': 5, '22': 5, '28': 5, '1': 4, '16': 4, '13': 4}
- cold_numbers_100: {'2': 0, '14': 0, '7': 1, '9': 1, '26': 1, '27': 1, '6': 1, '34': 1, '36': 1, '3': 2}

### Recommendations

- Consider hot numbers for 'follow the trend' prediction strategies
- Consider cold numbers for 'due to hit' prediction strategies
- Combine hot/cold analysis with other patterns for more robust predictions
- Monitor how hot/cold patterns evolve over time to adjust prediction strategies

### Visualizations

![hot_cold_30.png](hot_cold_30.png)

![hot_cold_50.png](hot_cold_50.png)

![hot_cold_100.png](hot_cold_100.png)

![hot_cold_200.png](hot_cold_200.png)

## Advanced Statistical Analysis

This analysis applies advanced statistical methods to identify subtle patterns in the Play Whe lottery data. It examines the randomness of the sequence, tests for cyclical patterns, and evaluates the overall predictability of the lottery outcomes.

### Key Findings

- The runs test (p-value: 0.9739) suggests that the sequence of winning numbers is random.
- No significant cyclical patterns detected in the frequency domain.
- The entropy ratio (0.9992) suggests that the winning numbers have high randomness, making prediction challenging.

### Statistical Results

- runs_test_statistic: -0.0327
- runs_test_pvalue: 0.9739
- entropy: 5.1658
- max_entropy: 5.1699
- entropy_ratio: 0.9992

### Recommendations

- Advanced statistical models may have limited effectiveness due to high randomness
- Focus on simpler patterns such as frequency-based or time-based patterns
- Consider ensemble approaches that combine multiple prediction strategies

### Visualizations

![power_spectrum.png](power_spectrum.png)

## Correlation Analysis

This analysis examines correlations between various features in the Play Whe lottery data, such as winning numbers, draw times, days of the week, and derived features. Strong correlations could indicate predictive relationships that can be exploited in prediction models.

### Key Findings

- Found 1 pairs of features with correlation stronger than 0.1.
- Correlation between draw_number and time_num: 0.1433

### Statistical Results

- strongest_correlation: draw_number and time_num: 0.1433
- number_strong_correlations: 1

### Recommendations

- Utilize the strongest correlations in prediction models
- Consider feature engineering based on correlated variables
- Implement models that can capture the relationships between correlated features

### Visualizations

![correlation_matrix.png](correlation_matrix.png)

## Conclusion and Insights

Based on the comprehensive analysis of Play Whe lottery data, the following key insights emerge:

1. **High Randomness Observed**: The analysis suggests that the Play Whe lottery results are largely random. This indicates that prediction may be challenging, though some patterns may still be useful.

2. **Limited Time-Dependent Patterns**: The analysis shows minimal variation in number distributions across different draw times and days of the week. This suggests that time-based prediction strategies may have limited effectiveness.

3. **Weak Sequential Dependencies**: The analysis shows limited sequential patterns in the winning numbers. This suggests that predictions based solely on previous draws may have limited effectiveness.

4. **Hot and Cold Numbers**: The analysis identifies hot numbers (11, 8, 18, 15, 19) that appear frequently in recent draws and cold numbers (2, 14, 7, 9, 26) that appear rarely. These patterns can inform both 'follow the trend' and 'due to hit' prediction strategies.

5. **Recommended Prediction Approach**: Based on all analyses, a hybrid prediction model that combines multiple strategies would likely be most effective. This could include:
   - Frequency-based weighting of numbers
   - Time-specific adjustments based on draw time and day
   - Sequential pattern recognition
   - Hot/cold number consideration

6. **Limitations and Considerations**: Despite the patterns identified, it's important to note that:
   - Past patterns may not continue in the future
   - The lottery system may have changed over time
   - Some apparent patterns may be due to random chance
   - Prediction models should be continuously updated with new data

These insights provide a foundation for developing prediction models that can potentially improve the odds of winning in the Play Whe lottery, though no prediction system can guarantee success.

