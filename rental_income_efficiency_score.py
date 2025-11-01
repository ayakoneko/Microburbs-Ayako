# Rental Income Efficiency Score (RIES) for Microburbs Test
# Calculates a Rental Income Efficiency Score for each suburb based on rental yield, occupancy rate, and cash flow margin.
# Helps identify properties that deliver stable and efficient rental returns.

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

## 1. Data load with pandas
df = pd.read_csv('rental_efficiency_input.csv')
print("Data loaded successfully.")
print(df.head())


## 2. Calculate indexs 

# Net Rental Yield = (Annual Rent - Annual Expenses) / Property Value
df['net_rental_yield'] = (df['annual_rent'] - df['annual_expenses']) / df['price'] * 100

# Cash Flow Margin = (Annual Rent - Annual Expenses) / Annual Rent
df['cash_flow_margin'] = (df['annual_rent'] - df['annual_expenses']) / df['annual_rent'] * 100

# Occupancy Rate = (1 - Vacancy Rate)
df['occupancy_rate'] = (1 - df['vacancy_rate']) * 100

print("\n Metrics calculated:")
print(df[['suburb', 'net_rental_yield', 'cash_flow_margin', 'occupancy_rate']].head())


# 3. Normalize Metrics (0-100 scale)
scaler = MinMaxScaler(feature_range=(0, 100))
df[['yield_norm', 'margin_norm', 'occupancy_norm']] = scaler.fit_transform(
    df[['net_rental_yield', 'cash_flow_margin', 'occupancy_rate']]
)

# 4. Compute RIES and Top 10 result 
# Weighted formula: 50% yield, 30% occupancy, 20% cash flow margin
df['RIES'] = (
    0.5 * df['yield_norm'] +
    0.3 * df['occupancy_norm'] +
    0.2 * df['margin_norm']
)

# Rank results
df = df.sort_values('RIES', ascending=False)
print("\n Top Suburbs by Rental Income Efficiency Score:")
print(df[['suburb', 'RIES']].head(10))


# 5. Visualization
plt.figure(figsize=(10,6))
plt.barh(df['suburb'], df['RIES'], color='skyblue')
plt.xlabel('Rental Income Efficiency Score (RIES)')
plt.ylabel('Suburb')
plt.title('Rental Income Efficiency by Suburb')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 6. Export Results in another csv
output_path = "rental_income_efficiency_results.csv"
df.to_csv(output_path, index=False)
print(f"\n Results saved to {output_path}")

# 7. Summary Notes ---
print("""
Interpretation:
- High RIES → efficient rental property with strong income and low vacancy.
- Low RIES → less efficient, potentially high costs or inconsistent occupancy.

Future Improvements:
- Integrate live rental and demographic data from Microburbs.
- Extend to time-series model for predictive rental efficiency trends.
""")