import pandas as pd
# Sample sales data
sales_data = {
 'Store_ID': [101, 102, 103, 104, 105, 106, 107, 108],
 'Revenue': [12000, 15000, 9000, 22000, 18000, 11000, 31000, 14000],
 'Customers': [150, 200, 110, 280, 210, 130, 390, 180]
}
df = pd.DataFrame(sales_data)
print("--- Overall Statistical Summary ---")
print(df[['Revenue', 'Customers']].describe())
# Individual metrics
print(f"\nMedian Revenue: ${df['Revenue'].median():,.2f}")
print(f"Revenue Standard Deviation: ${df['Revenue'].std():,.2f}")
