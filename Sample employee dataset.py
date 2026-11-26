import pandas as pd
# Sample employee dataset
emp_data = {
 'Employee': ['Anna', 'Ben', 'Cara', 'Dan', 'Ella', 'Fred'],
 'Department': ['Sales', 'Tech', 'Sales', 'Tech', 'Sales', 'HR'],
 'Salary': [50000, 85000, 52000, 90000, 48000, 60000],
 'Experience_Yrs': [2, 5, 3, 7, 1, 4]
}
df = pd.DataFrame(emp_data)
# Group by Department and calculate mean Salary and max Experience
dept_summary = df.groupby('Department').agg(
 Avg_Salary=('Salary', 'mean'),
 Max_Experience=('Experience_Yrs', 'max'),
 Employee_Count=('Employee', 'count')
).reset_index()
print("--- Department-wise Aggregations ---")
print(dept_summary)
