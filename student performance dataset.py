import pandas as pd
# Create a sample student performance dataset
data = {
 'Student': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank'],
 'Department': ['CS', 'CS', 'ECE', 'ECE', 'CS', 'ME'],
 'Score': [88, 42, 95, 67, 78, 55],
 'Attendance_%': [92, 70, 98, 80, 85, 60]
}
df = pd.DataFrame(data)
# Filter students in CS with Score > 75
cs_high_scorers = df[(df['Department'] == 'CS') & (df['Score'] > 75)]
print("--- All Students ---")
print(df)
print("\n--- CS Students with Score > 75 ---")
print(cs_high_scorers[['Student', 'Score', 'Attendance_%']])
