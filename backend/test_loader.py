from ml.data_loader import load_dataset



file_path = "../dataset/student_placement.xlsx"

df = load_dataset(file_path)

print("Dataset loaded successfully!")
print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())