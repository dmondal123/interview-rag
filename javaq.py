import pandas as pd

# Load the CSV file
file_path = "/Users/dmondal/Documents/simple-rag/all_questions_data.csv"  
df = pd.read_csv(file_path)

easy_questions = df[df["Difficulty Level"] == "Easy"].head(60)
medium_questions = df[df["Difficulty Level"] == "Medium"].head(60)
hard_questions = df[df["Difficulty Level"] == "Hard"].head(60)

# Concatenate the filtered data
filtered_df = pd.concat([easy_questions, medium_questions, hard_questions])

# Keep only required columns and rename them
filtered_df = filtered_df[["Question Text", "Difficulty Level"]].rename(
    columns={"Question Text": "question", "Difficulty Level": "difficulty"}
)

# Remove rows where 'question' is empty or NaN
filtered_df = filtered_df.dropna(subset=["question"])
filtered_df = filtered_df[filtered_df["question"].str.strip() != ""]

# Save to CSV without index
filtered_df.to_csv("filtered_questions.csv", index=False)

print("Filtered CSV saved as 'filtered_questions.csv'")

