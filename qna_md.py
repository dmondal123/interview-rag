import csv
import re

# Define the input and output file paths
input_file_md = "sql.md"  # Replace with your actual markdown file name
output_file_csv = "sql_questions.csv"

# Read the markdown file and process it
def process_md_file(input_file):
    data = []
    with open(input_file, 'r') as file:
        lines = file.readlines()

        question, answer = "", ""
        id_counter = 1

        for line in lines:
            line = line.strip()
            if line.startswith("##") and question and answer:
                # Save the previous question and answer
                answer = answer.replace("[Table of Contents](#SQL)", "").strip()
                data.append([id_counter, question.strip(), answer.strip()])
                id_counter += 1
                question, answer = "", ""

            if line.startswith("##"):
                # Start of a new question
                question = line[2:].strip()
            elif question:
                # Accumulate the answer
                answer += line + "\n"

        # Add the last question-answer pair if available
        if question and answer:
            answer = answer.replace("[Table of Contents](#SQL)", "").strip()
            data.append([id_counter, question.strip(), answer.strip()])

    return data

# Append the new data to the CSV file
def append_to_csv(data, output_file):
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in data:
            csv_writer.writerow(row)

# Process the markdown file and append to CSV
data_md = process_md_file(input_file_md)
append_to_csv(data_md, output_file_csv)

print(f"Data from {input_file_md} successfully appended to {output_file_csv}")
