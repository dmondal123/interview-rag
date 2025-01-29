import csv

# Define the input and output file paths
input_file = "sql.txt"  # Replace with your actual text file name
output_file = "sql_questions.csv"


# Read the text file and process it
def process_file(input_file):
    data = []
    with open(input_file, 'r') as file:
        lines = file.readlines()

        question, answer = "", ""
        id_counter = 1

        for line in lines:
            line = line.strip()
            if line.startswith("--") and question and answer:
                # Save the previous question and answer
                data.append([id_counter, question.strip(), answer.strip()])
                id_counter += 1
                question, answer = "", ""
            
            if line.startswith("--"):
                # Start of a new question
                question = line[2:].strip()
                if "Example:" in line:
                    example_index = line.find("Example:")
                    question = line[2:example_index].strip() + " " + line[example_index:].strip()
                question = question.lstrip("1234567890> ")  # Remove leading numbers and >
            elif question:
                # Accumulate the answer
                answer += line + "\n"

        # Add the last question-answer pair if available
        if question and answer:
            data.append([id_counter, question.strip(), answer.strip()])

    return data

# Write the processed data to a CSV file
def write_to_csv(data, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["id", "question", "answer"])
        csv_writer.writerows(data)

# Process the input file and write to CSV
data = process_file(input_file)
write_to_csv(data, output_file)

print(f"Data successfully written to {output_file}")