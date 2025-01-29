import re
import csv

# Function to extract IDs and questions from the markdown file
def extract_questions_from_md(file_path):
    with open(file_path, 'r') as md_file:
        # Read the content of the markdown file
        md_content = md_file.read()
    
    # Regular expression to capture the question ID and the question text
    question_pattern = r'(\d+)\.\s(.*?)\n'
    
    # Find all matches for the question ID and the question text
    questions = re.findall(question_pattern, md_content)
    
    return questions

# Function to save the extracted questions to a CSV file
def save_questions_to_csv(questions, output_file):
    # Open the CSV file for writing
    with open(output_file, 'w', newline='') as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)
        
        # Write the header
        writer.writerow(['ID', 'Question'])
        
        # Write the extracted questions
        for question in questions:
            writer.writerow(question)

# Main execution flow
def main():
    # Define the path of the markdown file
    md_file_path = 'java.md'  # Change this to your .md file path
    
    # Define the output CSV file path
    output_csv_file = 'java_questions.csv'
    
    # Extract questions from the markdown file
    questions = extract_questions_from_md(md_file_path)
    
    # Save the questions to a CSV file
    save_questions_to_csv(questions, output_csv_file)
    
    print(f'Questions have been successfully extracted and saved to {output_csv_file}')

# Run the script
if __name__ == '__main__':
    main()

