import faiss

import numpy as np
import time
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import random

def get_embeddings(query):
    pass

def parse_embedding(embedding):
    pass

def get_db_connection():
    pass

# Define topic to domain mapping
TOPIC_DOMAINS = {
    # SQL topics
    "Keys & Constraints": "sql",
    "Basic SQL Operations": "sql",
    "Indexes & Transactions": "sql",
    "Views & Derived Tables": "sql",
    "Joins & Views": "sql",
    "Users & Security": "sql",
    "Databases & Storage": "sql",
    "Data Types & Tablespaces": "sql",
    # Java topics
    "Arrays": "java",
    "Database": "java",
    "Classes and Keywords": "java",
    "Objects and Serialization": "java",
    "Inheritance and Interfaces": "java",
    "Dynamic Programming": "java",
    "DFS and BFS": "java",
    "Methods and Polymorphism": "java" 
}

def search_faiss_hnsw_with_difficulty(topic, difficulty, k=2, domain="sql"):
    """Modified to handle both SQL and Java questions"""
    query = f"{topic} {difficulty}"
    query_embedding = get_embeddings(query)
    if query_embedding is None:
        return []
    
    dimension = len(query_embedding)
    index = faiss.IndexHNSWFlat(dimension, 32)
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Select table based on domain
    table_name = "java_embeddings" if domain == "java" else "sql_embeddings"
    
    # Get embeddings for questions matching the difficulty
    cursor.execute(f"""
        SELECT id, question_embedding 
        FROM {table_name}
        WHERE difficulty = %s
    """, (difficulty,))
    
    data = cursor.fetchall()
    embeddings = [parse_embedding(row[1]) for row in data]
    ids = [row[0] for row in data]
    
    if not embeddings:
        cursor.close()
        conn.close()
        return []
    
    index.add(np.array(embeddings, dtype=np.float32))
    D, I = index.search(np.array([query_embedding], dtype=np.float32), k)
    
    results = []
    for idx in I[0]:
        cursor.execute(f"SELECT question FROM {table_name} WHERE id=%s", (ids[idx],))
        result = cursor.fetchone()
        if result:
            results.append(result[0])
    
    cursor.close()
    conn.close()
    return results

def create_sql_quiz():
    try:
        print("Initializing Quiz...")
        
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        print("LLM initialized successfully")

        # Modified prompt templates to handle both SQL and Java
        question_prompt = ChatPromptTemplate.from_template("""
            Based on the following {domain} question and its difficulty level, generate a concise version 
            of the question that tests the same concept: 
            Question: {question}
            Difficulty: {difficulty}
            Previous Performance: {previous_performance}
            
            Generate a question that is:
            - More challenging if the previous answer was correct
            - Slightly easier if the previous answer was incorrect
            - Related to the same topic as the previous question
            - Appropriate for {domain} programming
        """)

        question_chain = LLMChain(llm=llm, prompt=question_prompt)
        print("Question chain created successfully")

        # Modified topics list to include both SQL and Java topics
        topics = ["Keys & Constraints", "Arrays", "Database"]
        
        print("Building question pool...")
        question_pool = {}
        for topic in topics:
            print(f"Fetching questions for topic: {topic}")
            domain = TOPIC_DOMAINS.get(topic, "sql")  # Default to SQL if topic not found
            topic_questions = {
                'easy': search_faiss_hnsw_with_difficulty(topic, 'easy', k=2, domain=domain),
                'medium': search_faiss_hnsw_with_difficulty(topic, 'medium', k=2, domain=domain),
                'hard': search_faiss_hnsw_with_difficulty(topic, 'hard', k=2, domain=domain)
            }
            if not any(topic_questions.values()):
                print(f"Warning: No questions found for topic: {topic}")
            question_pool[topic] = topic_questions

        if not any(any(questions.values()) for questions in question_pool.values()):
            raise Exception("No questions available in the pool")

        # Rest of the quiz logic remains similar, but we need to modify the database queries
        current_topic = random.choice(topics)
        current_difficulty = 'medium'
        previous_performance = 'initial'
        score = 0
        
        print("\nWelcome to the Programming Quiz!\n")
        
        for i in range(5):
            try:
                print(f"\nPreparing question {i+1}...")
                # Select question based on current difficulty
                available_questions = question_pool[current_topic][current_difficulty]
                if not available_questions:
                    print(f"No more questions available for {current_difficulty} difficulty, falling back to medium")
                    available_questions = question_pool[current_topic]['medium']
                    if not available_questions:
                        print(f"No more questions available for topic {current_topic}")
                        continue
                
                question = random.choice(available_questions)
                available_questions.remove(question)  # Prevent repetition
                
                # Get question details from appropriate database table
                conn = get_db_connection()
                if conn is None:
                    raise Exception("Could not establish database connection")
                    
                cursor = conn.cursor()
                domain = TOPIC_DOMAINS.get(current_topic, "sql")
                table_name = "java_embeddings" if domain == "java" else "sql_embeddings"
                
                cursor.execute(f"""
                    SELECT question, answer, difficulty, sub_topic 
                    FROM {table_name}
                    WHERE question = %s
                """, (question,))
                    
                question_data = cursor.fetchone()
                if not question_data:
                    raise Exception(f"Could not fetch question details from database")
                
                orig_question, answer, difficulty, sub_topic = question_data
                
                # Generate question variation based on previous performance
                print("Generating question variation...")
                llm_response = question_chain.invoke(
                    input={
                        "question": orig_question,
                        "difficulty": difficulty,
                        "previous_performance": previous_performance,
                        "domain": domain
                    }
                )
                
                print(f"\nQuestion {i+1} (Difficulty: {current_difficulty}, Topic: {sub_topic}, Domain: {domain.upper()})")
                print(llm_response)
                
                user_answer = input("\nYour answer: ")
                
                # Compare with correct answer using LLM
                print("Evaluating answer...")
                comparison_prompt = ChatPromptTemplate.from_template(
                    "Compare the following {domain} answers and determine if they are functionally equivalent:\nUser's answer: {user_answer}\nCorrect answer: {correct_answer}\nRespond with only 'correct' or 'incorrect'."
                )
                comparison_chain = LLMChain(llm=llm, prompt=comparison_prompt)
                
                result = comparison_chain.invoke(
                    input={
                        "user_answer": user_answer,
                        "correct_answer": answer,
                        "domain": domain
                    }
                )
                
                if "correct" in str(result).lower():
                    print("Correct!")
                    score += 1
                    previous_performance = 'correct'
                    if current_difficulty == 'easy':
                        current_difficulty = 'medium'
                    elif current_difficulty == 'medium':
                        current_difficulty = 'hard'
                else:
                    print("Incorrect.")
                    print(f"The correct answer is:\n{answer}")
                    previous_performance = 'incorrect'
                    if current_difficulty == 'hard':
                        current_difficulty = 'medium'
                    elif current_difficulty == 'medium':
                        current_difficulty = 'easy'

            except Exception as e:
                print(f"Error during question {i+1}: {str(e)}")
                continue
            finally:
                try:
                    if cursor:
                        cursor.close()
                    if conn:
                        conn.close()
                except Exception as e:
                    print(f"Error closing database connections: {str(e)}")

        # Adjust final score message based on actual questions asked
        total_questions = i + 1  # i is 0-based, so add 1
        print(f"\nQuiz completed! Your score: {score}/{total_questions}")

    except Exception as e:
        print(f"Fatal error in create_sql_quiz: {str(e)}")