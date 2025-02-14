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

def search_faiss_hnsw_with_difficulty(topic, difficulty, k=2):
    query = f"{topic} {difficulty}"
    query_embedding = get_embeddings(query)
    if query_embedding is None:
        return []
    
    dimension = len(query_embedding)
    index = faiss.IndexHNSWFlat(dimension, 32)
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get embeddings for questions matching the difficulty
    cursor.execute("""
        SELECT id, question_embedding 
        FROM embeddings 
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
        cursor.execute("SELECT question FROM embeddings WHERE id=%s", (ids[idx],))
        result = cursor.fetchone()
        if result:
            results.append(result[0])
    
    cursor.close()
    conn.close()
    return results

def create_sql_quiz():
    try:
        print("Initializing SQL Quiz...")
        
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        print("LLM initialized successfully")

        # Modified prompt templates
        question_prompt = ChatPromptTemplate.from_template("""
            Based on the following SQL question and its difficulty level, generate a concise version 
            of the question that tests the same concept: 
            Question: {question}
            Difficulty: {difficulty}
            Previous Performance: {previous_performance}
            
            Generate a question that is:
            - More challenging if the previous answer was correct
            - Slightly easier if the previous answer was incorrect
            - Related to the same topic as the previous question
        """)

        question_chain = LLMChain(llm=llm, prompt=question_prompt)
        print("Question chain created successfully")

        # Get initial pool of questions using FAISS search
        topics = ["Joins & Views", "Keys & Constraints", "Basic SQL Operations", 
                 "Indexes & Transactions", "Views & Derived Tables"]
        
        print("Building question pool...")
        question_pool = {}
        for topic in topics:
            print(f"Fetching questions for topic: {topic}")
            topic_questions = {
                'easy': search_faiss_hnsw_with_difficulty(topic, 'easy', k=2),
                'medium': search_faiss_hnsw_with_difficulty(topic, 'medium', k=2),
                'hard': search_faiss_hnsw_with_difficulty(topic, 'hard', k=2)
            }
            if not any(topic_questions.values()):
                print(f"Warning: No questions found for topic: {topic}")
            question_pool[topic] = topic_questions

        if not any(any(questions.values()) for questions in question_pool.values()):
            raise Exception("No questions available in the pool")

        # Start with random topic and medium difficulty
        current_topic = random.choice(topics)
        current_difficulty = 'medium'
        previous_performance = 'initial'
        score = 0
        
        print("\nWelcome to the SQL Quiz!\n")
        
        for i in range(5):
            try:
                print(f"\nPreparing question {i+1}...")
                # Select question based on current difficulty
                available_questions = question_pool[current_topic][current_difficulty]
                if not available_questions:
                    print(f"No questions available for {current_difficulty} difficulty, falling back to medium")
                    available_questions = question_pool[current_topic]['medium']
                    if not available_questions:
                        raise Exception(f"No questions available for topic {current_topic}")
                
                question = random.choice(available_questions)
                available_questions.remove(question)  # Prevent repetition
                
                # Get question details from database
                conn = get_db_connection()
                if conn is None:
                    raise Exception("Could not establish database connection")
                    
                cursor = conn.cursor()
                cursor.execute("""
                        SELECT question, answer, difficulty, sub_topic 
                        FROM embeddings 
                        WHERE question = %s
                    """, (question,))
                    
                question_data = cursor.fetchone()
                if not question_data:
                    raise Exception(f"Could not fetch question details from database")
                
                orig_question, answer, difficulty, sub_topic = question_data
                
                # Generate question variation based on previous performance
                print("Generating question variation...")
                llm_response = question_chain.run(
                    question=orig_question,
                    difficulty=difficulty,
                    previous_performance=previous_performance
                )
                
                print(f"\nQuestion {i+1} (Difficulty: {current_difficulty}, Topic: {sub_topic})")
                print(llm_response)
                
                user_answer = input("\nYour answer: ")
                
                # Compare with correct answer using LLM
                print("Evaluating answer...")
                comparison_prompt = ChatPromptTemplate.from_template(
                    "Compare the following SQL answers and determine if they are functionally equivalent:\nUser's answer: {user_answer}\nCorrect answer: {correct_answer}\nRespond with only 'correct' or 'incorrect'."
                )
                comparison_chain = LLMChain(llm=llm, prompt=comparison_prompt)
                
                result = comparison_chain.run(user_answer=user_answer, correct_answer=answer)
                
                if "correct" in result.lower():
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
                    # Decrease difficulty for next question
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

        print(f"\nQuiz completed! Your score: {score}/5")

    except Exception as e:
        print(f"Fatal error in create_sql_quiz: {str(e)}")