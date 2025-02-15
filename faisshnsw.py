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

def search_faiss_hnsw_with_difficulty(topic, difficulty, domain, k=2):
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

def provide_quiz_feedback(user_answers, quiz_questions, final_score, total_questions):
    """
    Use OpenAI to generate a recruiter-friendly performance report
    
    Args:
        user_answers: List of (question, user_answer, correct_answer, topic, difficulty, domain) tuples
        quiz_questions: List of questions asked during the quiz
        final_score: The final score achieved by the candidate
        total_questions: Total number of questions attempted
        
    Returns:
        dict: Contains feedback text and performance statistics
    """
    try:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # Prepare performance metrics
        performance_data = []
        for q_data in user_answers:
            _, user_ans, correct_ans, topic, diff, domain = q_data
            performance_data.append({
                'topic': topic,
                'difficulty': diff,
                'domain': domain,
                'is_correct': user_ans.lower() == correct_ans.lower()
            })
        
        # Calculate domain-wise performance
        performance_by_domain = {}
        performance_by_difficulty = {'easy': [], 'medium': [], 'hard': []}
        
        for p in performance_data:
            domain = p['domain']
            if domain not in performance_by_domain:
                performance_by_domain[domain] = {'correct': 0, 'total': 0}
            performance_by_domain[domain]['total'] += 1
            if p['is_correct']:
                performance_by_domain[domain]['correct'] += 1
            performance_by_difficulty[p['difficulty']].append(p['is_correct'])
        
        feedback_prompt = ChatPromptTemplate.from_template("""
            You are a technical recruiter reviewing a candidate's programming quiz performance. 
            Generate a professional feedback report based on the following data:

            Overall Score: {correct_answers}/{total_questions}

            Domain Performance:
            {domain_performance}

            Difficulty Level Performance:
            {difficulty_performance}

            Please provide a concise professional report that includes:
            1. Overall assessment of technical skills
            2. Strengths and areas for improvement
            3. Recommendations for skill development
            
            Focus on evaluating:
            - Problem-solving capabilities
            - Technical proficiency in different domains
            - Ability to handle increasing difficulty
            
            Keep the tone professional and constructive. Do not mention specific questions or answers.
        """)
        
        # Format performance data
        domain_perf_str = "\n".join([
            f"{domain.upper()}: {stats['correct']}/{stats['total']} correct"
            for domain, stats in performance_by_domain.items()
        ])
        
        difficulty_perf_str = "\n".join([
            f"{diff.capitalize()}: {sum(results)}/{len(results)} correct"
            for diff, results in performance_by_difficulty.items()
            if results  # Only include difficulties that were attempted
        ])
        
        # Get AI feedback
        feedback_chain = LLMChain(llm=llm, prompt=feedback_prompt)
        feedback = feedback_chain.invoke(
            input={
                "correct_answers": final_score,
                "total_questions": total_questions,
                "domain_performance": domain_perf_str,
                "difficulty_performance": difficulty_perf_str
            }
        )
        
        # Return a dictionary containing all relevant information
        return {
            'feedback_text': feedback['text'],
            'domain_performance': {
                domain: f"{stats['correct']}/{stats['total']} correct"
                for domain, stats in performance_by_domain.items()
            },
            'difficulty_performance': {
                diff: f"{sum(results)}/{len(results)} correct"
                for diff, results in performance_by_difficulty.items()
                if results
            },
            'overall_score': f"{final_score}/{total_questions}"
        }
        
    except Exception as e:
        # Return basic statistics in case of error
        return {
            'feedback_text': f"Error generating detailed feedback: {str(e)}",
            'domain_performance': {
                domain: f"{stats['correct']}/{stats['total']} correct"
                for domain, stats in performance_by_domain.items()
            },
            'overall_score': f"{final_score}/{total_questions}"
        }

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
        
        user_answers = []  # List to store user answers and question details
        
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
                # Extract just the question field from the LLM response dictionary
                if isinstance(llm_response, dict) and 'question' in llm_response:
                    print(llm_response['question'])
                else:
                    print(llm_response['text'])  # fallback to text if question not found
                
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
                
                # Extract just the text field and check if it contains "correct"
                result_text = result['text'].lower()
                if result_text == "correct":
                    print("Correct!")
                    score += 1
                    previous_performance = 'correct'
                    if current_difficulty == 'easy':
                        current_difficulty = 'medium'
                    elif current_difficulty == 'medium':
                        current_difficulty = 'hard'
                else:
                    print("Incorrect.")
                    #print(f"The correct answer is:\n{answer}")
                    previous_performance = 'incorrect'
                    if current_difficulty == 'hard':
                        current_difficulty = 'medium'
                    elif current_difficulty == 'medium':
                        current_difficulty = 'easy'

                # After getting user's answer and evaluating it, store the details
                user_answers.append((
                    orig_question,
                    user_answer,
                    answer,
                    sub_topic,
                    difficulty,
                    domain
                ))

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

        # After quiz completion, provide detailed feedback
        feedback_info = provide_quiz_feedback(user_answers, question_pool, score, total_questions)
        print("\n=== Candidate Performance Report ===")
        print(feedback_info['feedback_text'])

    except Exception as e:
        print(f"Fatal error in create_sql_quiz: {str(e)}")