import faiss
import numpy as np
import time
def get_embeddings(query):
    pass

def parse_embedding(embedding):
    pass

def get_db_connection():
    pass

def search_faiss_hnsw(query, k = 5):
    query_embedding = get_embeddings (query)
    if query_embedding is None:
        return
    dimension = len(query_embedding)
    index = faiss.IndexHNSWFlat(dimension, 32)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, question_embedding FROM embeddings")
    data = cursor.fetchall()
    embeddings = [parse_embedding(row[1]) for row in data]
    ids = [row[0] for row in data]
    start_time = time.time()
    index.add(np.array(embeddings, dtype=np.float32))
    D, I = index.search(np.array([query_embedding], dtype=np.float32), k)
    results = []
    for idx in 1[0]:
        cursor.execute("SELECT question FROM embeddings WHERE id=%s", (ids [idx],))
        result = cursor.fetchone()
        if result:
            results.append(result[0])
            print(f" [HNSW] Question: {result[0]}")
    end_time = time.time()
    print("Total time [HNSW]: ", end_time - start_time)
    cursor.close()
    conn.close()
    return results

def create_sql_quiz():
    from langchain.llms import OpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    import random

    # Initialize OpenAI LLM
    llm = OpenAI(temperature=0.7)

    # Create prompt template for question selection
    question_prompt = PromptTemplate(
        input_variables=["context"],
        template="Based on the following SQL question and its difficulty level, generate a concise version of the question that tests the same concept: {context}"
    )

    # Create LLMChain
    question_chain = LLMChain(llm=llm, prompt=question_prompt)

    # Get initial pool of questions using FAISS search
    topics = ["Joins & Views", "Keys & Constraints", "Basic SQL Operations", 
              "Indexes & Transactions", "Views & Derived Tables"]
    
    question_pool = []
    for topic in topics:
        results = search_faiss_hnsw(topic, k=3)
        if results:
            question_pool.extend(results)

    # Randomly select 5 questions from the pool
    selected_questions = random.sample(question_pool, min(5, len(question_pool)))

    # Get database connection for fetching full question details
    conn = get_db_connection()
    cursor = conn.cursor()

    score = 0
    print("\nWelcome to the SQL Quiz!\n")

    for i, question in enumerate(selected_questions, 1):
        # Get question details from database
        cursor.execute("""
            SELECT question, answer, difficulty, sub_topic 
            FROM embeddings 
            WHERE question = %s
        """, (question,))
        
        question_data = cursor.fetchone()
        if question_data:
            orig_question, answer, difficulty, sub_topic = question_data
            
            # Generate a variation of the question using LLM
            llm_response = question_chain.run(context=f"Question: {orig_question}\nDifficulty: {difficulty}")
            
            print(f"\nQuestion {i} (Difficulty: {difficulty}, Topic: {sub_topic})")
            print(llm_response)
            
            user_answer = input("\nYour answer: ")
            
            # Compare with correct answer using LLM
            comparison_prompt = PromptTemplate(
                input_variables=["user_answer", "correct_answer"],
                template="Compare the following SQL answers and determine if they are functionally equivalent:\nUser's answer: {user_answer}\nCorrect answer: {correct_answer}\nRespond with only 'correct' or 'incorrect'."
            )
            comparison_chain = LLMChain(llm=llm, prompt=comparison_prompt)
            
            result = comparison_chain.run(user_answer=user_answer, correct_answer=answer)
            
            if "correct" in result.lower():
                print("Correct!")
                score += 1
            else:
                print("Incorrect.")
                print(f"The correct answer is:\n{answer}")

    print(f"\nQuiz completed! Your score: {score}/5")
    
    cursor.close()
    conn.close()
