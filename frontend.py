import chainlit as cl
from faisshnsw import search_faiss_hnsw_with_difficulty, TOPIC_DOMAINS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import random
import mysql.connector

quiz_state = {
    "current_topic": None,
    "current_difficulty": "medium",
    "previous_performance": "initial",
    "score": 0,
    "question_number": 0,
    "total_questions": 5,
    "user_answers": [],
    "question_pool": {},
    "llm": None,
    "question_chain": None
}

@cl.on_chat_start
async def start():
    # Initialize the quiz
    try:
        # Send welcome message
        await cl.Message(
            content="Welcome to the Programming Quiz! We'll test your knowledge of SQL and Java concepts."
        ).send()
        
        # Initialize LLM and chains
        quiz_state["llm"] = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        
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
        
        quiz_state["question_chain"] = LLMChain(llm=quiz_state["llm"], prompt=question_prompt)
        
        # Build question pool
        topics = ["Keys & Constraints", "Arrays", "Database"]
        for topic in topics:
            domain = TOPIC_DOMAINS.get(topic, "sql")
            topic_questions = {
                'easy': search_faiss_hnsw_with_difficulty(topic, 'easy', domain=domain),
                'medium': search_faiss_hnsw_with_difficulty(topic, 'medium', domain=domain),
                'hard': search_faiss_hnsw_with_difficulty(topic, 'hard', domain=domain)
            }
            quiz_state["question_pool"][topic] = topic_questions
        
        quiz_state["current_topic"] = random.choice(topics)
        
        # Start first question
        await ask_question()
        
    except Exception as e:
        await cl.Message(
            content=f"Error initializing quiz: {str(e)}"
        ).send()

async def ask_question():
    try:
        quiz_state["question_number"] += 1
        if quiz_state["question_number"] > quiz_state["total_questions"]:
            await end_quiz()
            return
        
        # Get question from pool
        available_questions = quiz_state["question_pool"][quiz_state["current_topic"]][quiz_state["current_difficulty"]]
        if not available_questions:
            available_questions = quiz_state["question_pool"][quiz_state["current_topic"]]['medium']
        
        question = random.choice(available_questions)
        available_questions.remove(question)
        
        # Get correct answer from database
        conn = get_db_connection()
        cursor = conn.cursor()
        domain = TOPIC_DOMAINS.get(quiz_state["current_topic"], "sql")
        table_name = "java_embeddings" if domain == "java" else "sql_embeddings"
        
        cursor.execute(f"""
            SELECT question, answer 
            FROM {table_name}
            WHERE question = %s
        """, (question,))
        
        question_data = cursor.fetchone()
        if not question_data:
            raise Exception("Could not fetch question details from database")
        
        _, correct_answer = question_data
        cursor.close()
        conn.close()
        
        # Generate question variation
        response = await quiz_state["question_chain"].ainvoke(
            input={
                "question": question,
                "difficulty": quiz_state["current_difficulty"],
                "previous_performance": quiz_state["previous_performance"],
                "domain": domain
            }
        )
        
        # Store current question for evaluation
        quiz_state["current_question"] = {
            "question": question,
            "domain": domain,
            "difficulty": quiz_state["current_difficulty"],
            "topic": quiz_state["current_topic"],
            "correct_answer": correct_answer
        }
        
        # Display question
        await cl.Message(
            content=f"Question {quiz_state['question_number']} of {quiz_state['total_questions']}\n"
                   f"Difficulty: {quiz_state['current_difficulty']}\n"
                   f"Topic: {quiz_state['current_topic']}\n"
                   f"Domain: {domain.upper()}\n\n"
                   f"{response['question']}"
        ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"Error generating question: {str(e)}"
        ).send()

@cl.on_message
async def handle_answer(message: cl.Message):
    try:
        # Evaluate answer
        comparison_prompt = ChatPromptTemplate.from_template(
            "Compare the following {domain} answers and determine if they are functionally equivalent:\n"
            "User's answer: {user_answer}\n"
            "Correct answer: {correct_answer}\n"
            "Respond with only 'correct' or 'incorrect'."
        )
        comparison_chain = LLMChain(llm=quiz_state["llm"], prompt=comparison_prompt)
        
        result = await comparison_chain.ainvoke(
            input={
                "user_answer": message.content,
                "correct_answer": quiz_state["current_question"]["correct_answer"],
                "domain": quiz_state["current_question"]["domain"]
            }
        )
        
        is_correct = "correct" in result['text'].lower()
        
        # Update quiz state
        if is_correct:
            quiz_state["score"] += 1
            quiz_state["previous_performance"] = "correct"
            if quiz_state["current_difficulty"] == "easy":
                quiz_state["current_difficulty"] = "medium"
            elif quiz_state["current_difficulty"] == "medium":
                quiz_state["current_difficulty"] = "hard"
            await cl.Message(content="Correct!").send()
        else:
            quiz_state["previous_performance"] = "incorrect"
            if quiz_state["current_difficulty"] == "hard":
                quiz_state["current_difficulty"] = "medium"
            elif quiz_state["current_difficulty"] == "medium":
                quiz_state["current_difficulty"] = "easy"
            await cl.Message(content="Incorrect.").send()
        
        # Store answer
        quiz_state["user_answers"].append((
            quiz_state["current_question"]["question"],
            message.content,
            quiz_state["current_question"]["correct_answer"],
            quiz_state["current_question"]["topic"],
            quiz_state["current_question"]["difficulty"],
            quiz_state["current_question"]["domain"]
        ))
        
        # Ask next question
        await ask_question()
        
    except Exception as e:
        await cl.Message(
            content=f"Error evaluating answer: {str(e)}"
        ).send()

async def end_quiz():
    try:
        # Display final score
        await cl.Message(
            content=f"Quiz completed! Your score: {quiz_state['score']}/{quiz_state['total_questions']}"
        ).send()
        
        # Generate and display feedback
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
            
            Keep the tone professional and constructive.
        """)
        
        # Calculate performance metrics
        performance_by_domain = {}
        performance_by_difficulty = {'easy': [], 'medium': [], 'hard': []}
        
        for answer_data in quiz_state["user_answers"]:
            _, _, _, _, diff, domain = answer_data
            if domain not in performance_by_domain:
                performance_by_domain[domain] = {'correct': 0, 'total': 0}
            performance_by_domain[domain]['total'] += 1
            is_correct = answer_data[1].lower() == answer_data[2].lower()
            if is_correct:
                performance_by_domain[domain]['correct'] += 1
            performance_by_difficulty[diff].append(is_correct)
        
        domain_perf_str = "\n".join([
            f"{domain.upper()}: {stats['correct']}/{stats['total']} correct"
            for domain, stats in performance_by_domain.items()
        ])
        
        difficulty_perf_str = "\n".join([
            f"{diff.capitalize()}: {sum(results)}/{len(results)} correct"
            for diff, results in performance_by_difficulty.items()
            if results
        ])
        
        feedback_chain = LLMChain(llm=quiz_state["llm"], prompt=feedback_prompt)
        feedback = await feedback_chain.ainvoke(
            input={
                "correct_answers": quiz_state["score"],
                "total_questions": quiz_state["total_questions"],
                "domain_performance": domain_perf_str,
                "difficulty_performance": difficulty_perf_str
            }
        )
        
        await cl.Message(
            content=f"=== Performance Analysis ===\n\n{feedback['text']}"
        ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"Error generating feedback: {str(e)}"
        ).send()

def get_db_connection():
    # This function should return a database connection object
    # For now, we'll use a placeholder
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="programming_quiz"
    )
