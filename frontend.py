import streamlit as st
import faisshnsw
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import random

def initialize_session_state():
    """Initialize session state variables"""
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'quiz_complete' not in st.session_state:
        st.session_state.quiz_complete = False
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = []
    if 'current_topic' not in st.session_state:
        st.session_state.current_topic = random.choice(list(faisshnsw.TOPIC_DOMAINS.keys()))
    if 'current_difficulty' not in st.session_state:
        st.session_state.current_difficulty = 'medium'
    if 'previous_performance' not in st.session_state:
        st.session_state.previous_performance = 'initial'
    if 'question_pool' not in st.session_state:
        st.session_state.question_pool = {}

def reset_quiz():
    """Reset all quiz-related session state variables"""
    st.session_state.current_question = 0
    st.session_state.score = 0
    st.session_state.quiz_complete = False
    st.session_state.user_answers = []
    st.session_state.current_topic = random.choice(list(faisshnsw.TOPIC_DOMAINS.keys()))
    st.session_state.current_difficulty = 'medium'
    st.session_state.previous_performance = 'initial'
    st.session_state.question_pool = {}

def main():
    st.set_page_config(page_title="Programming Quiz", page_icon="🎯")
    initialize_session_state()

    st.title("Programming Quiz Application")
    st.write("Test your SQL and Java knowledge!")

    if not st.session_state.quiz_complete:
        if st.session_state.current_question == 0:
            if st.button("Start Quiz"):
                # Initialize question pool
                with st.spinner("Preparing questions..."):
                    topics = ["Keys & Constraints", "Arrays", "Database"]
                    for topic in topics:
                        domain = faisshnsw.TOPIC_DOMAINS.get(topic, "sql")
                        st.session_state.question_pool[topic] = {
                            'easy': faisshnsw.search_faiss_hnsw_with_difficulty(topic, 'easy', domain=domain),
                            'medium': faisshnsw.search_faiss_hnsw_with_difficulty(topic, 'medium', domain=domain),
                            'hard': faisshnsw.search_faiss_hnsw_with_difficulty(topic, 'hard', domain=domain)
                        }
                st.session_state.current_question += 1
                st.experimental_rerun()

        elif st.session_state.current_question <= 5:
            st.write(f"Question {st.session_state.current_question}/5")
            st.write(f"Current Difficulty: {st.session_state.current_difficulty}")
            
            try:
                # Get current question
                available_questions = st.session_state.question_pool[st.session_state.current_topic][st.session_state.current_difficulty]
                if not available_questions:
                    st.warning("Falling back to medium difficulty questions...")
                    available_questions = st.session_state.question_pool[st.session_state.current_topic]['medium']
                
                question = random.choice(available_questions)
                available_questions.remove(question)

                # Get question details from database
                conn = faisshnsw.get_db_connection()
                cursor = conn.cursor()
                domain = faisshnsw.TOPIC_DOMAINS.get(st.session_state.current_topic, "sql")
                table_name = "java_embeddings" if domain == "java" else "sql_embeddings"
                
                cursor.execute(f"""
                    SELECT question, answer, difficulty, sub_topic 
                    FROM {table_name}
                    WHERE question = %s
                """, (question,))
                
                question_data = cursor.fetchone()
                if question_data:
                    orig_question, answer, difficulty, sub_topic = question_data
                    
                    # Display question info
                    st.write(f"Topic: {sub_topic} ({domain.upper()})")
                    st.write(orig_question)
                    
                    # Get user answer
                    user_answer = st.text_area("Your answer:", key=f"answer_{st.session_state.current_question}")
                    
                    if st.button("Submit Answer"):
                        with st.spinner("Evaluating answer..."):
                            # Evaluate answer using LLM
                            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
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
                            
                            is_correct = result['text'].lower() == "correct"
                            
                            if is_correct:
                                st.success("Correct!")
                                st.session_state.score += 1
                                st.session_state.previous_performance = 'correct'
                                if st.session_state.current_difficulty == 'easy':
                                    st.session_state.current_difficulty = 'medium'
                                elif st.session_state.current_difficulty == 'medium':
                                    st.session_state.current_difficulty = 'hard'
                            else:
                                st.error("Incorrect")
                                st.session_state.previous_performance = 'incorrect'
                                if st.session_state.current_difficulty == 'hard':
                                    st.session_state.current_difficulty = 'medium'
                                elif st.session_state.current_difficulty == 'medium':
                                    st.session_state.current_difficulty = 'easy'
                            
                            # Store answer details
                            st.session_state.user_answers.append((
                                orig_question,
                                user_answer,
                                answer,
                                sub_topic,
                                difficulty,
                                domain
                            ))
                            
                            st.session_state.current_question += 1
                            if st.session_state.current_question > 5:
                                st.session_state.quiz_complete = True
                            st.experimental_rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                if 'cursor' in locals():
                    cursor.close()
                if 'conn' in locals():
                    conn.close()

    if st.session_state.quiz_complete:
        st.write("### Quiz Complete!")
        st.write(f"Final Score: {st.session_state.score}/5")
        
        with st.spinner("Generating performance report..."):
            feedback_data = faisshnsw.provide_quiz_feedback(
                st.session_state.user_answers,
                st.session_state.question_pool,
                st.session_state.score,
                5
            )
            
            if feedback_data:
                st.write("### Performance Report")
                st.write(feedback_data['feedback_text'])
                
                st.write("### Performance Breakdown")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Domain Performance**")
                    for domain, score in feedback_data['domain_performance'].items():
                        st.write(f"{domain}: {score}")
                
                with col2:
                    st.write("**Difficulty Level Performance**")
                    if 'difficulty_performance' in feedback_data:
                        for diff, score in feedback_data['difficulty_performance'].items():
                            st.write(f"{diff.capitalize()}: {score}")
        
        if st.button("Start New Quiz"):
            reset_quiz()
            st.experimental_rerun()

if __name__ == "__main__":
    main()
