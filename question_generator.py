# import pandas as pd
# import os
# import streamlit as st
# from langchain.memory import ChatMessageHistory
# from cohere import Client

# # Initialize Cohere client and LangChain memory
# cohere_api_key = "J4vDW2CC7v1geIbrI3AauhRZMtXjiCK6wy1Bdexk"
# cohere_client = Client(api_key=cohere_api_key)
# memory = ChatMessageHistory()

# # File to store previously generated questions
# db_file = "generated_questions.csv"

# # Load existing questions if the file exists
# if os.path.exists(db_file):
#     db = pd.read_csv(db_file)
# else:
#     db = pd.DataFrame(columns=["question", "option_a", "option_b", "option_c", "option_d", "answer"])

# # Define a function to parse generated MCQs
# def parse_mcq(mcq_text):
#     lines = mcq_text.strip().split("\n")
#     question = lines[0][3:].strip()  # Remove 'Q: ' prefix
#     options = [line.strip() for line in lines[1:5]]  # Extract the options (a, b, c, d)
    
#     # Attempt to extract the answer from the last line, if it exists
#     answer = None
#     for line in lines:
#         if line.lower().startswith('answer:'):
#             answer = line.split(":")[1].strip()  # Extract answer after 'Answer:'
#             break  # Exit loop once answer is found
    
#     return question, options, answer

# # Define a function to generate unique MCQs with time limits based on difficulty
# def generate_mcq(topic, existing_questions, difficulty="hard"):
#     if difficulty == "easy":
#         complexity = "a single-step calculation or direct application of a basic concept"
#         time_limit = "1 minute"
#     elif difficulty == "medium":
#         complexity = "a multi-step problem involving intermediate concepts"
#         time_limit = "1.5 minutes"
#     else:  # "hard"
#         complexity = "a challenging multi-step problem requiring advanced reasoning"
#         time_limit = "2.5 minutes"

#     prompt = f"""
#     Create a {difficulty}-level multiple-choice question strictly on the topic "{topic}" that adheres to the following:
#     - The question must only relate to "{topic}" without introducing other topics like geometry or probability.
#     - The complexity must match the description: {complexity}.
#     - The problem must be solvable in under {time_limit} by someone familiar with "{topic}".
#     - Include four unique answer choices (a, b, c, d) and specify the correct answer.
#     - Avoid ambiguity, ensure clarity, and use distinct numerical or logical examples.

#     Example format:
#     Q: <question>? 
#     a) Option1 
#     b) Option2 
#     c) Option3 
#     d) Option4 
#     Answer: Correct option
#     """
    
#     response = cohere_client.generate(
#         model='command-xlarge-nightly',
#         prompt=prompt,
#         max_tokens=300,
#         temperature=0.7,
#     )

#     return response.generations[0].text.strip()


# # Generate multiple questions with a limit for a single topic
# def generate_multiple_mcqs(topic, num_questions, difficulty="hard"):
#     generated_questions = []

#     # Convert existing questions to a list
#     existing_questions = db["question"].tolist() if not db.empty else []
    
#     # Generate the specified number of questions for the selected topic
#     while len(generated_questions) < num_questions:
#         new_question = generate_mcq(topic, existing_questions, difficulty)
        
#         # Parse and ensure uniqueness
#         question, options, answer = parse_mcq(new_question)
#         if question not in existing_questions and answer:  # Ensure answer is present
#             generated_questions.append({
#                 "question": question,
#                 "option_a": options[0],
#                 "option_b": options[1],
#                 "option_c": options[2],
#                 "option_d": options[3],
#                 "answer": answer,
#             })
#             existing_questions.append(question)  # Add to in-memory check

#     return generated_questions

# # Streamlit interface
# st.title("MCQ Generator")
# st.write("Generate unique multiple-choice questions based on a selected topic.")

# # List of topics
# topics_list = [
#     "Numbers", "Percentage", "Profit and Loss", "Average", "Ratio and Proportion",
#     "Mixture and Alligation", "Time and Work", "Time Speed Distance", "Pipes and Cisterns",
#     "Algebra", "Trigonometry, Height, and Distance", "Geometry", "Probability",
#     "Permutation and Combination (PnC)", "Age"
# ]

# # Topic input: Allow selecting a single topic
# topic = st.selectbox("Select a topic for the MCQs:", topics_list)

# # Difficulty input
# difficulty = st.selectbox("Select difficulty level:", ["easy", "medium", "hard"])

# # Number of questions: Input without upper limit
# num_questions = st.number_input(
#     "Enter the number of questions:",
#     min_value=1,  # Minimum value set to 1
#     value=5,      # Default value
#     step=1        # Increment step
# )

# # Button to generate questions
# if st.button("Generate Questions"):
#     if not topic:
#         st.error("Please select a topic.")
#     else:
#         with st.spinner("Generating questions..."):
#             # Generate new questions for the selected topic
#             new_questions = generate_multiple_mcqs(topic, num_questions, difficulty)

#             # Display questions
#             for idx, question_data in enumerate(new_questions, 1):
#                 st.write(f"**Question {idx}:** {question_data['question']}")
#                 st.write(f"a) {question_data['option_a']}")
#                 st.write(f"b) {question_data['option_b']}")
#                 st.write(f"c) {question_data['option_c']}")
#                 st.write(f"d) {question_data['option_d']}")
#                 st.write(f"**Answer:** {question_data['answer']}")
#                 st.write("---")

#             # Append new questions to the database
#             new_db = pd.DataFrame(new_questions)
#             db = pd.concat([db, new_db], ignore_index=True)

#             # Save updated database to CSV
#             db.to_csv(db_file, index=False)

#             st.success(f"New questions saved to {db_file}")




# import pandas as pd
# import streamlit as st
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.llms import OpenAI
# import requests
# import json

# # Initialize memory with a more recent approach (adapted to LangChain's new API)
# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory

# # Initialize memory (updated)
# memory = ConversationBufferMemory()

# # Define a function to parse generated MCQs
# def parse_mcq(mcq_text):
#     mcq_text = mcq_text.replace("\n", " ").strip()  # Replace newlines with spaces and trim extra spaces
#     options_start = ["a)", "b)", "c)", "d)"]
#     lines = mcq_text.split(" ")
#     question = []
#     options = []
#     answer = None

#     for i, line in enumerate(lines):
#         if any(line.startswith(option) for option in options_start):
#             options.append(" ".join(lines[i:]).split(line, 1)[1].strip())
#         elif line.lower().startswith("answer:"):
#             answer = line.split(":", 1)[1].strip()
#         else:
#             question.append(line)

#     question = " ".join(question).replace("Q:", "").strip()
#     return question, options, answer

# # Function to generate MCQs using the Groq API (Llama 3-8b-8192 model)
# def generate_mcq_with_groq(topic, difficulty):
#     if difficulty == "easy":
#         complexity = "a single-step calculation or direct application of a basic concept"
#         time_limit = "1 minute"
#     elif difficulty == "medium":
#         complexity = "a multi-step problem involving intermediate concepts"
#         time_limit = "1.5 minutes"
#     else:
#         complexity = "a challenging multi-step problem requiring advanced reasoning"
#         time_limit = "2.5 minutes"

#     prompt = f"""
#     Create a {difficulty}-level multiple-choice question strictly on the topic "{topic}" that adheres to the following:
#     - The question must only relate to "{topic}" without introducing other topics like geometry or probability.
#     - The complexity must match the description: {complexity}.
#     - The problem must be solvable in under {time_limit} by someone familiar with "{topic}".
#     - Include four unique answer choices (a, b, c, d) and specify the correct answer.
#     - Avoid ambiguity, ensure clarity, and use distinct numerical or logical examples.

#     Example format:
#     Q: <question>? 
#     a) Option1 
#     b) Option2 
#     c) Option3 
#     d) Option4 
#     Answer: Correct option
#     """

#     # API call to Groq (replace the URL with Groq's actual endpoint)
#     groq_api_url = "https://api.groq.com/v1/models/meta-llama/Meta-Llama-3-8B-Instruct"
#     headers = {
#         "Authorization": "Bearer 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",  # Replace with your API key
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "model": "llama3-8b-8192",
#         "prompt": prompt,
#         "max_tokens": 300,
#         "temperature": 0.7
#     }

#     response = requests.post(groq_api_url, headers=headers, json=payload)

#     # Check for errors in the response
#     if response.status_code == 200:
#         result = response.json()
#         return result.get('generated_text', '').strip()
#     else:
#         return f"Error: {response.status_code}, {response.text}"

# # Generate multiple questions
# def generate_multiple_mcqs(topic, num_questions, difficulty="hard"):
#     generated_questions = []

#     # Retrieve existing questions from memory
#     existing_questions = [msg.content for msg in memory.chat_memory.messages if "question" in msg.content]
    
#     while len(generated_questions) < num_questions:
#         # Generate MCQ using Groq API (Llama 3-8b-8192 model)
#         new_question = generate_mcq_with_groq(topic, difficulty)
        
#         # Parse and ensure uniqueness
#         question, options, answer = parse_mcq(new_question)
#         if question not in existing_questions and answer:
#             question_data = {
#                 "question": question,
#                 "option_a": options[0],
#                 "option_b": options[1],
#                 "option_c": options[2],
#                 "option_d": options[3],
#                 "answer": answer,
#             }
#             generated_questions.append(question_data)
#             memory.chat_memory.add_user_message(f"{question}\nAnswer: {answer}")  # Save full Q&A

#     return generated_questions

# # Streamlit interface
# st.title("MCQ Generator using Llama 3-8b-8192 (Groq)")
# st.write("Generate unique multiple-choice questions based on a selected topic.")

# # List of topics
# topics_list = [
#     "Numbers", "Percentage", "Profit and Loss", "Average", "Ratio and Proportion",
#     "Mixture and Alligation", "Time and Work", "Time Speed Distance", "Pipes and Cisterns",
#     "Algebra", "Trigonometry, Height, and Distance", "Geometry", "Probability",
#     "Permutation and Combination (PnC)", "Age"
# ]

# # Topic input: Allow selecting a single topic
# topic = st.selectbox("Select a topic for the MCQs:", topics_list)

# # Difficulty input
# difficulty = st.selectbox("Select difficulty level:", ["easy", "medium", "hard"])

# # Number of questions: Input without upper limit
# num_questions = st.number_input(
#     "Enter the number of questions:",
#     min_value=1, 
#     value=5,  
#     step=1  
# )

# # Button to generate questions
# if st.button("Generate Questions"):
#     if not topic:
#         st.error("Please select a topic.")
#     else:
#         with st.spinner("Generating questions..."):
#             new_questions = generate_multiple_mcqs(topic, num_questions, difficulty)
#             for idx, question_data in enumerate(new_questions, 1):
#                 st.write(f"**Question {idx}:** {question_data['question']}")
#                 st.write(f"a) {question_data['option_a']}")
#                 st.write(f"b) {question_data['option_b']}")
#                 st.write(f"c) {question_data['option_c']}")
#                 st.write(f"d) {question_data['option_d']}")
#                 st.write(f"**Answer:** {question_data['answer']}")
#                 st.write("---")

#             st.success("New questions added to memory!")

# # Button to display saved memory
# if st.button("View Saved Questions"):
#     messages = memory.chat_memory.messages
#     if messages:
#         st.write("### Saved Questions:")
#         for idx, msg in enumerate(messages, 1):
#             st.write(f"**{idx}.** {msg.content}")
#     else:
#         st.write("No questions saved in memory yet.")


# import streamlit as st
# from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
# from langchain.schema import messages_from_dict, messages_to_dict
# from groq import Groq
# import time

# # Initialize the Groq client
# client = Groq(api_key="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")

# # Initialize in-memory chat history
# chat_history = ChatMessageHistory()

# # Function to parse generated MCQs
# def parse_mcq(mcq_text):
#     mcq_text = mcq_text.replace("\n", " ").strip()
#     options_start = ["a)", "b)", "c)", "d)"]
#     lines = mcq_text.split(" ")
#     question = []
#     options = []
#     answer = None

#     for i, line in enumerate(lines):
#         if any(line.startswith(option) for option in options_start):
#             options.append(" ".join(lines[i:]).split(line, 1)[1].strip())
#         elif "Answer:" in line:
#             answer = line.split(":", 1)[1].strip()
#         else:
#             question.append(line)

#     question = " ".join(question).replace("Q:", "").strip()
#     return question, options, answer

# # Function to generate MCQs using the Groq API
# def generate_mcqs_with_groq(topic, difficulty, num_questions):
#     if difficulty == "easy":
#         complexity = "a single-step calculation or direct application of a basic concept"
#         time_limit = "1 minute"
#     elif difficulty == "medium":
#         complexity = "a multi-step problem involving intermediate concepts"
#         time_limit = "1.5 minutes"
#     else:
#         complexity = "a challenging multi-step problem requiring advanced reasoning"
#         time_limit = "2.5 minutes"

#     prompt = f"""
#     Create {num_questions} unique {difficulty}-level multiple-choice questions strictly on the topic "{topic}" that adhere to the following:
#     - Each question must only relate to "{topic}" without introducing other topics like geometry or probability.
#     - The complexity must match the description: {complexity}.
#     - The problem must be solvable in under {time_limit} by someone familiar with "{topic}".
#     - Include four unique answer choices (a, b, c, d) and specify the correct answer for each question.
#     """
#     try:
#         st.write("Sending prompt to Groq API...")  # Debug log
#         chat_completion = client.chat.completions.create(
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": prompt},
#             ],
#             model="llama3-8b-8192",
#             temperature=0.7,
#             max_tokens=1000,  # Adjust based on expected response length
#         )
#         st.write("Received response from Groq API.")  # Debug log
#         return chat_completion.choices[0].message.content.strip()
#     except Exception as e:
#         return f"Error: {e}"

# # Generate multiple MCQs
# def generate_multiple_mcqs(topic, num_questions, difficulty="hard"):
#     generated_questions = []
#     existing_questions = [msg.content for msg in chat_history.messages[-50:] if "question" in msg.content]  # Limit to last 50 messages

#     st.write("Generating questions...")
#     while len(generated_questions) < num_questions:
#         new_questions_text = generate_mcqs_with_groq(topic, difficulty, num_questions - len(generated_questions))
#         new_questions = new_questions_text.split("\n\n")  # Assuming Groq returns multiple questions separated by newlines

#         for new_question in new_questions:
#             question, options, answer = parse_mcq(new_question)
#             if question and question not in existing_questions and options and answer:
#                 question_data = {
#                     "question": question,
#                     "option_a": options[0],
#                     "option_b": options[1],
#                     "option_c": options[2],
#                     "option_d": options[3],
#                     "answer": answer,
#                 }
#                 generated_questions.append(question_data)
#                 chat_history.add_user_message(f"{question}\nAnswer: {answer}")

#             if len(generated_questions) >= num_questions:
#                 break

#     return generated_questions

# # Streamlit Interface
# st.title("MCQ Generator using Groq")
# st.write("Generate unique multiple-choice questions based on a selected topic.")

# # List of topics
# topics_list = [
#     "Numbers", "Percentage", "Profit and Loss", "Average", "Ratio and Proportion",
#     "Mixture and Alligation", "Time and Work", "Time Speed Distance", "Pipes and Cisterns",
#     "Algebra", "Trigonometry, Height, and Distance", "Geometry", "Probability",
#     "Permutation and Combination (PnC)", "Age"
# ]

# # Topic input
# topic = st.selectbox("Select a topic for the MCQs:", topics_list)

# # Difficulty input
# difficulty = st.selectbox("Select difficulty level:", ["easy", "medium", "hard"])

# # Number of questions input
# num_questions = st.number_input(
#     "Enter the number of questions:",
#     min_value=1, value=5, step=1
# )

# # Button to generate questions
# if st.button("Generate Questions"):
#     if not topic:
#         st.error("Please select a topic.")
#     else:
#         with st.spinner("Generating questions..."):
#             start_time = time.time()
#             new_questions = generate_multiple_mcqs(topic, num_questions, difficulty)
#             end_time = time.time()

#             st.write(f"Time taken: {end_time - start_time:.2f} seconds")  # Log time taken

#             if new_questions:
#                 for idx, question_data in enumerate(new_questions, 1):
#                     st.write(f"**Question {idx}:** {question_data['question']}")
#                     st.write(f"a) {question_data['option_a']}")
#                     st.write(f"b) {question_data['option_b']}")
#                     st.write(f"c) {question_data['option_c']}")
#                     st.write(f"d) {question_data['option_d']}")
#                     st.write(f"**Answer:** {question_data['answer']}")
#                     st.write("---")
#                 st.success("Questions generated successfully!")
#             else:
#                 st.error("Failed to generate questions. Please try again.")

# # Button to view saved questions
# if st.button("View Saved Questions"):
#     messages = chat_history.messages
#     if messages:
#         st.write("### Saved Questions:")
#         for idx, msg in enumerate(messages, 1):
#             st.write(f"**{idx}.** {msg.content}")
#     else:
#         st.write("No questions saved in memory yet.")



import pandas as pd
import os
import streamlit as st
from groq import Groq

# Initialize Groq client
groq_api_key = "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"  # Replace with your actual API key
groq_client = Groq(api_key=groq_api_key)

# File to store previously generated questions
db_file = "generated_questions.csv"

# Load existing questions if the file exists
if os.path.exists(db_file):
    db = pd.read_csv(db_file)
else:
    db = pd.DataFrame(columns=["question", "option_a", "option_b", "option_c", "option_d", "answer"])

# Function to log API responses
def log_api_response(response_text):
    """Logs raw API responses for debugging purposes."""
    with open("api_responses.log", "a") as log_file:
        log_file.write(response_text + "\n\n")

# Function to parse the MCQ response
def parse_mcq(mcq_text):
    """Parses the MCQ text returned by the API and ensures all components are valid."""
    log_api_response(mcq_text)  # Log raw response for debugging

    lines = mcq_text.strip().split("\n")
    question = None
    options = []
    answer = None

    for line in lines:
        if line.lower().startswith("q:"):
            question = line[2:].strip()
        elif line.lower().startswith(("a)", "b)", "c)", "d)")):
            options.append(line[3:].strip())
        elif "answer:" in line.lower():
            answer = line.split(":", 1)[1].strip().lower()

    # Validation
    if not question or len(options) != 4 or answer not in ["a", "b", "c", "d"]:
        raise ValueError("The API response did not include a complete and valid MCQ.")

    return question, options, answer

# Function to generate fallback MCQ
def synthesize_options(topic):
    """Generate simple fallback options if the API fails."""
    return [f"{topic} Option 1", f"{topic} Option 2", f"{topic} Option 3", f"{topic} Option 4"], "a"

# Function to generate MCQ with retries and fallback
def generate_mcq_with_retry(topic, difficulty="hard", max_retries=3):
    """Generates an MCQ using the API with retries for incomplete or invalid responses."""
    for attempt in range(max_retries):
        try:
            response = generate_mcq(topic, difficulty)
            question, options, answer = parse_mcq(response)
            
            # Check if the question already exists in the database
            if question in db['question'].values:
                st.warning(f"Duplicate question found. Regenerating...")
                continue  # Skip this and retry
            
            return question, options, answer
        except ValueError as e:
            st.warning(f"Attempt {attempt + 1} failed: {e}")
            continue

    # Fallback if all retries fail
    st.error("API failed after multiple attempts. Generating fallback MCQ.")
    fallback_question = f"Sample Question on {topic}"
    fallback_options, fallback_answer = synthesize_options(topic)
    return fallback_question, fallback_options, fallback_answer

# Function to generate a single MCQ
def generate_mcq(topic, difficulty="hard"):
    """Generate a single MCQ using the Groq API."""
    complexity_map = {
        "easy": "a single-step calculation or direct application of a basic concept",
        "medium": "a multi-step problem involving intermediate concepts",
        "hard": "a challenging multi-step problem requiring advanced reasoning"
    }
    time_limit_map = {
        "easy": "1 minute",
        "medium": "1.5 minutes",
        "hard": "2.5 minutes"
    }

    prompt = f"""
    Create a {difficulty}-level multiple-choice question strictly on the topic "{topic}" that adheres to the following:
    - The question must only relate to "{topic}" without introducing other topics like geometry or probability.
    - The complexity must match the description: {complexity_map[difficulty]}.
    - The problem must be solvable in under {time_limit_map[difficulty]} by someone familiar with "{topic}".
    - Include exactly four unique answer choices (a, b, c, d) and specify the correct answer.
    - Ensure clarity, avoid ambiguity, and use distinct numerical or logical examples.
    
    Example format:
    Q: What is 2 + 2? 
    a) 3 
    b) 4 
    c) 5 
    d) 6 
    Answer: b
    """

    try:
        response = groq_client.chat.completions.create(
            messages=[ 
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Error in API call: {e}")

# Streamlit interface
st.title("MCQ Generator using Groq API")
st.write("Generate unique multiple-choice questions based on a selected topic.")

# List of topics
topics_list = [
    "Numbers", "Percentage", "Profit and Loss", "Average", "Ratio and Proportion",
    "Mixture and Alligation", "Time and Work", "Time Speed Distance", "Pipes and Cisterns",
    "Algebra", "Trigonometry, Height, and Distance", "Geometry", "Probability",
    "Permutation and Combination (PnC)", "Age"
]

# User inputs
topic = st.selectbox("Select a topic for the MCQs:", topics_list)
difficulty = st.selectbox("Select difficulty level:", ["easy", "medium", "hard"])
num_questions = st.number_input("Enter the number of questions:", min_value=1, value=5, step=1)

# Generate questions button
if st.button("Generate Questions"):
    if not topic:
        st.error("Please select a topic.")
    else:
        with st.spinner("Generating questions..."):
            generated_questions = []
            for _ in range(num_questions):
                question, options, answer = generate_mcq_with_retry(topic, difficulty)
                generated_questions.append({
                    "question": question,
                    "option_a": options[0],
                    "option_b": options[1],
                    "option_c": options[2],
                    "option_d": options[3],
                    "answer": answer,
                })

            # Save questions to the database
            new_questions_df = pd.DataFrame(generated_questions)
            db = pd.concat([db, new_questions_df], ignore_index=True)
            db.to_csv(db_file, index=False)

            # Display questions
            for idx, q in enumerate(generated_questions, start=1):
                st.write(f"**Question {idx}:** {q['question']}")
                st.write(f"a) {q['option_a']}")
                st.write(f"b) {q['option_b']}")
                st.write(f"c) {q['option_c']}")
                st.write(f"d) {q['option_d']}")
                st.write(f"**Answer:** {q['answer']}")
                st.write("---")

