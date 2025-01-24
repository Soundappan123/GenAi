import os
import streamlit as st
from langchain import PromptTemplate, LLMChain
from langchain.llms import Cohere
from langchain.memory import ConversationBufferMemory
import cohere

# Set the Cohere API key as an environment variable
os.environ["COHERE_API_KEY"] = "J4vDW2CC7v1geIbrI3AauhRZMtXjiCK6wy1Bdexk"  # Replace with your Cohere API key

# Define the Cohere model for text generation
llm = Cohere(model="command-xlarge")  # You can choose a different model if needed

# Define a LangChain prompt template
template = """
You are a helpful assistant. Answer the user's question clearly and concisely.

User: {user_input}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["user_input"], template=template
)

# Limit conversation history
memory = ConversationBufferMemory(max_context_size=3)

# Create LangChain LLMChain
chatbot = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

# Streamlit UI Setup
st.set_page_config(page_title="Cohere ChatBot", layout="wide")

st.title("Chat with Cohere Bot")

# Layout to display conversation
if "history" not in st.session_state:
    st.session_state.history = []

# Show previous messages
for message in st.session_state.history:
    if message.startswith("User:"):
        st.markdown(f'<div style="text-align: left; background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin-bottom: 10px;">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="text-align: left; background-color: #FFFFFF; padding: 10px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #ddd;">{message}</div>', unsafe_allow_html=True)

# User Input for the Chat
user_input = st.text_input("Enter your query:", "", key="input")

if user_input:
    if user_input.lower() == "exit":
        st.write("Goodbye!")
    else:
        # Get response from chatbot
        response = chatbot.run({"user_input": user_input})
        st.session_state.history.append(f"User: {user_input}")
        st.session_state.history.append(f"Bot: {response}")

        # Display chat history
        for message in st.session_state.history:
            if message.startswith("User:"):
                st.markdown(f'<div style="text-align: left; background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin-bottom: 10px;">{message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="text-align: left; background-color: #FFFFFF; padding: 10px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #ddd;">{message}</div>', unsafe_allow_html=True)
