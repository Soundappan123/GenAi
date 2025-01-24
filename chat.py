# import streamlit as st
# from typing import Sequence
# from typing_extensions import Annotated, TypedDict
# from langchain_groq import ChatGroq
# from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import START, StateGraph
# from langgraph.graph.message import add_messages
# import os

# # Set up the Groq API key

# # Initialize the Groq model
# model = ChatGroq(model="llama3-8b-8192",api_key="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")

# # Define a custom state schema
# class State(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]
#     language: str

# # Define the prompt template
# prompt_template = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )

# # Store for session histories
# store = {}

# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     """Retrieve or create a chat history for a given session ID."""
#     if session_id not in store:
#         store[session_id] = InMemoryChatMessageHistory()
#     return store[session_id]

# # Wrap the model with session-specific chat history
# with_message_history = RunnableWithMessageHistory(model, get_session_history)

# # Streamlit App
# st.title("Dynamic Chatbot with Session Management")
# st.sidebar.title("Session Management")

# # Session ID input
# session_id = st.sidebar.text_input("Session ID", "default_session")
# language = st.sidebar.selectbox("Select Language", ["English", "Spanish", "French"], index=0)

# # Retrieve session-specific chat history
# chat_history = get_session_history(session_id)
# if "system_message" not in st.session_state:
#     st.session_state.system_message = SystemMessage(content=f"You are a helpful assistant. Answer all questions to the best of your ability in {language}.")

# # Chat input and output
# user_input = st.text_input("You:", placeholder="Type your message here...", key="user_input")
# if st.button("Send"):
#     if user_input:
#         # Add user input to the history
#         chat_history.add_user_message(user_input)
        
#         # Prepare input messages
#         input_messages = [st.session_state.system_message] + chat_history.messages
        
#         # Invoke the chatbot
#         output = with_message_history.invoke(input_messages, config={"session_id": session_id})
        
#         # Add chatbot response to history
#         chat_history.add_ai_message(output.content)
#         st.session_state.user_input = ""  # Clear input box

# # Display the conversation
# st.subheader("Chat History")
# for message in chat_history.messages:
#     if isinstance(message, HumanMessage):
#         st.markdown(f"**You:** {message.content}")
#     elif isinstance(message, SystemMessage):
#         st.markdown(f"**System:** {message.content}")
#     else:
#         st.markdown(f"**Chatbot:** {message.content}")




# import streamlit as st
# from typing import Sequence
# from typing_extensions import Annotated, TypedDict
# from langchain_groq import ChatGroq
# from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# import os

# # Set up the Groq API key
# groq_api_key = "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"  # Replace with your actual API key
# os.environ["GROQ_API_KEY"] = groq_api_key

# # Initialize the Groq model
# model = ChatGroq(model="llama3-8b-8192")

# # Define a custom state schema
# class State(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], list]
#     language: str

# # Define the prompt template
# prompt_template = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )

# # Store for session histories
# store = {}

# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     """Retrieve or create a chat history for a given session ID."""
#     if session_id not in store:
#         store[session_id] = InMemoryChatMessageHistory()
#     return store[session_id]

# # Wrap the model with session-specific chat history
# with_message_history = RunnableWithMessageHistory(model, get_session_history)

# # Streamlit App
# st.title("Dynamic Chatbot with Session Management")
# st.sidebar.title("Session Management")

# # Session ID input
# session_id = st.sidebar.text_input("Session ID", "default_session")

# # Language input
# language = st.sidebar.text_input("Language", "English", placeholder="Enter a language, e.g., English, French")

# # Retrieve session-specific chat history
# chat_history = get_session_history(session_id)
# if "system_message" not in st.session_state:
#     st.session_state.system_message = SystemMessage(content=f"You are a helpful assistant. Answer all questions to the best of your ability in {language}.")

# # Initialize chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Chat input and output
# with st.form("chat_form", clear_on_submit=True):
#     user_input = st.text_input("You:", placeholder="Type your message here...", key="user_input")
#     submitted = st.form_submit_button("Send")

# if submitted and user_input:
#     # Update the system message dynamically based on the language input
#     st.session_state.system_message = SystemMessage(content=f"You are a helpful assistant. Answer all questions to the best of your ability in {language}.")
    
#     # Add user input to the history
#     chat_history.add_user_message(user_input)
    
#     # Prepare input messages
#     input_messages = [st.session_state.system_message] + chat_history.messages
    
#     # Invoke the chatbot
#     output = with_message_history.invoke(input_messages, config={"session_id": session_id})
    
#     # Add chatbot response to history
#     chat_history.add_ai_message(output.content)
    
#     # Append to chat history for UI
#     st.session_state.chat_history.append(("You", user_input))
#     st.session_state.chat_history.append(("Chatbot", output.content))

# # Display the conversation
# st.subheader("Chat History")
# for sender, message in st.session_state.chat_history:
#     if sender == "You":
#         st.markdown(f"**You:** {message}")
#     else:
#         st.markdown(f"**Chatbot:** {message}")


import streamlit as st
import os
from typing import Sequence
from typing_extensions import Annotated, TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

# Set up the Groq API key
os.environ["GROQ_API_KEY"] = st.text_input("Enter your Groq API key", type="password")

# Initialize the Groq model
model = ChatGroq(model='llama3-8b-8192')

# Define a custom state schema
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

# Define the prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Store for session histories
store = {}

def get_session_history(session_id: str):
    """Retrieve or create a chat history for a given session ID."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Wrap the model with session-specific chat history
with_message_history = RunnableWithMessageHistory(model, get_session_history)

# Add memory to persist context
memory = MemorySaver()

# Define the workflow graph
workflow = StateGraph(state_schema=State)

def call_model(state: State):
    """Generate a response using the wrapped model."""
    prompt = prompt_template.invoke(state)
    response = with_message_history.invoke(prompt)
    return {"messages": [response]}

# Add nodes and edges to the workflow graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Compile the workflow
app = workflow.compile(checkpointer=memory)

# Streamlit Chatbot
st.title("Dynamic Chatbot with Session Management")

# Input for session ID
session_id = st.text_input("Enter your session ID:")

if session_id:
    # Initialize session history
    chat_history = get_session_history(session_id)
    language = "English"  # Default language
    system_message = SystemMessage(content=f"You are a helpful assistant. Answer all questions to the best of your ability in {language}.")
    
    # Input for user query
    user_input = st.text_input("Your Query:")
    if st.button("Send Query") and user_input:
        # Add user message to history
        chat_history.add_user_message(user_input)
        
        # Format the messages
        input_messages = [system_message] + chat_history.messages
        
        # Ensure messages are of correct type
        input_messages = [message for message in input_messages if isinstance(message, BaseMessage)]
        
        # Pass the session_id in the config dictionary for both state and memory
        config = {"configurable": {"session_id": session_id}}
        
        # Invoke the chatbot and pass the config with session_id
        output = with_message_history.invoke(input_messages, config=config)
        
        # Display the chatbot response
        st.text_area("Chatbot Response", value=output.content, height=200)
        
        # Add AI response to history
        chat_history.add_ai_message(output.content)

    # Display chat history
    st.subheader("Chat History")
    for message in chat_history.messages:
        st.write(f"{'You' if isinstance(message, HumanMessage) else 'Chatbot'}: {message.content}")
