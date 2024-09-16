import os
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# Set the OpenAI API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Initialize the language model with streaming
llm = ChatOpenAI(
    streaming=True,
    temperature=0.7
)

# Create a prompt template for the interviewee
interviewee_template = """
You are an interviewee for a software engineering position. Your name is Vishwajeet and you have 5 years of experience in Python development.
You specialize in backend development and have worked extensively with Django and Flask. NOTE: you have worked on every technology that will be asked.
When answering questions, provide concrete examples from your experience. Be concise but specific.

Question: {question}
Vishwajeet's Answer:"""

interviewee_prompt = PromptTemplate(
    input_variables=["question"],
    template=interviewee_template
)

# Function to run the interview with streaming
def run_interview(question):
    prompt = interviewee_prompt.format(question=question)
    return llm.stream([HumanMessage(content=prompt)])

# Initialize session state for messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title("Interview Assistant - Python Developer")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])

# Accept user input
if user_input := st.chat_input("Enter your question here..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate assistant response and display in chat message container
    response_generator = run_interview(user_input)
    full_response = ""
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        for chunk in response_generator:
            content = chunk.content
            full_response += content
            # Update the message placeholder
            message_placeholder.markdown(full_response)
    # Add assistant message to session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})
