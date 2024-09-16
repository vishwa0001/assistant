import os
import gradio as gr
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
import streamlit as st
import os
import gradio as gr
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

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

# Gradio interface
def interview_interface(message, history):
    history = history or []
    response_generator = run_interview(message)
    response = ""
    for chunk in response_generator:
        response += chunk.content
        yield history + [(message, response)]

# Gradio app layout
with gr.Blocks() as app:
    gr.Markdown("## Interview Assistant - Python Developer")
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Enter your question")
    clear = gr.Button("Clear")

    msg.submit(interview_interface, [msg, chatbot], chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch the Gradio app
if __name__ == "__main__":
    app.queue().launch(server_port=7860)