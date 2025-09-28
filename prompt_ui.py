from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()

# Configure OpenRouter model
@st.cache_resource
def load_model():
    return ChatOpenAI(
        model='gpt-3.5-turbo',
        temperature=0.7, 
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1"
    )

llm = load_model()

st.header('🤖 AI Research Tool Demo using model gpt-3.5-turbo')
st.caption(' I\'m Rajesh-your research assistanat')

paper_input = st.selectbox( "Select Research Paper Name", ["Select...","Attention is All You Need","BERT:pre-training of Deep Bidirectional Transformers","GPT-3:Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly","Technical","Code-Oriented","Mathematical"])

length_input = st.selectbox( "Select Explanation Length", ["short (1-2 paragraphs)","medium(3-4 paragraphs)","long(detailed explanation)"])
#template

template = load_prompt('template.json')
# fill the placeholders
prompt=template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
})


col1, col2 = st.columns([1, 4])
with col1:
    ask_button = st.button("Summarize", type="primary")
with col2:
    if st.button("🗑️ Clear"):
        st.session_state.user_input = ""
        st.rerun()

if ask_button:
    if prompt:
        with st.spinner('🤔 Thinking...'):
            try:
                # Update model with current settings
                current_model = ChatOpenAI(
                    model='gpt-3.5-turbo',
                    temperature=0.7,
                    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                    openai_api_base="https://openrouter.ai/api/v1"
                )
                
                result = current_model.invoke(prompt)
                
               
                st.write(result.content)
                

                    
            except Exception as e:
                st.error(f" Error: {str(e)}")
                st.write("Please check your API key and internet connection.")
    else:
        st.warning(" Please enter a question to get an answer.")
