import os
from apikey import OPENAI_apikey
from apikey import SERPER_apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool


os.environ['OPENAI_API_KEY'] = OPENAI_apikey
os.environ['SERPER_API_KEY'] = SERPER_apikey


#APP Framework
st.title(':red[BibleGPT] ✝')
st.caption('Created by Alex Seo')
st.markdown(':black[Ask any faith-based questions!]')
prompt = st.text_input('Plug in your prompt here')


# Prompt Templates

google_template = PromptTemplate(
    input_variables=['google_search'],
    template='What does the Christian Bible say about: {google_search}?'
    # ^change to whatever fits
)

title_template = PromptTemplate(
    input_variables=['topic'], 
    template='Give me about 3-4 relevant biblical scripture with its respective biblical contexts, regarding this topic: {topic}' 
     #^change this to whatever fits

)

script_template = PromptTemplate(
    input_variables=['title'], 
    template='write me a very short advice based on this title TITLE: {title} ' 
    # ^change this to whatever fits

)

search=GoogleSerperAPIWrapper(k=3) #k=number of results

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
google_memory = ConversationBufferMemory(input_key='google_search', memory_key='chat_history')


# LLMs 
llm = OpenAI(temperature=0.9, max_tokens=1000) 

title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)
google_chain = LLMChain(llm=llm, prompt=google_template, verbose=True, output_key='google_search', memory=google_memory)

tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search",
    )
]

# Function to run the chains and get outputs
def run_chains(title, google_search):
    title_output = title_chain.run(topic=title)
    script_output = script_chain.run(title=title_output) # Use the title_output directly as input for script_chain
    google_output = google_chain.run(google_search=google_search)
    return title_output, script_output, google_output



#show stuff to screen if there is prompt
if prompt:
    title_result, script_result, google_result = run_chains(title=prompt, google_search=prompt)

    st.write(google_result) # response from google
    st.empty()
    st.divider() # dividing each result
    st.write(title_result) # response from openai relevant bible verses
    st.empty()
    st.divider() # dividing each result
    st.write(script_result) # response from openai advice
    
