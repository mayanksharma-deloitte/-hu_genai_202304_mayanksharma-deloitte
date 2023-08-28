import streamlit as st
import os
import base64
import matplotlib.pyplot as plt
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = "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"

# Initialize OpenAI model
llm_model = OpenAI(temperature=0.7)

# Prompt templates
step_breakdown_prompt = PromptTemplate(
    template="break down the user’s {instructions} into a series of steps in English",
    input_variables=['instructions']
)

mermaid_markup_prompt = PromptTemplate(
    template="Generate a flowchart using valid mermaid.js syntax using this statement: {break_sentence}",
    input_variables=['break_sentence']
)

aftererror_markup_prompt = PromptTemplate(
    template="Given the Mermaid markup, check for errors, and if any are found, rewrite the markup; otherwise, keep the original markup as is and give it as output: {markup}",
    input_variables=['markup']
)

# Chain setup
generate_breakdown = LLMChain(llm=llm_model, prompt=step_breakdown_prompt, output_key='break_sentence', verbose=True)
generate_markup = LLMChain(llm=llm_model, prompt=mermaid_markup_prompt, output_key='markup', verbose=True)
generate_aftererror = LLMChain(llm=llm_model, prompt=aftererror_markup_prompt, output_key='final_markup', verbose=True)
overall_chain = SequentialChain(
    chains=[generate_breakdown, generate_markup, generate_aftererror],
    verbose=True,
    input_variables=['instructions'],
    output_variables=["break_sentence", "markup", "final_markup"]
)

# Streamlit app
st.title("Langchain Streamlit App")
instructions = st.text_area("Enter instructions:")
if st.button("Generate"):
    with get_openai_callback() as cb:
        response = overall_chain({"instructions": instructions})
    print(cb)
    st.text("Generated Markup:")
    st.text(response["markup"])

    # Find the index of the word "graph"
    start_index = response['markup'].index("graph")
    extracted_content = response['markup'][start_index:]

    st.text("Extracted Content:")
    st.text(extracted_content)

    # Display the mermaid diagram
    graphbytes = extracted_content.encode("ascii")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    st.image("https://mermaid.ink/img/" + base64_string)

