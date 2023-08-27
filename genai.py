#!/usr/bin/env python
# coding: utf-8

# In[237]:


import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain.chains import SimpleSequentialChain

# In[238]:


os.environ['OPENAI_API_KEY'] = "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"

# In[239]:


llm_model = OpenAI(temperature=0.7)

# In[249]:


step_breakdown = "break down the user’s {instructions} into a series of steps in English"

mermaid_markup = "dont't use brackets inside the markup;Generate a flowchart using valid mermaid.js syntax using this statement: {break_sentence}"

errorcount_change = "Given the Mermaid markup, check for errors, and if any are found, rewrite the markup; otherwise, keep the original markup as is and give it as output: {markup}"

# In[250]:


step_breakdown_prompt = PromptTemplate(

    template=step_breakdown,
    input_variables=['instructions']
)

mermaid_markup_prompt = PromptTemplate(

    template=mermaid_markup,
    input_variables=['break_sentence']
)

aftererror_markup_prompt = PromptTemplate(

    template=errorcount_change,
    input_variables=['markup']
)

# In[251]:


generate_breakdown = LLMChain(llm=llm_model, prompt=step_breakdown_prompt, output_key='break_sentence', verbose=True)

generate_markup = LLMChain(llm=llm_model, prompt=mermaid_markup_prompt, output_key='markup', verbose=True)

generate_aftererror = LLMChain(llm=llm_model, prompt=aftererror_markup_prompt, output_key='final_markup', verbose=True)

# In[252]:


overall_chain = SequentialChain(

    chains=[generate_breakdown, generate_markup, generate_aftererror],
    verbose=True,
    input_variables=['instructions'],
    output_variables=["break_sentence", "markup", "final_markup"]

)

# In[253]:


from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    response = overall_chain({
                                 "instructions": "For christmas shopping, we first need to have some money. Then we go shopping, and then think what we want to buy. The things we can buy are iPhone, car or laptop"})
    print(cb)

# In[254]:


(response["markup"])

# In[255]:


# Find the index of the word "graph"
start_index = response['markup'].index("graph")

# Extract everything from the start_index to the end of the string
extracted_content = response['markup'][start_index:]

print(extracted_content)

# In[256]:


print(extracted_content)

# In[257]:

"""
import base64
from IPython.display import Image, display
import matplotlib.pyplot as plt


def mm(graph):
    graphbytes = graph.encode("ascii")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    display(Image(url="https://mermaid.ink/img/" + base64_string))


mm(
    extracted_content
)
"""


# In[261]:


# In[ ]:



