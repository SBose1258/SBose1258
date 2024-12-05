# -*- coding: utf-8 -*-
"""dataset_querybot_using_openai_api.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rjJ8VA3HCVX4ek7fvkj-0Bm4w29gO-Nr
"""

import pandas as pd
import os
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import json

path="/content/drive/MyDrive/Prompt_Project_Datasets/diabetes.csv"
df=pd.read_csv(path)
df.head(5)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def testing_queustion(question):
    """Summarizes the text using OpenAI's API."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": question}
        ],
        max_tokens=100,
        temperature=0.7
    )
    summary = response.choices[0].message.content
    return summary

testing_queustion("Say 'Now I am Up and Running to answer")

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)
def query_data(data_json, question, model_name):
    prompt = f"Based on the following relational data, answer the question:\n{data_json}\nQuestion: {question}"
    #prompt = f"Based on the following relational data, write Python code to create a visualization for the question:\n{data_json}\nQuestion: {question}"


    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "you are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        model=model_name,
    )
    return chat_completion.choices[0].message.content
# Limiting the data
limited_data = df.head(100)

# Convert DataFrame to JSON or other structured format
data_json = limited_data.to_json(orient='records')

# Define the model you want to use
#model_name = 'gpt-3.5-turbo'  # Change to the model we want to use
#model_name = 'gpt-4o-mini'
model_name = 'gpt-4o'

# Question to test
#question = "Summarize the text"

#query_data(data_json, question, model_name)

while True:
    question = input("Ask your question (or say 'bye' or 'tata' to exit): ")
    if question.lower() in ["bye", "tata"]:
        print("Have a nice day Goodbye!")
        break

    # Call query_data function and print response
    response = query_data(data_json, question, model_name)
    print("Response:", response)