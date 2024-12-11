from langchain import OpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
from dotenv import load_dotenv
import json
import streamlit as st

load_dotenv()

def csv_tool(filename):
    if filename is None:  # Check for file input
        raise ValueError("No file uploaded.")

    df = pd.read_csv(filename)
    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
    return create_pandas_dataframe_agent(llm, df, verbose=True)

def ask_agent(agent, query):
    """
    Query an agent and return the response as a string.

    Args:
        agent: The agent to query.
        query: The query to ask the agent.

    Returns:
        The response from the agent as a string.
    """
    prompt = (
        """
        Let's decode the way to respond to the queries. The responses depend on the type of information requested in the query. 

        1. If the query requires a table, format your answer like this:
           {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

        2. For a bar chart, respond like this:
           {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

        3. If a line chart is more appropriate, your reply should look like this:
           {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

        Note: We only accommodate two types of charts: "bar" and "line".

        4. For a plain question that doesn't need a chart or table, your response should be:
           {"answer": "Your answer goes here"}

        For example:
           {"answer": "The Product with the highest Orders is '15143Exfo'"}

        5. If the answer is not known or available, respond with:
           {"answer": "I do not know."}

        Return all output as a valid JSON string. Use double quotes around all strings (e.g., {"columns": ["Products", "Orders"], "data": [["51993Masc", 191], ["49631Foun", 152]]}). Do not use single quotes.

       Now, respond **ONLY** with a valid JSON string, without any additional text or formatting. Here's the query for you to work on: 
        """
        + query
    )

    # Query the agent
    response = agent.run(prompt)

    # Normalize the response to ensure valid JSON
    response = normalize_response(response)

    # Debugging output
    print(f"Raw response after normalization: {response}")

    return response

def normalize_response(response: str) -> str:
    """
    Normalize the response string to ensure it adheres to valid JSON formatting.

    Args:
        response: The raw response string from the agent.

    Returns:
        A normalized JSON string.
    """
    # Replace single quotes with double quotes
    response = response.replace("'", '"')
    return response

def decode_response(response: str) -> dict:
    """
    Decode the response string into a Python dictionary.

    Args:
        response: The response string to decode.

    Returns:
        A dictionary representation of the response.
    """
    try:
        response_dict = json.loads(response)
        return response_dict
    except json.JSONDecodeError as e:
        print(f"Error decoding response: {e}")
        print(f"Raw response was: {response}")
        return {"error": "Invalid response format. Please ensure the model outputs valid JSON."}

def write_answer(response_dict: dict):
    """
    Write the response to the Streamlit UI based on its type.

    Args:
        response_dict: The decoded response dictionary.
    """
    if "error" in response_dict:
        st.write(f"Error: {response_dict['error']}")
        return

    if "answer" in response_dict:
        st.write(response_dict["answer"])

    if "bar" in response_dict:
        data = response_dict["bar"]
        try:
            df = pd.DataFrame({
                col: [x[i] for x in data["data"]]  # Extract data column-wise
                for i, col in enumerate(data["columns"])
            })
            df.set_index(data["columns"][0], inplace=True)  # Set correct index column
            st.bar_chart(df)
        except (ValueError, IndexError) as e:
            st.write(f"Couldn't create bar chart: {e}")

    if "line" in response_dict:
        data = response_dict["line"]
        try:
            df = pd.DataFrame({
                col: [x[i] for x in data["data"]]  # Extract data column-wise
                for i, col in enumerate(data["columns"])
            })
            df.set_index(data["columns"][0], inplace=True)  # Set correct index column
            st.line_chart(df)
        except (ValueError, IndexError) as e:
            st.write(f"Couldn't create line chart: {e}")

    if "table" in response_dict:
        data = response_dict["table"]
        try:
            df = pd.DataFrame(data["data"], columns=data["columns"])
            st.table(df)
        except ValueError as e:
            st.write(f"Couldn't create table: {e}")

# Streamlit UI
st.set_page_config(page_title="👨‍💻 Talk with your File")
st.title("👨‍💻 Talk with your File")

st.write("Please upload your file below.")

data = st.file_uploader("Please Upload your File", type="csv")

query = st.text_area("Send a Message")

if st.button("Submit Query", type="primary"):
    try:
        # Create an agent from the CSV file
        agent = csv_tool(data)

        # Query the agent
        response = ask_agent(agent=agent, query=query)

        # Decode the response
        decoded_response = decode_response(response)

        # Write the response to the Streamlit app
        write_answer(decoded_response)
    except Exception as e:
        st.error(f"An error occurred: {e}")
