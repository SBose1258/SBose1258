from langchain import OpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
from dotenv import load_dotenv 
import json
import streamlit as st
load_dotenv()

def csv_tool(filename):
    if filename is None:  # Added check for file input
        raise ValueError("No file uploaded.")

    df = pd.read_csv(filename)
    llm=OpenAI(temperature=0, model_name="gpt-4o")
    return create_pandas_dataframe_agent(llm, df, verbose=True)
    
    #return create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)


def ask_agent(agent, query):
    """
    Query an agent and return the response as a string.

    Args:
        agent: The agent to query.
        query: The query to ask the agent.

    Returns:
        The response from the agent as a string.
    """
    # Prepare the prompt with query guidelines and formatting
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

        Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes. 
        For example: {"columns": ["Products", "Orders"], "data": [["51993Masc", 191], ["49631Foun", 152]]}

       Now, respond **ONLY** with a valid JSON string, without any additional text or formatting. Here's the query for you to work on: 
        """
        + query
    )

     # Log the raw response to debug issues
    response = agent.run(prompt)
    
    # Debugging output
    print(f"Raw response from agent: {response}")  
    return str(response)

# Updated decode_response with improved error handling
def decode_response(response: str) -> dict:
    try:
        response_dict = json.loads(response)
        return response_dict
    except json.JSONDecodeError as e:
        print(f"Error decoding response: {e}")
        print(f"Raw response was: {response}")
        return {"error": "Invalid response format. Please ensure the model outputs valid JSON."}

#write answer
def write_answer(response_dict: dict):
    if "error" in response_dict:
        st.write(f"Error: {response_dict['error']}")
        return

    if "answer" in response_dict:
        st.write(response_dict["answer"])

    if "bar" in response_dict:
        data = response_dict["bar"]
        try:
            df_data = {
                col: [x[i] for x in data["data"]]  # Fixed data extraction
                for i, col in enumerate(data["columns"])
            }
            df = pd.DataFrame(df_data)
            df.set_index(data["columns"][0], inplace=True)  # Set correct index column
            st.bar_chart(df)
        except (ValueError, IndexError) as e:
            print(f"Couldn't create bar chart: {e}")
    
    if "line" in response_dict:
        data = response_dict["line"]
        try:
            df_data = {
                col: [x[i] for x in data["data"]]  # Fixed data extraction
                for i, col in enumerate(data["columns"])
            }
            df = pd.DataFrame(df_data)
            df.set_index(data["columns"][0], inplace=True)  # Set correct index column
            st.line_chart(df)
        except (ValueError, IndexError) as e:
            print(f"Couldn't create line chart: {e}")

    if "table" in response_dict:
        data = response_dict["table"]
        try:
            # Create DataFrame from the response data
            df = pd.DataFrame(data["data"], columns=data["columns"])

            # Attempt to convert numeric columns to proper types
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')  # Non-numeric columns remain unchanged

            st.table(df)
        except ValueError as e:
            print(f"Couldn't create table: {e}")




#streamlit UI    
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
