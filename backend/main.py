from fastapi import FastAPI
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fastapi.responses import FileResponse
import openai
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI


# Load environment variables (API key)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI
app = FastAPI()

# Load Titanic dataset
df = sns.load_dataset("titanic")

# Set up LangChain LLM
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)
template = PromptTemplate.from_template("Answer the following question: {question}")

# Create LangChain chain
def query_langchain(question):
    chain = LLMChain(llm=llm, prompt=template)
    return chain.run(question)

@app.get("/")
def home():
    return {"message": "Titanic Chatbot with LangChain is Running!"}

@app.get("/query")
def query_data(question: str):
    question = question.lower()

    # Titanic-specific questions
    if "percentage of passengers were male" in question:
        male_percentage = df['sex'].value_counts(normalize=True)['male'] * 100
        return {"answer": f"{male_percentage:.2f}% of passengers were male."}

    elif "average ticket fare" in question:
        avg_fare = df["fare"].mean()
        return {"answer": f"The average ticket fare was ${avg_fare:.2f}."}

    elif "how many passengers embarked" in question:
        embarked_counts = df["embark_town"].value_counts().to_dict()
        return {"answer": embarked_counts}

    # If it's a general question, use LangChain
    else:
        response = query_langchain(question)
        return {"answer": response}
    
# Function to generate the histogram
@app.get("/visualization/age_histogram")
def age_histogram():
    plt.figure(figsize=(8, 6))
    sns.histplot(df["age"].dropna(), bins=20, kde=True, color="blue")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.title("Titanic Passenger Age Distribution")
    
    # Save the histogram
    image_path = "age_histogram.png"
    plt.savefig(image_path)
    plt.close()
    
    # Return the image file
    return FileResponse(image_path, media_type="image/png")

