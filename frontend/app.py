import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.title("🚢 Titanic Chatbot")

query = st.text_input("Ask a question about the Titanic dataset:")

if st.button("Ask"):
    response = requests.get(f"{BACKEND_URL}/query", params={"question": query})
    answer = response.json()["answer"]
    st.write("🤖 Chatbot:", answer)

st.subheader("Passenger Age Histogram")
if st.button("Show Histogram"):
    st.image(f"{BACKEND_URL}/visualization/age_histogram")
