import streamlit as st
import requests

# Set up the Streamlit UI
st.title("Call Transcript Sentiment Analysis")

# File uploader to upload text files
uploaded_files = st.file_uploader("Upload Call Transcripts", type="txt", accept_multiple_files=True)

# Backend API URL
API_URL = "http://localhost:5000/analyze"

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Read the file content
        content = uploaded_file.read().decode("utf-8")
        
        # Display the uploaded transcript
        st.subheader(f"Transcript: {uploaded_file.name}")
        st.text(content)
        
        # Send the transcript to the Flask backend for sentiment analysis
        response = requests.post(API_URL, json={"text": content})
        
        if response.ok:
            sentiment = response.json()
            st.subheader("Sentiment Analysis Results:")
            st.json(sentiment)
        else:
            st.error("Error: Unable to analyze the sentiment.")
