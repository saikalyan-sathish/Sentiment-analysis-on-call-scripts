import requests
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from transformers import AutoTokenizer

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Hugging Face API URL and API Key
API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
HF_API_KEY = os.getenv('HF_API_KEY')

# Ensure the API key is loaded correctly
if not HF_API_KEY:
    raise ValueError("HF_API_KEY is not set in the environment. Please provide a valid key.")
else:
    print("API Key Loaded Successfully.")

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# Initialize Hugging Face tokenizer to properly handle token limits
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

def query_huggingface_api(text):
    """
    Function to query the Hugging Face API for sentiment analysis.
    """
    try:
        # Sending the request to Hugging Face API
        response = requests.post(API_URL, headers=headers, json={"inputs": text})
        
        # Debug: Log the response status and content for troubleshooting
        print(f"Response Status: {response.status_code}")
        print(f"Response Content: {response.text}")
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to get a response from Hugging Face API: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        # Log request exception if any
        print(f"Request error: {e}")
        return {"error": str(e)}

def split_text_to_tokens(text, max_tokens=500):
    """
    Function to preprocess text, convert it to tokens, and split into chunks of max_tokens.
    """
    # Tokenize the text to handle word-level chunking properly
    tokenized_text = tokenizer.encode(text, truncation=False, padding=False)
    
    # Debug: Log the length of tokenized text
    print(f"Tokenized text length: {len(tokenized_text)}")

    # Check if the text exceeds the max_length and split accordingly
    text_chunks = [tokenizer.decode(tokenized_text[i:i + max_tokens], skip_special_tokens=True) 
                   for i in range(0, len(tokenized_text), max_tokens)]
    
    # Debug: Log how many chunks were created
    print(f"Created {len(text_chunks)} chunks.")
    return text_chunks

def map_label_to_sentiment(label):
    """
    Map the model's label (LABEL_0, LABEL_1, LABEL_2) to positive/negative sentiment labels.
    """
    if label == 'LABEL_0':
        return 'negative'
    elif label == 'LABEL_2':
        return 'positive'
    else:
        return 'neutral'  # Optional: you can ignore this if you want only positive/negative

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """
    Endpoint to analyze text for sentiment using Hugging Face model.
    """
    try:
        # Get JSON data from the request
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Split the text into chunks if it's too long (in tokens, not characters)
        text_chunks = split_text_to_tokens(text)

        # Perform sentiment analysis for each chunk
        sentiments = []
        for chunk in text_chunks:
            sentiment = query_huggingface_api(chunk)
            if 'error' not in sentiment:
                # Process the sentiment labels to return only "positive" and "negative"
                processed_sentiment = []
                for result in sentiment[0]:  # Fix this line to iterate over the first list in the response
                    sentiment_label = map_label_to_sentiment(result['label'])
                    if sentiment_label != 'neutral':  # Only include positive and negative
                        sentiment_score = result['score']
                        processed_sentiment.append({
                            'sentiment': sentiment_label,
                            'score': sentiment_score
                        })
                sentiments.append(processed_sentiment)
            else:
                print(f"Error in chunk analysis: {sentiment['error']}")

        # Aggregate results (for simplicity, we'll return all chunk responses)
        return jsonify({"sentiments": sentiments})

    except Exception as e:
        # Catch any other errors
        print(f"Error processing the request: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
