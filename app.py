from flask import Flask, render_template, jsonify, request
from src.helper import create_embedding_model
from src.prompt import *
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from dotenv import load_dotenv
import os

app = Flask(__name__) # initialize Flask app

# home route
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)