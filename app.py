from flask import Flask, render_template, jsonify, request
from src.helper import create_embedding_model
from src.prompt import *
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from dotenv import load_dotenv
import os

