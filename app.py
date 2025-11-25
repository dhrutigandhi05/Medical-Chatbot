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

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embedding = create_embedding_model()
index_name = "medical-chatbot"

# load existing vector store
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embedding,
    index_name=index_name
)

# retrieve top 3 similar/relevant responses from the knowledge base
retriever = docsearch.as_retriever(
    search_type="similarity", 
    search_kwargs={"k":3}
) 

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name) # load tokenizer
tokenizer.pad_token = tokenizer.eos_token # silent warning about no pad token
# load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,   # CPU friendly
    device_map="cpu"
)

# create text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=False,
    temperature=0.0,
    return_full_text=False
)

# chatModel = HuggingFacePipeline(pipeline=pipe)

# build prompt for llama model
def build_llama_prompt(context: str, question: str) -> str:
    # construct messages
    messages = [
        {"role": "system", "content": system_prompts},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    # apply chat template using tokenizer
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

# RAG answer function
def rag_answer(question: str) -> str:
    docs = retriever.get_relevant_documents(question) # retrieve relevant documents
    context = "\n\n".join(d.page_content for d in docs) # build context string

    prompt = build_llama_prompt(context, question) # build prompt for llama model
    answer = pipe(prompt)[0]["generated_text"].strip() # generate answer

    return answer

@app.route("/get", methods=["POST"])
def chat():
    user_msg = request.form.get("msg") or request.form.get("message") # get user message

    # handle empty input
    if not user_msg or not user_msg.strip():
        return "Please enter a question."

    print("User:", user_msg)
    answer = rag_answer(user_msg)
    print("Answer:", answer)

    return answer

# home route
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)