system_prompts = (
    "You are a medical assistant. Use only the provided context. "
    "If the context does not contain the answer, say you don't know. "
    "Answer in 1 to 3 sentences. Be concise."
)

# create messages for chat completion
def make_messages(context: str, question: str):
    return [
        {"role": "system", "content": system_prompts},
        {"role": "user", "content": f"### Context\n{context}\n\n### Question\n{question}"}
    ]