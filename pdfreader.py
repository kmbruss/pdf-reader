import warnings
warnings.filterwarnings("ignore")
 
import re
import sys
import numpy as np
import ollama
import pdfplumber
from sentence_transformers import SentenceTransformer, util
from textblob import TextBlob

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

PDF_PATH = "essay.pdf"

def correct_query(text):
    return str(TextBlob(text).correct())

 
try:
    pages = []
    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)

except FileNotFoundError:
    print(f"Error: '{PDF_PATH}' not found.")
    sys.exit(1)
 
if not pages:
    print("Error: No text could be extracted from the PDF.")
    sys.exit(1)
 
full_text = "\n\n".join(pages)

# -----------------------
# CHUNKING (sentence-based with overlap)
# -----------------------
def split_into_chunks(text, chunk_size = 5, overlap = 1):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(sentences), step):
        chunk = '. '.join(sentences[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

chunks = split_into_chunks(full_text)

# -----------------------
# EMBEDDINGS
# -----------------------
print("Embedding document chunks…")
chunk_embeddings = embedding_model.encode(chunks, show_progress_bar=True)

# -----------------------
# META-QUESTION & FOLLOW-UP DETECTION
# -----------------------
META_CONCEPTS = [
    "main argument", "central thesis", "overall point",
    "summary of the essay", "what the author concludes",
    "purpose of the essay", "core claim"
]
meta_concept_embeddings = embedding_model.encode(META_CONCEPTS)

def is_meta_question(query, threshold=0.45):
    query_emb = embedding_model.encode(query)
    scores = util.cos_sim(query_emb, meta_concept_embeddings)[0]
    return float(scores.max()) > threshold
 
 
FOLLOWUP_CONCEPTS = [
    "tell me more about that",
    "can you elaborate",
    "why is that",
    "give me an example of what you said",
    "explain further",
    "what do you mean by that"
]
followup_concept_embeddings = embedding_model.encode(FOLLOWUP_CONCEPTS)

def is_followup(query, history, threshold=0.55):
    if len(history) < 2:
        return False
    previous_query = history[-2]["content"]
    previous_answer = history[-1]["content"]
    previous_context = previous_query + " " + previous_answer
    
    query_emb = embedding_model.encode(query)
    previous_context_emb = embedding_model.encode(previous_context)

    similarity = util.cos_sim(query_emb, previous_context_emb)[0]
    return float(similarity.max()) > threshold
 
# -----------------------
# RETRIEVAL
# -----------------------
 
def find_meta_chunks(top_k=5):
    intro = chunks[:3]
    conclusion = chunks[-2:]
    combined = intro + conclusion
    return combined[:top_k]
 
def find_similar_chunks(query, top_k=5):
    query_embedding = embedding_model.encode(query)
    similarities = np.array(util.cos_sim(query_embedding, chunk_embeddings)[0])
 
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    # Always return at least the best match regardless of threshold
    results = [chunks[i] for i in top_indices if similarities[i] > 0.25]
    if not results:
        results = [chunks[i] for i in top_indices[:5]]
 
    # For meta/thesis questions, swap in purpose-built retrieval
    if is_meta_question(query):
        results = find_meta_chunks(top_k)
 
    return results

# -----------------------
# CHAT MEMORY
# -----------------------
chat_history = []
last_chunks  = []          # fix: initialised before the loop
MAX_HISTORY  = 10
 
SYSTEM_PROMPT = """
You are a pdf assistant that answers questions 
    based on the provided PDF context.
 

- if the query is just something not related to the pdf or a general question,
    you can answer it like a normal assistant without the context 
    and without mentioning the pdf.
- Use information from the CONTEXT section to answer.
- If there is truly no relevant information in the context, 
    say "I couldn't find that in the document."
- If the answer can be reasonably inferred from the context, 
    go ahead and answer — you don't need an exact quote.- Quote directly when asked for exact wording, thesis statements, or definitions.
- Be concise unless the user asks for detail."""

# -----------------------
# MAIN LOOP
# -----------------------
print("\n📄 PDF Chat ready! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    user_input = correct_query(user_input)

    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    
    if is_followup(user_input, chat_history) and last_chunks:
        relevant_chunks = last_chunks  # reuse previous chunks
    else:
        relevant_chunks = find_similar_chunks(user_input)
        last_chunks = relevant_chunks  # save for potential follow-up

    context = "\n\n---\n\n".join(relevant_chunks)

    # -----------------------
    # BETTER PROMPT
    # -----------------------
    augmented_user_message = f""" CONTEXT: {context} QUESTION: {user_input}"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *chat_history,
            {"role": "user", "content": augmented_user_message}
        ]
    )

    answer = response["message"]["content"]
    # -----------------------
    # STORE MEMORY
    # -----------------------
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": answer})

    if len(chat_history) > MAX_HISTORY:
        chat_history = chat_history[-MAX_HISTORY:]

    print("\nPDF Reader:", answer, "\n")