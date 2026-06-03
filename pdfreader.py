from pypdf import PdfReader
import ollama

# load pdf and extract text
reader = PdfReader("essay.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if (page_text):
        text += page_text + "\n"

text = " ".join(text.split())

# split text into chunks of x characters
def split_text(text, chunk_size=1000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
chunks = split_text(text)



# send text to ollama
print("\n📄 PDF Chat ready! Type 'exit' to quit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    prompt = f"""
        You are answering questions about a PDF.

        Use ONLY the information inside the PDF.
        If the PDF does not contain the answer, say:
        "I couldn't find that in the document."

        PDF content: {text}

        Question: {user_input}
        """
    
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    print()
    answer = response["message"]["content"]
    print(f"PDF Reader: {answer}\n")


