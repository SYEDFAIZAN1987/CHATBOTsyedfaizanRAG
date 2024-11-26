#%% Packages
import os
import re
from dotenv import load_dotenv
from pprint import pprint
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import chromadb
import openai

# Load environment variables from .env file
load_dotenv()

# Ensure the OpenAI API key is set
openai.api_key = os.getenv("OPENAI_API_KEY")

#%% Load the PDF file
project_report_file = "ALY_6080_Experential_learning_Group_1_Module_12_Capstone_Sponsor_Deliverable.pdf"
reader = PdfReader(project_report_file)
report_texts = [page.extract_text().strip() for page in reader.pages]

#%% Filter out unnecessary sections (adjust as needed)
filtered_texts = report_texts[5:-5]  # Modify indices based on the document structure

# Remove headers/footers or other unwanted text patterns
cleaned_texts = [re.sub(r'\d+\n.*?\n', '', text) for text in filtered_texts]

#%% Split text into chunks
char_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=0.2
)

texts_char_splitted = char_splitter.split_text('\n\n'.join(cleaned_texts))
print(f"Number of chunks: {len(texts_char_splitted)}")

#%% Token splitting for efficient querying
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0.2,
    tokens_per_chunk=256
)

texts_token_splitted = []
for text in texts_char_splitted:
    try:
        texts_token_splitted.extend(token_splitter.split_text(text))
    except Exception as e:
        print(f"Error in text: {text}. Error: {e}")
        continue

print(f"Number of tokenized chunks: {len(texts_token_splitted)}")

#%% Initialize Vector Database
chroma_client = chromadb.PersistentClient(path="db")
chroma_collection = chroma_client.get_or_create_collection("aly6080")

#%% Add documents to the database
ids = [str(i) for i in range(len(texts_token_splitted))]
chroma_collection.add(
    ids=ids,
    documents=texts_token_splitted
)

#%% Define RAG Query Function
def rag(query, n_results=5):
    """Retrieve and generate response based on the query."""
    try:
        # Query the database
        res = chroma_collection.query(query_texts=[query], n_results=n_results)
        docs = res["documents"][0]
        joined_information = "; ".join([f"{doc}" for doc in docs])

        # Prepare conversation with context and query
        messages = [
            {
                "role": "system",
                "content": "You are an expert on socio-economic analysis for the Greater Toronto Area, with specific knowledge from the ALY 6080 Group 1 report. Answer queries using only the provided document context."
            },
            {"role": "user", "content": f"Question: {query}\nContext: {joined_information}"}
        ]

        # Generate response using OpenAI
        model = "gpt-3.5-turbo"
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages
        )
        content = response.choices[0].message.content
        return content
    except Exception as e:
        return f"Error generating response: {str(e)}"

#%% Example Query
query = "What are the key findings regarding financial stability in the Greater Toronto Area?"
response = rag(query=query, n_results=5)
pprint(response)
