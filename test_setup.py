# test_setup.py
import os
from dotenv import load_dotenv

# Test 1: Environment
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key and api_key != "your_key_here":
    print("✓ API key loaded")
else:
    print("✗ API key missing - check your .env file")

# Test 2: Core imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    print("✓ LangChain + Gemini import works")
except ImportError as e:
    print(f"✗ Import error: {e}")

try:
    from langchain_community.document_loaders import PyPDFLoader
    print("✓ PDF loader import works")
except ImportError as e:
    print(f"✗ Import error: {e}")

try:
    from langchain_community.vectorstores import Chroma
    print("✓ ChromaDB import works")
except ImportError as e:
    print(f"✗ Import error: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("✓ Sentence transformers import works")
except ImportError as e:
    print(f"✗ Import error: {e}")

# Test 3: Check data folder
pdf_count = len([f for f in os.listdir("data") if f.endswith(".pdf")])
if pdf_count > 0:
    print(f"✓ Found {pdf_count} PDF(s) in data folder")
else:
    print("✗ No PDFs in data folder - add some papers")

# Test 4: Test Gemini connection
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    response = llm.invoke("Say 'hello' and nothing else")
    print(f"✓ Gemini API works: {response.content}")
except Exception as e:
    print(f"✗ Gemini API error: {e}")

print("\n--- Setup complete! ---")