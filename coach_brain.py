import os
import chromadb
from chromadb.utils import embedding_functions
from google import genai

print("🧠 Initializing Cloud-Powered Generative AI Coaching Layer...")

# ==================================================
# 1. SETUP CLOUD LLM ENGINE (GEMINI API)
# ==================================================
# PASTE YOUR GOOGLE AI STUDIO API KEY HERE:
import streamlit as st
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Initialize the blazing fast Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# ==================================================
# 2. CONNECT LOCAL KNOWLEDGE BASE
# ==================================================
db_path = os.path.join(os.getcwd(), "chroma_db")
chroma_client = chromadb.PersistentClient(path=db_path)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_collection(name="pt_recovery_protocols", embedding_function=ef)

def generate_ai_coaching_plan(movement_telemetry):
    print(f"\n📥 Processing Telemetry: {movement_telemetry}")
    
    # VECTOR SEARCH: Find the protocol in our local database
    print("🔍 Scanning local vector space for clinical protocols...")
    db_results = collection.query(query_texts=[movement_telemetry], n_results=1)
    
    matched_protocol = db_results['documents'][0][0]
    protocol_title = db_results['metadatas'][0][0]['title']
    print(f"🎯 Retrieved Reference: {protocol_title}")

    # CONSTRUCT THE PROMPT
    prompt = f"""
    You are an elite Sports Physical Therapist and Biomechanics Coach at Iron Founder AI. 
    Your job is to take raw movement telemetry and a specific clinical reference protocol, 
    and synthesize them into a highly clear, professional, and encouraging action plan for an athlete.

    ATHLETE TELEMETRY DATA:
    {movement_telemetry}

    CLINICAL REFERENCE PROTOCOL:
    {matched_protocol}

    INSTRUCTIONS:
    Write a 3-step actionable recovery plan tailored specifically to the athlete's telemetry. 
    Incorporate the exercises from the reference protocol. Keep the tone motivational, elite, and highly concise.
    """

    # GENERATION: Hit the Google Cloud API
    print("⚡ Beaming context to Gemini Cloud Engine...")
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Gemini API Error: {e}"

# ==================================================
# LIVE TEST RUN
# ==================================================
if __name__ == "__main__":
    # Simulate a heavy valgus error caught by your Streamlit app
    mock_streamlit_telemetry = (
        "Client: John Doe. Exercise: Back Squat. "
        "Metrics: Max Flexion 96.6 degrees. Trunk lean 9.2 degrees. "
        "CRITICAL FLAG: Knee Valgus Risk is at 53.4% during maximum depth under a 2.91x BW kinetic load."
    )
    
    ai_plan = generate_ai_coaching_plan(mock_streamlit_telemetry)
    
    print("\n==================================================")
    print("🚀 GENERATED COACHING PLAN FROM GEMINI CLOUD:")
    print("==================================================")
    print(ai_plan)