import os
import chromadb
from chromadb.utils import embedding_functions

print("🧠 Waking up the Vector Database...")

# Connect to the local ChromaDB folder
db_path = os.path.join(os.getcwd(), "chroma_db")
client = chromadb.PersistentClient(path=db_path)

# Initialize the embedding engine
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Grab our specific collection
collection = client.get_or_create_collection(name="pt_recovery_protocols", embedding_function=ef)

# ==========================================
# 📚 THE ULTIMATE CLINICAL PROTOCOL LIBRARY
# ==========================================
protocols = [
    # --- SQUAT ---
    {
        "id": "squat_valgus",
        "title": "High Valgus Collapse Protocol",
        "text": "Condition: High Knee Valgus Risk detected during squat. Intervention: 1. Banded Glute Bridges (3x15). 2. Lateral Monster Walks. 3. Tempo Goblet Squats with a mini-band forcing external rotation."
    },
    {
        "id": "squat_depth",
        "title": "Limited Squat Depth Protocol",
        "text": "Condition: Severe limitation in knee flexion during squatting. Intervention: 1. Box squats to progressive depths. 2. Banded ankle dorsiflexion mobilization. 3. Heel-elevated squats to bypass ankle restrictions."
    },
    
    # --- OVERHEAD PRESS ---
    {
        "id": "press_asymmetry",
        "title": "Overhead Press Asymmetry Protocol",
        "text": "Condition: Significant left/right asymmetry during overhead lockout. Intervention: 1. Thoracic Spine Extension mobilization. 2. Unilateral latissimus dorsi release. 3. Tall-kneeling unilateral overhead kettlebell holds."
    },
    
    # --- DEADLIFT ---
    {
        "id": "deadlift_lumbar_flexion",
        "title": "Dangerous Lumbar Flexion (Rounding) Protocol",
        "text": "Condition: Excessive trunk lean indicating dangerous lower back rounding. Intervention: 1. PVC Pipe Hip Hinges to groove neutral spine. 2. Isometric Core Bracing (Bird-Dogs/Pallof Presses). 3. Romanian Deadlifts (RDLs) with slow eccentric focus."
    },

    # --- RUNNING GAIT ---
    {
        "id": "run_pelvic_drop",
        "title": "Trendelenburg (Pelvic Drop) Protocol",
        "text": "Condition: Significant pelvic drop during the running stance phase, indicating severe gluteus medius weakness. Intervention: 1. Side-lying hip abductions. 2. Single-leg stance variations (eyes closed, uneven surface) to build proprioception. 3. Weighted step-ups focusing on keeping the pelvis perfectly level."
    },

    # --- JUMP LANDING ---
    {
        "id": "jump_stiff_landing",
        "title": "Poor Shock Absorption (Stiff Landing) Protocol",
        "text": "Condition: Knee flexion angle near 180 degrees upon landing, transferring massive force to the passive joints and risking ACL injury. Intervention: 1. Altitude drops from a low box focusing on 'ninja' (silent) landings. 2. Snap-downs to groove the rapid hip hinge pattern. 3. Depth jumps emphasizing immediate rebound and deep joint absorption."
    },

    # --- LUNGE ---
    {
        "id": "lunge_instability",
        "title": "Unilateral Valgus Instability Protocol",
        "text": "Condition: Knee valgus or severe shaking during unilateral lunge loading. Intervention: 1. Bulgarian split squats with a light contralateral weight to force core engagement. 2. Banded lateral toe taps in a quarter squat. 3. Slow eccentric reverse lunges focusing on perfect knee-over-toe tracking."
    }
]

# Extract the data into ChromaDB format
ids = [p["id"] for p in protocols]
documents = [p["text"] for p in protocols]
metadatas = [{"title": p["title"]} for p in protocols]

print("📥 Injecting ultimate clinical knowledge...")

# Upsert adds new protocols and updates existing ones
collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

print(f"✅ Successfully embedded {len(protocols)} protocols into the RAG memory!")