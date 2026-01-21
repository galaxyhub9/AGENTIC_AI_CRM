from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mysql.connector
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from typing import Optional
from langchain_core.messages import SystemMessage

app = FastAPI()

# Enable CORS so React can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Connection ---
def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root", # <--- UPDATE THIS
        database="hcp_crm"
    )

# --- LangGraph Tools [cite: 60, 62] ---

@tool
def log_interaction(
    hcp_name: str, 
    type: str, 
    date: str, 
    topics: Optional[str] = "None", 
    sentiment: Optional[str] = "None", 
    outcomes: Optional[str] = "None"
):
    """
    Log a NEW interaction. 
    - ONLY fill fields the user explicitly mentioned. 
    - If the user didn't mention a field (like topics or outcomes), pass "None".
    - DO NOT guess values.
    """
    try:
        db = get_db()
        cursor = db.cursor()
        sql = """INSERT INTO interactions 
                 (hcp_name, interaction_type, interaction_date, topics_discussed, sentiment, outcomes) 
                 VALUES (%s, %s, %s, %s, %s, %s)"""
        val = (hcp_name, type, date, topics, sentiment, outcomes)
        cursor.execute(sql, val)
        db.commit()
        return "✅ Interaction logged successfully."
    except Exception as e:
        return f"❌ Error logging interaction: {str(e)}"

@tool
def search_hcp(name_query: str):
    """Search for an HCP in the directory by name to verify they exist."""
    db = get_db()
    cursor = db.cursor()
    # Simple search query
    cursor.execute(f"SELECT name, specialty, hospital FROM hcp_directory WHERE name LIKE '%{name_query}%'")
    result = cursor.fetchall()
    return str(result) if result else "No HCP found."

@tool
def edit_interaction(
    hcp_name: Optional[str] = None, 
    type: Optional[str] = None, 
    date: Optional[str] = None, 
    topics: Optional[str] = None, 
    sentiment: Optional[str] = None, 
    outcomes: Optional[str] = None
):
    """
    EDIT or UPDATE the LAST logged interaction.
    - Use this when the user says "change", "correct", "update", or "it was actually...".
    - ONLY pass the arguments that need changing. Leave others as None.
    """
    try:
        db = get_db()
        cursor = db.cursor()
        
        updates = []
        values = []
        
        # Only add fields that are NOT None
        if hcp_name and hcp_name != "None":
            updates.append("hcp_name = %s")
            values.append(hcp_name)
        if type and type != "None":
            updates.append("interaction_type = %s")
            values.append(type)
        if date and date != "None":
            updates.append("interaction_date = %s")
            values.append(date)
        if topics and topics != "None":
            updates.append("topics_discussed = %s")
            values.append(topics)
        if sentiment and sentiment != "None":
            updates.append("sentiment = %s")
            values.append(sentiment)
        if outcomes and outcomes != "None":
            updates.append("outcomes = %s")
            values.append(outcomes)
            
        if not updates:
            return "No changes requested."

        # Update the specific row
        sql = f"UPDATE interactions SET {', '.join(updates)} ORDER BY id DESC LIMIT 1"
        cursor.execute(sql, tuple(values))
        db.commit()
        
        return "✅ Interaction updated in database."
    except Exception as e:
        return f"❌ Error updating: {str(e)}"

@tool
def check_compliance(text: str):
    """
    UNIQUE TOOL: Scans interaction notes for risky/off-label keywords. 
    Use this before logging if the user discusses drug efficacy.
    """
    risky_keywords = ["guarantee", "cure", "miracle", "off-label"]
    flagged = [word for word in risky_keywords if word in text.lower()]
    if flagged:
        return f"⚠️ COMPLIANCE WARNING: Found restricted terms: {flagged}. Please rephrase."
    return "✅ Compliance Check Passed."

# --- Agent Setup ---
# Using llama-3.3-70b-versatile model as required [cite: 16]
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key="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") 

# List of tools the agent can use
tools = [log_interaction, search_hcp, edit_interaction, check_compliance]


agent_executor = create_react_agent(llm, tools)

# --- Define System Prompt String ---
SYSTEM_PROMPT = (
    "You are a precise data entry assistant. Follow these rules STRICTLY:\n"
    "1. NEW vs EDIT: If the user says 'log this', use 'log_interaction'. "
    "If the user says 'change', 'update', 'correct', or 'it was actually', use 'edit_interaction'.\n"
    "2. NO GUESSING: When logging, if the user did not say the interaction type, log it as 'None'. "
    "If they didn't mention outcomes, log 'None'.\n"
    "3. DATES: Always convert 'today' to YYYY-MM-DD format.\n"
    "4. ARGUMENTS: When using 'edit_interaction', ONLY pass the specific field to be changed."
)

# --- API Endpoint ---
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    # FIX 2: Inject the System Prompt manually into the messages list
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        ("user", req.message)
    ]
    
    result = agent_executor.invoke({"messages": messages})
    last_message = result["messages"][-1].content
    
    # Extract Tool Data (to update UI)
    extracted_data = None
    for msg in result["messages"]:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool in msg.tool_calls:
                if tool['name'] in ['log_interaction', 'edit_interaction']:
                    extracted_data = tool['args']

    return {
        "response": last_message, 
        "form_data": extracted_data 
    }
# Run with: uvicorn main:app --reload