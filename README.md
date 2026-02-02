# AI-First CRM: HCP Log Interaction Module

## 👋 About This Project
In this project I focused on solving a major pain point for field representatives: **data entry**. Instead of forcing users to fill out endless dropdowns, I designed a "Log Interaction Screen" that allows them to just type (or paste) their notes naturally.

The core of this project is an **AI Agent (built with LangGraph)** that takes that natural language and automatically fills out the structured database fields.




https://github.com/user-attachments/assets/8891133b-77d3-4da8-b7e7-a7b6a183ab06



Full Demo Video: https://github.com/galaxyhub9/AGENTIC_AI_CRM/releases/tag/demo

## 🛠️ Tech Stack & Key Decisions
* **Frontend:** React + Redux (for managing the complex state of the form).
* **Backend:** Python (FastAPI).
* **AI Engine:** LangGraph for the agent workflow.
* **LLM:** Groq API (using the `llama-3.3-70b-versatile` model for fast inference).
* **Database:** PostgreSQL.

## 🚀 Key Features
**1. The "Hybrid" Interface**
I realized that sometimes reps want to type quickly, and sometimes they need to edit manually. So, the UI supports both:
* **Chat Mode:** "Met Dr. Smith, he loved Product X." (AI processes this).
* **Form Mode:** Standard manual entry.
* *Note:* Both stay in sync—if the AI fills the form, the user can still tweak it manually.

**2. The LangGraph Agent**
Instead of just a simple API call, I used LangGraph to create a stateful agent. It doesn't just "guess"; it uses specific **Tools** to ensure the data is accurate before saving it.

## 🤖 LangGraph Agent Tools
The agent utilizes the following 5 custom tools (defined in `backend/agent.py`):

1. **`log_interaction`**
   - **Purpose:** The core logger. Logs a NEW interaction into the database.
   - **Triggers:** "Log meeting", "Record this", "I met Dr. Smith".
   - **Data Captured:** Name, Type, Date, Topics, Materials, Sentiment, Outcomes, Follow-up.

2. **`edit_interaction`**
   - **Purpose:** The fixer. Modifies the *last* logged interaction if the user made a mistake.
   - **Triggers:** "Change the date", "Update the outcome", "Correct that".

3. **`get_interaction_history`**
   - **Purpose:** Memory. Retrieves the last 3 interactions for a specific HCP to help the rep prepare.
   - **Triggers:** "What did we discuss last time?", "History with Dr. Smith?".

4. **`get_hcp_profile`**
   - **Purpose:** Context. Fetches the doctor's static bio, specialty, and best visiting time.
   - **Triggers:** "Who is he?", "When should I visit?", "Show profile".

5. **`check_sample_stock`**
   - **Purpose:** Inventory Expert. Checks the database for available product samples.
   - **Triggers:** "Do I have samples?", "How many boxes of Product X?".
## ⚙️ How to Run Locally

**Backend**
```bash
cd crm-backend
# I used venv for isolation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# IMPORTANT: Create a .env file with your GROQ_API_KEY
uvicorn main:app --reload
```

**Frontend**
```bash
cd crm-frontend

# Install dependencies (React, Redux, Axios, etc.)
npm install

# Start the development server
npm start
```
