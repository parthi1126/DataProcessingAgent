# agent.py
# GenAI Data Processing & Automation Agent
# VERSION: Stateful "In-Memory" (with JSON State Fix)
# This agent holds the dataframe as a JSON string in its state
# to make it serializable for the checkpointer.

import os
import json
import warnings
import sys
import traceback
import io
from typing import TypedDict, List, Optional, Dict, Any
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import re

# ----------------------------------------------------------------------
# 1. Imports
# ----------------------------------------------------------------------
import utils  # Our debugged analysis library
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.sqlite import SqliteSaver

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 2. LLM
# ----------------------------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY missing in .env")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.0,
    max_output_tokens=8192,
)

# ----------------------------------------------------------------------
# 3. State
# ----------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    """
    This is the agent's memory (the 'state').
    'df_json' will hold our dataframe as a JSON string.
    """
    file_path: str
    df_json: str  # <--- MODIFIED: We store the df as a JSON string
    user_prompt: str
    analysis_report: str
    plan: str
    generated_code: str
    execution_log: List[str]
    error: Optional[str]
    feedback: Optional[str]
    retries: int
    max_retries: int

# ----------------------------------------------------------------------
# 4. LLM Output Schema
# ----------------------------------------------------------------------
class PlanAndCode(BaseModel):
    """
    This class forces the LLM to return JSON in this exact structure.
    """
    plan: str = Field(description="A concise, bullet-point plan of the data processing steps to be taken.")
    python_code: str = Field(description="A single, executable Python code block to transform the pandas DataFrame named 'df'.")

structured_llm = llm.with_structured_output(PlanAndCode)

# ----------------------------------------------------------------------
# 5. Agent Nodes
# ----------------------------------------------------------------------
def data_preview_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 1: Runs utils.py, gets the report AND the dataframe,
    converts the dataframe to a JSON string, and saves both to the state.
    """
    print("\n--- 🔬 NODE: Data Preview ---")
    try:
        # Calls the function from utils.py
        result = utils.combine_analysis_outputs(state["file_path"])
        if "error" in result:
            raise ValueError(result["error"])
        
        # --- MODIFIED: Convert DataFrame to JSON string ---
        df = result["dataframe"]
        df_json = df.to_json(orient='split')
        
        print(f"✓ Analysis complete. Report generated. DF loaded and serialized (Shape: {df.shape})")
        return {
            "analysis_report": result["quality_report"],
            "df_json": df_json,  # <--- MODIFIED: Save the df_json to the state
            "retries": 0,
            "max_retries": 3,
            "execution_log": [],
            "error": None,
        }
    except Exception as e:
        print(f"✗ Error in Preview Node: {e}")
        return {"error": f"Preview failed: {e}"}


def plan_generation_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 2: The "Brain". (This node is unchanged)
    """
    print("\n--- 🧠 NODE: Plan Generation ---")
    
    system = """
You are an expert Data Engineer AI Agent. Your job is to write a Python script 
to clean and process a dataset based on a user's request and a data analysis report.

**Rules:**
1.  You will be given a data analysis report and a user goal.
2.  You MUST generate a JSON object with "plan" and "python_code" keys.
3.  The Python code will be executed in an environment where the dataframe 
    is already loaded into a variable named `df`.
4.  Your code MUST NOT include data loading (e.g., `pd.read_csv`) or data 
    saving (e.g., `df.to_csv`). It must only contain the transformation logic.
5.  The code MUST be a single, executable block. Import any libraries you need
    (e.g., `import numpy as np`, `from sklearn.preprocessing import ...`).
6.  Always be safe: use `df.get('col_name')` instead of `df['col_name']` to avoid
    KeyErrors. Use `pd.to_numeric(..., errors='coerce')` for safe conversion.
7.  You can call functions from the `utils.py` library, like `utils.parse_indian_number()`.
8.  If you are correcting an error, analyze the error message and the failed
    code, then generate a new, corrected `python_code` and `plan`.
"""
    context = f"**Data Analysis Report:**\n{state.get('analysis_report', 'N/A')}\n\n**User's Goal:**\n{state['user_prompt']}"

    if state.get("error"):
        print("   ...addressing previous error")
        log = state.get("execution_log", [])
        context += f"\n\n**CRITICAL: The previous attempt failed.**\n**Error:**\n{state['error']}\n**Failed Code:**\n{state.get('generated_code', 'N/A')}\n**Execution Log:**\n{log[-1] if log else 'N/A'}\n\nPlease generate a new plan and corrected python_code."

    try:
        resp = structured_llm.invoke([SystemMessage(content=system), HumanMessage(content=context)])
        print(f"✓ Plan generated (Step 1: {resp.plan.splitlines()[0]}...)")
        return { "plan": resp.plan, "generated_code": resp.python_code, "error": None }
    except Exception as e:
        print(f"✗ Error in Plan Node (LLM failed): {e}")
        return {"error": f"LLM failed: {e}"}


def code_execution_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 3: The "Hands".
    1. Deserializes the DataFrame from the state's JSON string.
    2. Executes the LLM's code on the in-memory DataFrame.
    3. Serializes the modified DataFrame back into a JSON string for the state.
    """
    print("\n--- 💻 NODE: Code Execution (In-Memory w/ JSON) ---")
    code = state.get("generated_code")
    df_json = state.get("df_json")
    
    if not code or not df_json:
        return {"error": "Missing generated_code or df_json in state."}
    
    # --- MODIFIED: Deserialize DataFrame from JSON string ---
    try:
        df_to_process = pd.read_json(io.StringIO(df_json), orient='split')
        print(f"   ...DataFrame deserialized from state. Shape: {df_to_process.shape}")
    except Exception as e:
        return {"error": f"Failed to deserialize DataFrame from state: {e}"}

    # Create the execution scope ("sandbox")
    local_scope = {
        "df": df_to_process,
        "pd": pd,
        "np": np,
        "re": re,
        "utils": utils,
        "parse_indian_number": utils.parse_indian_number
    }
    
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    log_output = ""
    
    try:
        # --- Execute the LLM's code in the defined scope ---
        exec(code, {}, local_scope)
        
        # Get the processed dataframe back from the scope
        processed_df = local_scope['df']
        log_output = buffer.getvalue()
        
        print(f"✓ Code executed successfully. New DF shape: {processed_df.shape}")
        log = f"SUCCESS | Output:\n{log_output}"
        
        # --- MODIFIED: Serialize the *new* DataFrame back to JSON ---
        new_df_json = processed_df.to_json(orient='split')
        
        return {
            "df_json": new_df_json,  # <--- MODIFIED: Save new JSON to state
            "execution_log": state.get("execution_log", []) + [log],
            "error": None,
        }
    except Exception as e:
        tb = traceback.format_exc()
        log_output = buffer.getvalue()
        log = f"ERROR | Output:\n{log_output}\nTraceback:\n{tb}"
        print(f"✗ Code execution failed.")
        return {
            "execution_log": state.get("execution_log", []) + [log],
            "error": f"{type(e).__name__}: {e}",
            "retries": state.get("retries", 0) + 1,
        }
    finally:
        sys.stdout = old_stdout


# ----------------------------------------------------------------------
# 6. Graph Routing (Unchanged)
# ----------------------------------------------------------------------
def route_initial(state: AgentState) -> str: 
    if state.get("feedback"):
        print("--- 🚦 ROUTE: Feedback Loop ---")
        return "generate_plan"
    if not state.get("analysis_report"):
        print("--- 🚦 ROUTE: New Job ---")
        return "data_preview"
    return "generate_plan"

def should_retry(state: AgentState) -> str:
    r = state.get("retries", 0)
    m = state.get("max_retries", 3)
    
    if state.get("error"):
        if r < m:
            print(f"--- 🚦 ROUTE: Retrying (Attempt {r+1}/{m}) ---")
            return "generate_plan"
        else:
            print(f"--- 🚦 ROUTE: Max Retries Reached ---")
            return END
            
    print("--- 🚦 ROUTE: Success ---")
    return END

# ----------------------------------------------------------------------
# 7. Build the Graph (Unchanged)
# ----------------------------------------------------------------------
def build_graph(checkpointer):
    print("--- 🏗️ Building Agent Graph ---")
    g = StateGraph(AgentState)
    
    g.add_node("data_preview", data_preview_node)
    g.add_node("generate_plan", plan_generation_node)
    g.add_node("execute_code", code_execution_node)

    g.add_conditional_edges(START, route_initial, {
        "data_preview": "data_preview",
        "generate_plan": "generate_plan",
    })
    g.add_edge("data_preview", "generate_plan")
    g.add_edge("generate_plan", "execute_code")
    g.add_conditional_edges("execute_code", should_retry, {
        "generate_plan": "generate_plan",
        END: END,
    })
    
    return g.compile(checkpointer=checkpointer)
# ----------------------------------------------------------------------
# 8. Run the Agent
# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    FILE_PATH = r"E:\downloads\archive (18)\laptopData.csv"
    USER_PROMPT = """
Your task is to autonomously clean and prepare this dataset for general analysis.

I have not provided specific instructions. Instead, I want you to use the **'Data Analysis Report'** that you just generated.

Please use the **'Key Issues'** and **'Recommendations'** sections of that report to create and execute your *own* plan.

Your goals are to:
1.  **Fix all critical issues** you identified in the report.
2.  **Handle missing data** using the methods you recommended (e.g., median, mode, or dropping).
3.  **Remove duplicate rows.**
4.  **Correct data types** (e.g., convert string-based numbers to float/int).
5.  **Perform any feature engineering** that you recommended (like extracting values from text).
6.  **Address any outliers or high-cardinality columns** as you see fit.

In short: **Act as an expert data engineer, analyze your own report, and clean the data based on your findings.**
"""
    THREAD_ID = "laptop-json-agent-v1" # New ID

    print("\n" + "="*80)
    print("🚀 GENAI STATEFUL DATA AGENT (JSON/DB-STATE)")
    print(f"   CWD: {os.getcwd()}")
    print("="*80 + "\n")

    with SqliteSaver.from_conn_string("checkpoints.sqlite") as saver:
        
        app = build_graph(saver)
        
        inputs = {"file_path": FILE_PATH, "user_prompt": USER_PROMPT}
        config = {"configurable": {"thread_id": THREAD_ID}}

        for event in app.stream(inputs, config, stream_mode="updates"):
            node = list(event.keys())[0]
            if node == START or node == "__end__":
                continue
            
            print(f"\n--- STATE UPDATE (from node: {node}) ---")
            
            # Filter out the 'df_json' key so we don't print a huge string
            update_keys = [k for k in event[node].keys() if k != 'df_json']
            print(f"Keys updated: {update_keys}")

        # --- FINAL CHECK (Deserialize from JSON) ---
        print(f"\n" + "="*80)
        print("✅ AGENT RUN COMPLETE. Checking final state...")
        
        final_state = app.get_state(config)
        
        # --- THIS IS THE FIX ---
        # The state is inside the .values attribute of the snapshot
        final_df_json = final_state.values.get('df_json')

        if final_df_json:
            try:
                final_df = pd.read_json(io.StringIO(final_df_json), orient='split')
                print(f"   Final DF Shape: {final_df.shape}")
                print("\n--- Final DataFrame (Head) ---")
                print(final_df.head())
                print("---------------------------------")
                
                if input("\n> Do you want to save this processed data to 'final_output.csv'? (y/n): ").lower() == 'y':
                    final_df.to_csv("final_output.csv", index=False)
                    print(f"✓ Saved to {os.path.abspath('final_output.csv')}")
            except Exception as e:
                print(f"❌ FAIL: Could not deserialize final dataframe from state. Error: {e}")
        else:
            print(f"❌ FAIL: No dataframe (df_json) found in final state.")
            print("   Check the execution_log in the console above for the error.")
        
        print("="*80)

    print("\n" + "="*80)
    print("RUN COMPLETE")
    print("="*80)