# agent.py
# GenAI Data Processing & Automation Agent
# VERSION: Stateful (with JSON Fix) + Validation Loop
# This agent holds the dataframe as a JSON string in its state,
# processes it, and then validates the result against the user's goal.

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
    model="gemini-2.5-pro",  # <--- Corrected to the model we know works
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
    df_json: str  # We store the df as a JSON string
    user_prompt: str
    analysis_report: str
    plan: str
    generated_code: str
    execution_log: List[str]
    error: Optional[str]
    feedback: Optional[str]  # Manual feedback from a human user
    validation_feedback: Optional[str]  # <--- NEW: Feedback from the AI's self-validation
    retries: int
    max_retries: int

# ----------------------------------------------------------------------
# 4. LLM Output Schemas
# ----------------------------------------------------------------------
class PlanAndCode(BaseModel):
    """
    Forces the LLM to return a plan and the code to execute.
    """
    plan: str = Field(description="A concise, bullet-point plan of the data processing steps to be taken.")
    python_code: str = Field(description="A single, executable Python code block to transform the pandas DataFrame named 'df'.")

# This new schema is for our validation node
class ValidationResult(BaseModel):
    """
    The result of the AI's self-validation step.
    """
    satisfied: bool = Field(description="Boolean flag, True if the dataframe meets all user requirements, False otherwise.")
    feedback: Optional[str] = Field(description="If not satisfied, a concise explanation of what is wrong or missing. This will be used to re-plan.")

# We now have two structured LLM callers
structured_llm = llm.with_structured_output(PlanAndCode)
validation_llm = llm.with_structured_output(ValidationResult)


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
        result = utils.combine_analysis_outputs(state["file_path"])
        if "error" in result:
            raise ValueError(result["error"])
        
        df = result["dataframe"]
        df_json = df.to_json(orient='split')
        
        print(f"✓ Analysis complete. Report generated. DF loaded and serialized (Shape: {df.shape})")
        return {
            "analysis_report": result["quality_report"],
            "df_json": df_json,
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
    Node 2: The "Brain". Now handles code errors AND validation feedback.
    """
    print("\n--- 🧠 NODE: Plan Generation ---")
    
    # --- UPDATED SYSTEM PROMPT ---
    system = """
You are an expert Data Engineer AI Agent. Your job is to write a Python script 
to clean and process a dataset based on a user's request, a data analysis report,
and (if provided) critical feedback on your previous attempt.

**Rules:**
1.  You MUST generate a JSON object with "plan" and "python_code" keys.
2.  The Python code will be executed in an environment where the dataframe 
    is already loaded into a variable named `df`.
3.  Your code MUST NOT include data loading (e.g., `pd.read_csv`) or data 
    saving (e.g., `df.to_csv`).
4.  Import any libraries you need (e.g., `import numpy as np`).
5.  Always be safe: use `df.get('col_name')`, `pd.to_numeric(..., errors='coerce')`.
6.  If you are correcting an error or validation feedback, prioritize fixing it.
7.  **CRITICAL RULE:** Before using any `.str` accessor on a column, you
    MUST first convert it to string and fill NaNs to prevent 'AttributeError'.
    **Correct:** `df['col_name'] = df['col_name'].astype(str).fillna('Unknown').str.strip()`
    **Incorrect:** `df['col_name'] = df['col_name'].str.strip()`
"""
    context = f"**Data Analysis Report:**\n{state.get('analysis_report', 'N/A')}\n\n**User's Goal:**\n{state['user_prompt']}"

    # --- This logic is correct ---
    if state.get("error"):
        print("   ...addressing previous EXECUTION ERROR")
        log = state.get("execution_log", [])
        context += f"\n\n**CRITICAL: The previous code FAILED TO RUN.**\n**Error:**\n{state['error']}\n**Failed Code:**\n{state.get('generated_code', 'N/A')}\n**Execution Log:**\n{log[-1] if log else 'N/A'}\n\nPlease generate a new plan and corrected python_code to fix this error."
    
    elif state.get("validation_feedback"):
        print("   ...addressing previous VALIDATION FEEDBACK")
        context += f"\n\n**CRITICAL: The previous run was LOGICALLY INCORRECT.**\n**Validation Feedback:**\n{state['validation_feedback']}\n\nPlease generate a new plan and code to fix this logical error. This is a high-priority correction."

    try:
        resp = structured_llm.invoke([SystemMessage(content=system), HumanMessage(content=context)])
        print(f"✓ Plan generated (Step 1: {resp.plan.splitlines()[0]}...)")
        return { 
            "plan": resp.plan, 
            "generated_code": resp.python_code, 
            "error": None,
            "validation_feedback": None # Clear old feedback
        }
    except Exception as e:
        print(f"✗ Error in Plan Node (LLM failed): {e}")
        return {"error": f"LLM failed: {e}"}


def code_execution_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 3: The "Hands". Deserializes, executes code, serializes.
    (This node is unchanged)
    """
    print("\n--- 💻 NODE: Code Execution (In-Memory w/ JSON) ---")
    code = state.get("generated_code")
    df_json = state.get("df_json")
    
    if not code or not df_json:
        return {"error": "Missing generated_code or df_json in state."}
    
    try:
        df_to_process = pd.read_json(io.StringIO(df_json), orient='split')
        print(f"   ...DataFrame deserialized from state. Shape: {df_to_process.shape}")
    except Exception as e:
        return {"error": f"Failed to deserialize DataFrame from state: {e}"}

    local_scope = {
        "df": df_to_process,
        "pd": pd, "np": np, "re": re,
        "utils": utils, "parse_indian_number": utils.parse_indian_number
    }
    
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    log_output = ""
    
    try:
        exec(code, {}, local_scope)
        processed_df = local_scope['df']
        log_output = buffer.getvalue()
        print(f"✓ Code executed successfully. New DF shape: {processed_df.shape}")
        log = f"SUCCESS | Output:\n{log_output}"
        
        new_df_json = processed_df.to_json(orient='split')
        
        return {
            "df_json": new_df_json,
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

# --- NEW NODE ---
def validation_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 4: The "Auditor".
    Compares the processed dataframe against the original user prompt to check if
    all goals were met.
    """
    print("\n--- 🧐 NODE: Validation ---")
    
    df_json = state.get("df_json")
    user_prompt = state.get("user_prompt")
    
    if not df_json or not user_prompt:
        return {"error": "Missing df_json or user_prompt for validation."}

    try:
        df = pd.read_json(io.StringIO(df_json), orient='split')
        # Create a concise preview of the *processed* data
        preview = {
            'shape': df.shape,
            'columns': list(df.columns),
            'head (first 2 rows)': df.head(2).to_dict('records'),
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': {k: str(v) for k, v in df.dtypes.to_dict().items()}, # Make dtypes JSON-safe
        }
        
        system = """
You are a Data Quality Auditor. You will be given a user's original request
and a preview of the *final processed dataframe*.

Your sole job is to determine if the final dataframe successfully meets
all of the user's requirements.

- If YES, return `{"satisfied": true}`.
- If NO, return `{"satisfied": false, "feedback": "..."}`.
  The feedback MUST be a concise, actionable instruction for the
  Data Engineer on what to fix. (e.g., "The 'gross' column was dropped,
  but the user needed it. Please re-run and ensure 'gross' is retained
  and cleaned.")
"""
        context = f"""
**Original User Goal:**
{user_prompt}

**Preview of Final Processed Data:**
{json.dumps(preview, indent=2)}

Does this data satisfy all requirements of the original user goal?
"""
        resp = validation_llm.invoke([SystemMessage(content=system), HumanMessage(content=context)])
        
        if resp.satisfied:
            print("✓ Validation PASSED.")
            return {"validation_feedback": None} # No feedback, we are done
        else:
            print(f"✗ Validation FAILED. Feedback: {resp.feedback}")
            return {"validation_feedback": resp.feedback}
            
    except Exception as e:
        print(f"✗ Error in Validation Node: {e}")
        return {"error": f"Validation failed: {e}"}


# ----------------------------------------------------------------------
# 6. Graph Routing
# ----------------------------------------------------------------------
def route_initial(state: AgentState) -> str: 
    if state.get("feedback"):
        print("--- 🚦 ROUTE: Manual Feedback Loop ---")
        return "generate_plan"
    if not state.get("analysis_report"):
        print("--- 🚦 ROUTE: New Job ---")
        return "data_preview"
    return "generate_plan"

# --- RENAMED & MODIFIED ---
def route_after_execution(state: AgentState) -> str:
    """
    Checks for code execution errors.
    If error, retry plan. If success, go to validation.
    """
    r = state.get("retries", 0)
    m = state.get("max_retries", 3)
    
    if state.get("error"):
        if r < m:
            print(f"--- 🚦 ROUTE: Retrying Code Error (Attempt {r+1}/{m}) ---")
            return "generate_plan" # Code error, retry
        else:
            print(f"--- 🚦 ROUTE: Max Retries Reached ---")
            return END # Max retries, give up
            
    print("--- 🚦 ROUTE: Code OK, Proceeding to Validation ---")
    return "validation_node" # Code ran, now validate logic

# --- NEW ROUTER ---
def route_after_validation(state: AgentState) -> str:
    """
    Checks the result of the validation_node.
    If feedback exists, loop back to planner. Otherwise, end.
    """
    if state.get("validation_feedback"):
        # We have logical feedback, re-plan
        print(f"--- 🚦 ROUTE: Validation Failed, Re-planning ---")
        return "generate_plan"
    
    # No feedback, we are done
    print("--- 🚦 ROUTE: Validation Passed, Success! ---")
    return END

# ----------------------------------------------------------------------
# 7. Build the Graph
# ----------------------------------------------------------------------
def build_graph(checkpointer):
    print("--- 🏗️ Building Agent Graph ---")
    g = StateGraph(AgentState)
    
    g.add_node("data_preview", data_preview_node)
    g.add_node("generate_plan", plan_generation_node)
    g.add_node("execute_code", code_execution_node)
    g.add_node("validation_node", validation_node) # <--- ADDED New Node

    g.add_conditional_edges(START, route_initial, {
        "data_preview": "data_preview",
        "generate_plan": "generate_plan",
    })
    g.add_edge("data_preview", "generate_plan")
    g.add_edge("generate_plan", "execute_code")
    
    # --- MODIFIED: Route from execute_code to validation_node ---
    g.add_conditional_edges("execute_code", route_after_execution, {
        "generate_plan": "generate_plan",
        "validation_node": "validation_node", # <--- New path
        END: END,
    })
    
    # --- NEW: Route from validation_node back to planner or END ---
    g.add_conditional_edges("validation_node", route_after_validation, {
        "generate_plan": "generate_plan",
        END: END,
    })
    
    return g.compile(checkpointer=checkpointer)
# ----------------------------------------------------------------------
# 8. Run the Agent
# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    FILE_PATH = r"E:\downloads\archive (19)\movies.csv"
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
    THREAD_ID = "Movie-json-agent-v1" # New ID

    print("\n" + "="*80)
    print("🚀 GENAI STATEFUL DATA AGENT (w/ Validation Loop)")
    print(f"   CWD: {os.getcwd()}")
    print("="*80 + "\n")

    with SqliteSaver.from_conn_string("checkpoints.sqlite") as saver:
        
        app = build_graph(saver)
        
        inputs = {"file_path": FILE_PATH, "user_prompt": USER_PROMPT}
        config = {"configurable": {"thread_id": THREAD_ID}}

        # --- UPDATED STREAMING LOOP FOR BETTER DEBUGGING ---
        for event in app.stream(inputs, config, stream_mode="updates"):
            node = list(event.keys())[0]
            if node == START or node == "__end__":
                continue
            
            print(f"\n--- STATE UPDATE (from node: {node}) ---")
            
            # Get the dictionary of updates from the node
            update_data = event[node]
            
            # Print all keys *except* the big ones
            update_keys = [k for k in update_data.keys() if k not in ['df_json', 'execution_log', 'plan']]
            if update_keys:
                print(f"Keys updated: {update_keys}")

            # --- THIS IS THE CRITICAL DEBUGGING PART ---
            # If an error was just added, print it
            if 'error' in update_data and update_data['error']:
                print(f"\n!!! ERROR DETECTED: {update_data['error']} !!!")
            
            # If the execution_log was just updated (which happens on error or success)
            # print the *last* log entry.
            if 'execution_log' in update_data:
                print("\n--- Execution Log (Last Entry) ---")
                print(update_data['execution_log'][-1])
                print("---------------------------------")
        
        # --- FINAL CHECK (Deserialize from JSON) ---
        print(f"\n" + "="*80)
        print("✅ AGENT RUN COMPLETE. Checking final state...")
        
        final_state = app.get_state(config)
        
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
            # --- ALSO PRINT THE FINAL ERROR LOG IF IT FAILED ---
            if final_state.values.get('error'):
                print(f"   Final Error: {final_state.values.get('error')}")
                print("   See execution log in the console scrollback for details.")
        
        print("="*80)

    print("\n" + "="*80)
    print("RUN COMPLETE")
    print("="*80)