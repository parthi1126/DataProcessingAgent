<div align="center">

```
██████╗  █████╗ ████████╗ █████╗      █████╗  ██████╗ ███████╗███╗   ██╗████████╗
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
██║  ██║███████║   ██║   ███████║    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   
██║  ██║██╔══██║   ██║   ██╔══██║    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   
██████╔╝██║  ██║   ██║   ██║  ██║    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   
```

```
  ██████╗ ██████╗  ██████╗  ██████╗███████╗███████╗███████╗██╗███╗   ██╗ ██████╗ 
  ██╔══██╗██╔══██╗██╔═══██╗██╔════╝██╔════╝██╔════╝██╔════╝██║████╗  ██║██╔════╝ 
  ██████╔╝██████╔╝██║   ██║██║     █████╗  ███████╗███████╗██║██╔██╗ ██║██║  ███╗
  ██╔═══╝ ██╔══██╗██║   ██║██║     ██╔══╝  ╚════██║╚════██║██║██║╚██╗██║██║   ██║
  ██║     ██║  ██║╚██████╔╝╚██████╗███████╗███████║███████║██║██║ ╚████║╚██████╔╝
  ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚══════╝╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝ 
```

### `[ PROJECT ] ──► Data Preprocessing Agent`
### `[ MODEL ] ──► Gemini 2.5 Pro`
### `[ FRAMEWORK ] ──► LangGraph Stateful Agent`
### `[ BACKEND ] ──► LangChain · Pydantic · Pandas · SQLite`
### `[ STATUS ] ──► AGENT ONLINE ◆ SELF-VALIDATING ◆ AUTONOMOUS`

---

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agent-FF6B35?style=for-the-badge&logo=langchain&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini_2.5_Pro-LLM_Brain-8E75B2?style=for-the-badge&logo=google&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Orchestration-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Engine-150458?style=for-the-badge&logo=pandas&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Checkpoint_Store-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-Schema_Validation-E92063?style=for-the-badge&logo=pydantic&logoColor=white)

</div>

---

## `◈ WHAT IS THIS?`

**DataProcessingAgent** is a **fully autonomous, self-correcting AI agent** that cleans and transforms raw datasets — without any manual intervention. Drop in a CSV/Excel/JSON file, describe your goal in plain English, and the agent:

1. **Analyses** the data statistically — missing values, outliers, cardinality, correlations
2. **Plans** a cleaning pipeline using Gemini 2.5 Pro as the reasoning brain
3. **Executes** Python code in a sandboxed in-memory environment
4. **Validates** its own output against your original goal
5. **Self-corrects** — if validation fails or code crashes, it re-plans and retries (up to 3×)

> The agent holds the entire dataframe **serialized as a JSON string inside its state graph** — no files on disk, no intermediate CSVs, fully stateful and resumable via SQLite checkpointing.

---

## `◈ SYSTEM ARCHITECTURE`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LANGGRAPH STATEFUL AGENT                              │
│                                                                             │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│   │              │    │              │    │              │                  │
│   │  DATA        │───►│  PLAN        │───►│  CODE        │                 │
│   │  PREVIEW     │    │  GENERATION  │    │  EXECUTION   │                 │
│   │  NODE 🔬     │    │  NODE 🧠     │    │  NODE 💻     │                 │
│   │              │    │              │    │              │                  │
│   └──────────────┘    └──────┬───────┘    └──────┬───────┘                 │
│                              │                   │                         │
│                       ┌──────▼──────┐     ┌──────▼──────┐                 │
│                       │  Error?     │     │  Error?     │                  │
│                       │  ◄ RE-PLAN  │     │  ◄ RE-PLAN  │                 │
│                       │  (retry ×3) │     │  else ──►   │                 │
│                       └─────────────┘     └──────┬───────┘                 │
│                                                  │                         │
│                                         ┌────────▼────────┐                │
│                                         │                 │                 │
│                                         │  VALIDATION     │                 │
│                                         │  NODE 🧐        │                 │
│                                         │  (AI Auditor)   │                 │
│                                         │                 │                 │
│                                         └────────┬────────┘                │
│                                                  │                         │
│                                         ┌────────▼────────┐                │
│                                         │  Satisfied?     │                 │
│                                         │  YES ──► END ✅  │                │
│                                         │  NO  ──► RE-PLAN│                │
│                                         └─────────────────┘                │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    AgentState (TypedDict)                           │  │
│   │  file_path · df_json · user_prompt · analysis_report               │  │
│   │  plan · generated_code · execution_log · error                     │  │
│   │  validation_feedback · retries · max_retries                       │  │
│   └────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│                    SQLite Checkpointer (persistent memory)                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## `◈ FILE STRUCTURE`

```
DataProcessingAgent/
├── agent.py          ← 🧠 LangGraph agent: nodes, routing, graph assembly, runner
├── utils.py          ← 🔬 Data analysis library: ingest, analyze, report
└── allinone.py       ← 🧪 Self-contained test runner (all functions + 5-test suite)
```

---

## `◈ AGENT NODES — DEEP DIVE`

### `🔬 NODE 1 — data_preview_node`
**File:** `agent.py → data_preview_node()`

The agent's **sensory input layer**. Calls `utils.combine_analysis_outputs()` which runs the full 9-section analysis pipeline on the raw file. The resulting DataFrame is immediately serialized into a JSON string (`orient='split'`) and stored in `AgentState.df_json` — this is the only time the file is touched. From here, all processing is purely in-memory.

```python
df_json = df.to_json(orient='split')   # DataFrame → JSON string → lives in state
return {
    "analysis_report": result["quality_report"],
    "df_json": df_json,
    "retries": 0,
    "max_retries": 3,
}
```

**State keys written:** `analysis_report`, `df_json`, `retries`, `max_retries`, `execution_log`, `error`

---

### `🧠 NODE 2 — plan_generation_node`
**File:** `agent.py → plan_generation_node()`

The **reasoning brain** of the agent. Uses `gemini-2.5-pro` with `structured_output` (Pydantic `PlanAndCode` schema) to guarantee the LLM always returns both a plan and executable Python code — never freeform text.

**Three modes of operation:**
- **Fresh run** — plans from the analysis report + user goal
- **Code error recovery** — receives the traceback + failed code; re-plans a fix
- **Validation feedback** — receives the AI auditor's critique; re-plans logically

```python
class PlanAndCode(BaseModel):
    plan: str         # Bullet-point cleaning strategy
    python_code: str  # Executable code block operating on `df`

structured_llm = llm.with_structured_output(PlanAndCode)
```

**Critical constraint injected into every system prompt:**
```
Before using .str accessor on any column, MUST convert to string and fill NaNs:
  CORRECT:   df['col'] = df['col'].astype(str).fillna('Unknown').str.strip()
  INCORRECT: df['col'] = df['col'].str.strip()   # Crashes on mixed-type columns
```

**State keys written:** `plan`, `generated_code`, `error` (cleared), `validation_feedback` (cleared)

---

### `💻 NODE 3 — code_execution_node`
**File:** `agent.py → code_execution_node()`

The **sandboxed execution engine**. Deserializes the DataFrame from JSON state, runs the LLM-generated code via `exec()` in an isolated local scope, captures stdout, and re-serializes the result back into state — no file I/O ever touches disk.

```python
# Sandboxed execution scope — only these names are visible to the LLM's code
local_scope = {
    "df": df_to_process,
    "pd": pd, "np": np, "re": re,
    "utils": utils,
    "parse_indian_number": utils.parse_indian_number
}
exec(code, {}, local_scope)        # Isolated: {} = empty globals
processed_df = local_scope['df']   # Retrieve mutated df
new_df_json = processed_df.to_json(orient='split')  # Back to state
```

On **success:** updates `df_json`, appends `SUCCESS` to `execution_log`, clears `error`
On **failure:** appends full traceback to `execution_log`, sets `error`, increments `retries`

---

### `🧐 NODE 4 — validation_node`
**File:** `agent.py → validation_node()`

The **AI Quality Auditor** — a second Gemini invocation acting as a completely separate reviewer. It receives a concise preview of the *processed* DataFrame (shape, dtypes, nulls, first 2 rows) and the original user goal, then returns a structured `ValidationResult`:

```python
class ValidationResult(BaseModel):
    satisfied: bool              # True = done, False = re-plan
    feedback: Optional[str]      # Actionable critique for the planner

validation_llm = llm.with_structured_output(ValidationResult)
```

If `satisfied=False`, the feedback string is written to `AgentState.validation_feedback` and the router sends control back to `plan_generation_node` — which treats fixing the logical error as the highest priority.

---

## `◈ ROUTING LOGIC`

```python
# START → decides entry point
route_initial(state) → "data_preview" | "generate_plan"
    # Manual feedback injected (human-in-loop) → generate_plan
    # Fresh file input                          → data_preview

# After code execution → decides next step
route_after_execution(state) → "generate_plan" | "validation_node" | END
    # Code error + retries remaining  → generate_plan  (retry)
    # Code error + max retries hit    → END            (give up)
    # Code success                    → validation_node

# After validation → decides final outcome
route_after_validation(state) → "generate_plan" | END
    # validation_feedback exists      → generate_plan  (logical fix)
    # No feedback                     → END ✅          (success)
```

---

## `◈ UTILS.PY — ANALYSIS ENGINE`

`utils.py` is the agent's **data intelligence layer** — a standalone library of 5 functions that power Node 1.

### `ingest_data(file_path)` — Smart File Loader

Supports CSV, Excel, JSON. Applies **domain-aware type coercion** during load:
- Columns named `*price*` / `*cost*` → `parse_indian_number()` (handles ₹, Lakh, Cr, K)
- Columns named `*sqft*` / `*square*` / `*feet*` → `pd.to_numeric(..., errors='coerce')`
- Columns named `*date*` → `pd.to_datetime(..., errors='coerce')`
- Rejects files > 50MB with a framework recommendation (Dask / PySpark / BigQuery)

### `parse_indian_number(text)` — Currency Parser

Converts messy Indian financial strings to clean floats:
```
"₹1.2 Lakh"      →   120000.0
"10Cr"            →   100000000.0
"5,000"           →   5000.0
"1500 per sqft"   →   1500.0   (strips "per sqft" before parsing)
"N/A"             →   NaN
```

### `analyze_data(df)` — 9-Section Statistical Report

| Section | What it computes |
|---|---|
| **A. General** | Shape, dtypes, memory usage MB, file size, recommended framework |
| **B. Missing Values** | Count + % per column, completely null columns, total missing |
| **C. Duplicates** | Row-level duplicate count + %, column-pair exact duplicates |
| **D. Numeric Summary** | describe(), IQR, skewness, kurtosis per numeric column |
| **E. Categorical Summary** | Unique count, high-cardinality flag (> 10% of rows), top-5 values |
| **F. Correlation** | Full Pearson matrix, high-corr pairs (>0.8 threshold), recommendations |
| **G. Outliers** | IQR method (1.5×IQR rule): count, %, lower/upper bounds per column |
| **H. Domain Flags** | Potential ID cols, date cols, monetary cols, high-missing cols (>50%) |
| **I. Visual Insights** | Text distribution summaries for top 3 numeric and categorical cols |

### `generate_quality_report(preview, analysis)` — LLM Narrative

Sends the full analysis dict to Gemini and gets back a **300–500 word professional data quality report** structured as: Overview → Key Issues → Insights & Recommendations → Next Steps. Gracefully falls back to a deterministic rule-based mock report if the LLM is unavailable.

### `combine_analysis_outputs(file_path)` — Master Orchestrator

Single call that chains all of the above. Returns one result dict with `ingestion`, `summary`, `analysis`, `quality_report`, `sample_rows`, and the loaded `dataframe` object — which `data_preview_node` serializes into agent state.

---

## `◈ AGENT STATE — FULL SCHEMA`

```python
class AgentState(TypedDict, total=False):
    file_path:           str            # Path to the input dataset
    df_json:             str            # DataFrame as JSON string (orient='split')
    user_prompt:         str            # Natural language cleaning goal
    analysis_report:     str            # LLM quality report from utils
    plan:                str            # Bullet-point cleaning plan from planner node
    generated_code:      str            # Python code block to exec
    execution_log:       List[str]      # Append-only log of every exec attempt
    error:               Optional[str]  # Last execution error (None = clean)
    feedback:            Optional[str]  # Manual human feedback (human-in-loop hook)
    validation_feedback: Optional[str]  # AI auditor critique (triggers re-plan)
    retries:             int            # Current retry counter
    max_retries:         int            # Hard ceiling — default 3
```

---

## `◈ CHECKPOINTING — PERSISTENT MEMORY`

```python
with SqliteSaver.from_conn_string("checkpoints.sqlite") as saver:
    app = build_graph(saver)
    app.stream(inputs, config={"configurable": {"thread_id": "run-001"}})
```

Every node transition writes a versioned snapshot to `checkpoints.sqlite`. This enables:

- **Resume interrupted runs** — restart with the same `thread_id` and pick up exactly where the agent stopped
- **Human-in-the-loop** — pause after any node, inject `feedback` into state, re-invoke to course-correct
- **Full audit trail** — every state version is timestamped, versioned, and queryable

---

## `◈ SELF-CORRECTION LOOP — TRACE EXAMPLE`

```
Attempt 1:  LLM generates code → exec() crashes
            error = "AttributeError: 'float' object has no attribute 'str'"
            retries = 1
            → route_after_execution → "generate_plan"

Attempt 2:  LLM receives error + failed code → regenerates with .astype(str) guard
            exec() succeeds → route_after_execution → "validation_node"

Validation: AI auditor checks processed df vs user goal
            "The 'gross' column was dropped but the user needed it cleaned"
            validation_feedback = "Retain and clean the 'gross' column"
            → route_after_validation → "generate_plan"

Attempt 3:  LLM receives validation critique → regenerates with 'gross' retained
            exec() succeeds → validation_node → satisfied = True → END ✅
```

---

## `◈ DEPENDENCY MAP`

| Package | Role |
|---|---|
| `langchain-google-genai` | Gemini 2.5 Pro interface — the reasoning engine |
| `langgraph` | Stateful graph framework — nodes, edges, conditional routing |
| `langchain-core` | `HumanMessage`, `SystemMessage` primitives |
| `pydantic` | Structured output schemas (`PlanAndCode`, `ValidationResult`) |
| `langgraph[checkpoint-sqlite]` | SQLite state persistence via `SqliteSaver` |
| `pandas` | DataFrame operations — load, transform, serialize |
| `numpy` | Numerical operations inside LLM-generated cleaning code |
| `scipy` | Statistical functions (IQR, kurtosis, skewness) in analysis |
| `python-dotenv` | Loads `GOOGLE_API_KEY` from `.env` |
| `matplotlib` / `seaborn` | Visualization support (available in `allinone.py`) |

---

## `◈ SETUP & RUN`

### 1. Clone & Install

```bash
git clone https://github.com/parthi1126/DataProcessingAgent.git
cd DataProcessingAgent

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install langchain-google-genai langgraph langchain-core pydantic \
            pandas numpy scipy python-dotenv "langgraph[checkpoint-sqlite]"
```

### 2. Configure API Key

```bash
echo "GOOGLE_API_KEY=your_gemini_api_key_here" > .env
```

Get your key at: [Google AI Studio](https://aistudio.google.com)

### 3. Point at Your Dataset

Edit the `__main__` block in `agent.py`:

```python
FILE_PATH   = r"path/to/your/dataset.csv"   # CSV, Excel, or JSON — max 50MB
USER_PROMPT = """
    Clean this dataset: fix missing values, remove duplicates,
    correct data types, and flag outliers.
"""
THREAD_ID = "my-run-001"   # Reuse same ID to resume a previous run
```

### 4. Run the Agent

```bash
python agent.py
```

The agent streams each node's state updates live to the console. On completion, it prompts you to save the final cleaned DataFrame as `final_output.csv`.

### 5. Run the Test Suite (standalone)

```bash
# Edit file_path at the bottom of allinone.py, then:
python allinone.py
```

Runs 5 automated end-to-end tests: ingest → summary → analyze → LLM report → combine.

---

## `◈ allinone.py — TEST SUITE`

| Test | What it validates |
|---|---|
| **Test 1** | `ingest_data()` — file load, smart type coercion, Indian number parsing, shape |
| **Test 2** | `generate_summary_report()` — natural language overview string |
| **Test 3** | `analyze_data()` — all 9 analysis sections complete and error-free |
| **Test 4** | `generate_quality_report()` — Gemini narrative report (or mock fallback) |
| **Test 5** | `combine_analysis_outputs()` — full pipeline integration test |

---

## `◈ AUTHOR`

```
  ┌──────────────────────────────────────────────────────┐
  │  Parthiban R                                         │
  │  Final Year B.Tech — AI & Data Science               │
  │  Chennai Institute of Technology
  │                                                      │
  │  Project: Autonomous Data Preprocessing Agent        │
  │  Stack:   LangGraph · Gemini 2.5 Pro · LangChain     │
  │           Pydantic · Pandas · SQLite Checkpointing   │
  └──────────────────────────────────────────────────────┘
```

---

<div align="center">

```
  ┌────────────────────────────────────────────────────────────────┐
  │                                                                │
  │   AGENT LIFECYCLE                                              │
  │                                                                │
  │   INPUT ──► ANALYSE ──► PLAN ──► EXECUTE ──► VALIDATE         │
  │                           ▲           │            │           │
  │                           │  (error)  │            │ (fail)    │
  │                           └───────────┘            │           │
  │                           ▲                        │           │
  │                           └────────────────────────┘           │
  │                                                                │
  │                        ──► CLEAN DATA ✅                       │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
```

**An AI that audits its own work. A cleaner that never needs a human to check it.**

*Built with LangGraph · Powered by Gemini 2.5 Pro · Stateful by design*

</div>
