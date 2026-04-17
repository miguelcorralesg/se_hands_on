# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Agentic data analysis pipeline powered by NVIDIA AI Foundation models.
#
# Architecture — five specialised agents/tools that run in sequence per user query:
#
#   ┌─────────────────────────────────────────────────────────────┐
#   │  User question                                              │
#   └──────────────────────┬──────────────────────────────────────┘
#                          ▼
#   QueryUnderstandingTool  — classifies the intent: plot vs. data query
#                          ▼
#   CodeGenerationAgent    — selects the right prompt template and asks the LLM
#                            to write pandas (+ matplotlib) code
#                          ▼
#   ExecutionAgent         — runs the generated code in an isolated environment
#                          ▼
#   ReasoningAgent         — streams an LLM explanation of the result,
#                            extracting visible <think> tokens if the model exposes them
#                          ▼
#   DataInsightAgent       — (upload-time only) produces an initial summary of
#                            the dataset and suggests analysis questions
#
# The OpenAI-compatible client connects to NVIDIA's hosted inference endpoint,
# so no local GPU is required.

import os, io, re
import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI
import matplotlib.pyplot as plt
from typing import List, Any, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# The NVIDIA API is OpenAI-compatible, so we reuse the openai SDK and only
# override the base_url to point at NVIDIA's hosted endpoint.
API_BASE_URL = "https://integrate.api.nvidia.com/v1"
API_KEY = os.environ.get("NVIDIA_API_KEY")  # Set via: export NVIDIA_API_KEY="nvapi-..."

# Plot configuration
DEFAULT_FIGSIZE = (6, 4)
DEFAULT_DPI = 100

# Truncate long results before sending to the reasoning LLM to stay within token limits
MAX_RESULT_DISPLAY_LENGTH = 300

class ModelConfig:
    """Per-model LLM hyperparameters.

    Each agent/tool in the pipeline has different latency and accuracy needs,
    so temperature and max_tokens are tuned independently per agent:

    - QueryUnderstandingTool: very low temperature + very few tokens — we only
      need a deterministic "true" / "false" classification.
    - CodeGenerationAgent: low temperature — deterministic, syntactically correct code.
    - ReasoningAgent: moderate temperature — allows fluent, varied explanations.
    - DataInsightAgent: low temperature + fewer tokens — concise factual summaries.

    reasoning_false / reasoning_true are model-specific system-prompt prefixes
    that toggle the model's internal chain-of-thought (thinking) mode on or off.
    """

    def __init__(self, model_name: str, model_url: str, model_print_name: str,
                 # QueryUnderstandingTool parameters
                 query_understanding_temperature: float = 0.1,
                 query_understanding_max_tokens: int = 5,
                 # CodeGenerationAgent parameters
                 code_generation_temperature: float = 0.2,
                 code_generation_max_tokens: int = 1024,
                 # ReasoningAgent parameters
                 reasoning_temperature: float = 0.2,
                 reasoning_max_tokens: int = 1024,
                 # DataInsightAgent parameters
                 insights_temperature: float = 0.2,
                 insights_max_tokens: int = 512,
                 reasoning_false: str = "detailed thinking off",
                 reasoning_true: str = "detailed thinking on"):
        self.MODEL_NAME = model_name
        self.MODEL_URL = model_url
        self.MODEL_PRINT_NAME = model_print_name

        self.QUERY_UNDERSTANDING_TEMPERATURE = query_understanding_temperature
        self.QUERY_UNDERSTANDING_MAX_TOKENS = query_understanding_max_tokens
        self.CODE_GENERATION_TEMPERATURE = code_generation_temperature
        self.CODE_GENERATION_MAX_TOKENS = code_generation_max_tokens
        self.REASONING_TEMPERATURE = reasoning_temperature
        self.REASONING_MAX_TOKENS = reasoning_max_tokens
        self.INSIGHTS_TEMPERATURE = insights_temperature
        self.INSIGHTS_MAX_TOKENS = insights_max_tokens
        # System-prompt tokens that enable/disable visible model thinking
        self.REASONING_FALSE = reasoning_false
        self.REASONING_TRUE = reasoning_true

# ---------------------------------------------------------------------------
# Predefined model configurations
# ---------------------------------------------------------------------------
# Two NVIDIA Nemotron models are supported. They use different thinking-toggle
# conventions:
#   - Ultra 253B uses natural language ("detailed thinking on/off")
#   - Super 49B uses a slash-command ("/no_think" to disable, "" to enable)
MODEL_CONFIGS = {
    "llama-3-1-nemotron-ultra-v1": ModelConfig(
        model_name="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        model_url="https://build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1",
        model_print_name="NVIDIA Llama 3.1 Nemotron Ultra 253B v1",
        query_understanding_temperature=0.1,
        query_understanding_max_tokens=5,
        code_generation_temperature=0.2,
        code_generation_max_tokens=1024,
        reasoning_temperature=0.6,
        reasoning_max_tokens=1024,
        insights_temperature=0.2,
        insights_max_tokens=512,
        reasoning_false="detailed thinking off",
        reasoning_true="detailed thinking on"
    ),
    "llama-3-3-nemotron-super-v1-5": ModelConfig(
        model_name="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        model_url="https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5",
        model_print_name="NVIDIA Llama 3.3 Nemotron Super 49B v1.5",
        query_understanding_temperature=0.1,
        query_understanding_max_tokens=5,
        code_generation_temperature=0.0,
        code_generation_max_tokens=1024,
        reasoning_temperature=0.6,
        reasoning_max_tokens=2048,
        insights_temperature=0.2,
        insights_max_tokens=512,
        reasoning_false="/no_think",
        reasoning_true=""
    )
}

# Active model at startup — can be overridden with the DEFAULT_MODEL env var
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "llama-3-1-nemotron-ultra-v1")
Config = MODEL_CONFIGS.get(DEFAULT_MODEL, MODEL_CONFIGS["llama-3-1-nemotron-ultra-v1"])

# Single shared OpenAI-compatible client pointing at NVIDIA's endpoint
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

def get_current_config():
    """Return the ModelConfig for the model currently selected in the UI.

    The user can switch models at any time via the sidebar dropdown, which
    updates st.session_state.current_model. This helper ensures every agent
    always reads from the latest selection rather than a module-level global.
    """
    if "current_model" in st.session_state:
        return MODEL_CONFIGS[st.session_state.current_model]
    return MODEL_CONFIGS[DEFAULT_MODEL]

# ---------------------------------------------------------------------------
# QueryUnderstandingTool
# ---------------------------------------------------------------------------
def QueryUnderstandingTool(query: str) -> bool:
    """Classify whether the user's query requires a visualisation.

    Returns True  → route to PlotCodeGeneratorTool (matplotlib code).
    Returns False → route to CodeWritingTool (pandas-only code).

    Using the LLM for classification rather than keyword matching handles
    paraphrased requests (e.g. "draw me a picture of …" or "let me see …")
    more reliably. The prompt is carefully constrained so the model responds
    with only "true" or "false", keeping max_tokens very small and latency low.
    """
    current_config = get_current_config()

    full_prompt = f"""You are a query classifier. Your task is to determine if a user query is requesting a data visualization.

IMPORTANT: Respond with ONLY 'true' or 'false' (lowercase, no quotes, no punctuation).

Classify as 'true' ONLY if the query explicitly asks for:
- A plot, chart, graph, visualization, or figure
- To "show" or "display" data visually
- To "create" or "generate" a visual representation
- Words like: plot, chart, graph, visualize, show, display, create, generate, draw

Classify as 'false' for:
- Data analysis without visualization requests
- Statistical calculations, aggregations, filtering, sorting
- Questions about data content, counts, summaries
- Requests for tables, dataframes, or text results

User query: {query}"""

    messages = [
        {"role": "system", "content": current_config.REASONING_FALSE},  # thinking OFF — fast, deterministic
        {"role": "user", "content": full_prompt}
    ]

    response = client.chat.completions.create(
        model=current_config.MODEL_NAME,
        messages=messages,
        temperature=current_config.QUERY_UNDERSTANDING_TEMPERATURE,
        max_tokens=current_config.QUERY_UNDERSTANDING_MAX_TOKENS
    )

    intent_response = response.choices[0].message.content.strip().lower()
    return intent_response == "true"

# ---------------------------------------------------------------------------
# Code generation prompt templates
# ---------------------------------------------------------------------------

def CodeWritingTool(cols: List[str], query: str) -> str:
    """Return a prompt template for pandas-only (no-plot) code generation.

    The prompt enforces strict rules so the LLM produces safe, executable code:
    - Only pandas operations on the provided `df`; no file I/O or network calls.
    - Final answer must be stored in a variable named `result` so ExecutionAgent
      can retrieve it with env.get("result").
    - Code wrapped in a single ```python fence for reliable extraction via
      extract_first_code_block().
    """
    return f"""

    Given DataFrame `df` with columns:

    {', '.join(cols)}

    Write Python code (pandas **only**, no plotting) to answer:
    "{query}"

    Rules
    -----
    1. Use pandas operations on `df` only.
    2. Rely only on the columns in the DataFrame.
    3. Assign the final result to `result`.
    4. Return your answer inside a single markdown fence that starts with ```python and ends with ```.
    5. Do not include any explanations, comments, or prose outside the code block.
    6. Use **df** as the sole data source. **Do not** read files, fetch data, or use Streamlit.
    7. Do **not** import any libraries (pandas is already imported as pd).
    8. Handle missing values (`dropna`) before aggregations.

    Example
    -----
    ```python
    result = df.groupby("some_column")["a_numeric_col"].mean().sort_values(ascending=False)
    ```

    """


def PlotCodeGeneratorTool(cols: List[str], query: str) -> str:
    """Return a prompt template for pandas + matplotlib code generation.

    Similar constraints to CodeWritingTool but allows matplotlib (pre-imported
    as `plt`). The generated Figure/Axes object must be assigned to `result`
    so ExecutionAgent and ReasoningAgent can detect it as a plot.
    DEFAULT_FIGSIZE is injected to keep chart dimensions consistent across the UI.
    """
    return f"""

    Given DataFrame `df` with columns:

    {', '.join(cols)}

    Write Python code using pandas **and matplotlib** (as plt) to answer:
    "{query}"

    Rules
    -----
    1. Use pandas for data manipulation and matplotlib.pyplot (as plt) for plotting.
    2. Rely only on the columns in the DataFrame.
    3. Assign the final result (DataFrame, Series, scalar *or* matplotlib Figure) to a variable named `result`.
    4. Create only ONE relevant plot. Set `figsize={DEFAULT_FIGSIZE}`, add title/labels.
    5. Return your answer inside a single markdown fence that starts with ```python and ends with ```.
    6. Do not include any explanations, comments, or prose outside the code block.
    7. Handle missing values (`dropna`) before plotting/aggregations.

    """

# ---------------------------------------------------------------------------
# CodeGenerationAgent
# ---------------------------------------------------------------------------

def CodeGenerationAgent(query: str, df: pd.DataFrame, chat_context: Optional[str] = None):
    """Orchestrate intent classification → prompt selection → code generation.

    Steps:
    1. Call QueryUnderstandingTool to decide plot vs. data query.
    2. Choose PlotCodeGeneratorTool or CodeWritingTool accordingly.
    3. Build a full system prompt that includes recent conversation context
       so follow-up questions (e.g. "now sort it descending") work correctly.
    4. Call the LLM with thinking disabled (we want deterministic code, not prose).
    5. Extract the code block from the markdown response.

    Returns:
        code (str)        — the extracted Python code string
        should_plot (bool) — True if this is a visualisation query
        "" (str)          — reserved for future code-thinking content
    """
    should_plot = QueryUnderstandingTool(query)
    prompt = PlotCodeGeneratorTool(df.columns.tolist(), query) if should_plot else CodeWritingTool(df.columns.tolist(), query)

    # Inject the last few user turns so the model can resolve ambiguous references
    context_section = f"\nConversation context (recent user turns):\n{chat_context}\n" if chat_context else ""

    full_prompt = f"""You are a senior Python data analyst who writes clean, efficient code.
    Solve the given problem with optimal pandas operations. Be concise and focused.
    Your response must contain ONLY a properly-closed ```python code block with no explanations before or after (starts with ```python and ends with ```).
    Ensure your solution is correct, handles edge cases, and follows best practices for data analysis.
    If the latest user request references prior results ambiguously (e.g., "it", "that", "same groups"), infer intent from the conversation context and choose the most reasonable interpretation. {context_section}{prompt}"""

    current_config = get_current_config()

    messages = [
        {"role": "system", "content": current_config.REASONING_FALSE},  # thinking OFF for code gen
        {"role": "user", "content": full_prompt}
    ]

    response = client.chat.completions.create(
        model=current_config.MODEL_NAME,
        messages=messages,
        temperature=current_config.CODE_GENERATION_TEMPERATURE,
        max_tokens=current_config.CODE_GENERATION_MAX_TOKENS
    )

    full_response = response.choices[0].message.content
    code = extract_first_code_block(full_response)
    return code, should_plot, ""

# ---------------------------------------------------------------------------
# ExecutionAgent
# ---------------------------------------------------------------------------

def ExecutionAgent(code: str, df: pd.DataFrame, should_plot: bool):
    """Execute LLM-generated code in an isolated namespace and return the result.

    Security model: the code runs in an empty dict (`{}` as globals) with only
    explicitly whitelisted names (`pd`, `df`, optionally `plt` and `io`) injected
    as locals. This prevents the generated code from accessing arbitrary module
    globals or builtins beyond what pandas/matplotlib need.

    The convention is that generated code must assign its output to `result`.
    ExecutionAgent retrieves that variable from the execution environment after
    exec() completes.

    Returns the result object on success, or an error string prefixed with
    "Error executing code:" so downstream agents can detect failures.
    """
    env = {
        "pd": pd,
        "df": df
    }

    if should_plot:
        plt.rcParams["figure.dpi"] = DEFAULT_DPI
        env["plt"] = plt
        env["io"] = io

    try:
        exec(code, {}, env)
        result = env.get("result", None)

        if result is None:
            if "result" not in env:
                return "No result was assigned to 'result' variable"

        return result
    except Exception as exc:
        return f"Error executing code: {exc}"

# ---------------------------------------------------------------------------
# ReasoningCurator + ReasoningAgent
# ---------------------------------------------------------------------------

def ReasoningCurator(query: str, result: Any) -> str:
    """Build the prompt for the reasoning LLM based on what the result is.

    Handles three result types differently:
    - Error string  → ask the model to diagnose the failure.
    - Plot object   → describe the chart by its title so the model can explain
                      what it shows without seeing the image.
    - Data value    → truncate to MAX_RESULT_DISPLAY_LENGTH to stay within
                      token limits, then ask for a plain-English interpretation.
    """
    is_error = isinstance(result, str) and result.startswith("Error executing code")
    is_plot = isinstance(result, (plt.Figure, plt.Axes))

    if is_error:
        desc = result
    elif is_plot:
        title = ""
        if isinstance(result, plt.Figure):
            title = result._suptitle.get_text() if result._suptitle else ""
        elif isinstance(result, plt.Axes):
            title = result.get_title()
        desc = f"[Plot Object: {title or 'Chart'}]"
    else:
        desc = str(result)[:MAX_RESULT_DISPLAY_LENGTH]

    if is_plot:
        prompt = f'''
        The user asked: "{query}".
        Below is a description of the plot result:
        {desc}
        Explain in 2–3 concise sentences what the chart shows (no code talk).'''
    else:
        prompt = f'''
        The user asked: "{query}".
        The result value is: {desc}
        Explain in 2–3 concise sentences what this tells about the data (no mention of charts).'''
    return prompt


def ReasoningAgent(query: str, result: Any):
    """Stream the LLM's reasoning and separate visible thinking from the final answer.

    Thinking mode is enabled (REASONING_TRUE) so the model emits <think>…</think>
    tokens before its final response. This agent:
    1. Streams tokens as they arrive and updates a Streamlit placeholder in real time.
    2. Uses a simple state machine to detect when it is inside <think>…</think> blocks
       and accumulates that text separately as `thinking_content`.
    3. After streaming completes, strips all <think>…</think> blocks from the full
       response to produce the clean `cleaned` explanation shown to the user.

    Returns:
        thinking_content (str) — raw model chain-of-thought (shown in a collapsible)
        cleaned (str)          — final explanation with thinking blocks removed
    """
    current_config = get_current_config()
    prompt = ReasoningCurator(query, result)

    response = client.chat.completions.create(
        model=current_config.MODEL_NAME,
        messages=[
            {"role": "system", "content": current_config.REASONING_TRUE},  # thinking ON
            {"role": "user", "content": "You are an insightful data analyst. " + prompt}
        ],
        temperature=current_config.REASONING_TEMPERATURE,
        max_tokens=current_config.REASONING_MAX_TOKENS,
        stream=True
    )

    thinking_placeholder = st.empty()
    full_response = ""
    thinking_content = ""
    in_think = False

    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content
            full_response += token

            # State machine: track entry/exit from <think> blocks token-by-token
            if "<think>" in token:
                in_think = True
                token = token.split("<think>", 1)[1]
            if "</think>" in token:
                token = token.split("</think>", 1)[0]
                in_think = False
            if in_think or ("<think>" in full_response and "</think>" not in full_response):
                thinking_content += token
                # Render the growing thinking block in a live collapsible panel
                thinking_placeholder.markdown(
                    f'<details class="thinking" open><summary>🤔 Model Thinking</summary><pre>{thinking_content}</pre></details>',
                    unsafe_allow_html=True
                )

    # Strip all thinking blocks to get the clean explanation
    cleaned = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
    return thinking_content, cleaned

# ---------------------------------------------------------------------------
# DataFrameSummaryTool + DataInsightAgent
# ---------------------------------------------------------------------------

def DataFrameSummaryTool(df: pd.DataFrame) -> str:
    """Build a prompt that describes the DataFrame's structure for the LLM.

    Includes shape, column names, dtypes, and missing-value counts so the model
    can generate a meaningful summary without seeing the raw data. This runs once
    when a CSV is uploaded (not on every query) to give the user an instant
    orientation to their dataset.
    """
    prompt = f"""
        Given a dataset with {len(df)} rows and {len(df.columns)} columns:
        Columns: {', '.join(df.columns)}
        Data types: {df.dtypes.to_dict()}
        Missing values: {df.isnull().sum().to_dict()}

        Provide:
        1. A brief description of what this dataset contains
        2. 3-4 possible data analysis questions that could be explored
        Keep it concise and focused."""
    return prompt


def DataInsightAgent(df: pd.DataFrame) -> str:
    """Generate a brief dataset summary and suggested questions on upload.

    Called once when a new CSV is loaded (or when the user switches models).
    Uses thinking OFF and a conservative max_tokens to keep the response fast
    and focused — this is orientation content, not deep analysis.
    """
    current_config = get_current_config()
    prompt = DataFrameSummaryTool(df)
    try:
        response = client.chat.completions.create(
            model=current_config.MODEL_NAME,
            messages=[
                {"role": "system", "content": current_config.REASONING_FALSE},
                {"role": "user", "content": "You are a data analyst providing brief, focused insights. " + prompt}
            ],
            temperature=current_config.INSIGHTS_TEMPERATURE,
            max_tokens=current_config.INSIGHTS_MAX_TOKENS
        )
        return response.choices[0].message.content
    except Exception as exc:
        raise Exception(f"Error generating dataset insights: {exc}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_first_code_block(text: str) -> str:
    """Extract the first ```python…``` fenced code block from a markdown string.

    The code generation prompts instruct the LLM to wrap its output in a single
    fenced block. This function finds that block so ExecutionAgent receives only
    the raw Python code, not surrounding prose.
    """
    start = text.find("```python")
    if start == -1:
        return ""
    start += len("```python")
    end = text.find("```", start)
    if end == -1:
        return ""
    return text[start:end].strip()

# ---------------------------------------------------------------------------
# Main Streamlit Application
# ---------------------------------------------------------------------------

def main():
    """Render the two-column Streamlit layout and wire together all agents.

    Layout:
      Left column  (30%) — model selector, CSV upload, dataset preview, initial insights
      Right column (70%) — chat interface; each user message triggers the full
                           QueryUnderstanding → CodeGeneration → Execution → Reasoning pipeline

    Session state keys:
      df             — the loaded pandas DataFrame
      current_file   — name of the uploaded file (used to detect file changes)
      current_model  — key into MODEL_CONFIGS for the active model
      messages       — list of {role, content, plot_index} dicts for chat history
      plots          — list of matplotlib Figure objects (referenced by index in messages)
      insights       — cached LLM summary generated on upload
    """
    st.set_page_config(layout="wide")
    if "plots" not in st.session_state:
        st.session_state.plots = []
    if "current_model" not in st.session_state:
        st.session_state.current_model = DEFAULT_MODEL

    left, right = st.columns([3, 7])

    # ------------------------------------------------------------------
    # Left panel — controls and dataset overview
    # ------------------------------------------------------------------
    with left:
        st.header("Data Analysis Agent")

        # Model selector — switching model clears chat history and regenerates insights
        available_models = list(MODEL_CONFIGS.keys())
        model_display_names = {key: MODEL_CONFIGS[key].MODEL_PRINT_NAME for key in available_models}

        selected_model = st.selectbox(
            "Select Model",
            options=available_models,
            format_func=lambda x: model_display_names[x],
            index=available_models.index(st.session_state.current_model)
        )

        display_config = MODEL_CONFIGS[selected_model]
        st.markdown(f"<medium>Powered by <a href='{display_config.MODEL_URL}'>{display_config.MODEL_PRINT_NAME}</a></medium>", unsafe_allow_html=True)

        file = st.file_uploader("Choose CSV", type=["csv"], key="csv_uploader")

        # Handle model switch: clear history and re-run DataInsightAgent with new model
        if selected_model != st.session_state.current_model:
            st.session_state.current_model = selected_model
            new_config = MODEL_CONFIGS[selected_model]

            if "messages" in st.session_state:
                st.session_state.messages = []
            if "plots" in st.session_state:
                st.session_state.plots = []

            if "df" in st.session_state and file is not None:
                with st.spinner("Generating dataset insights with new model …"):
                    try:
                        st.session_state.insights = DataInsightAgent(st.session_state.df)
                        st.success(f"Insights updated with {new_config.MODEL_PRINT_NAME}")
                    except Exception as e:
                        st.error(f"Error updating insights: {str(e)}")
                        if "insights" in st.session_state:
                            del st.session_state.insights
                st.rerun()

        # Clean up state when the file is removed (without triggering on model switch)
        if not file and "df" in st.session_state and "current_file" in st.session_state:
            del st.session_state.df
            del st.session_state.current_file
            if "insights" in st.session_state:
                del st.session_state.insights
            st.rerun()

        if file:
            # Load CSV and run DataInsightAgent only when a new file is detected
            if ("df" not in st.session_state) or (st.session_state.get("current_file") != file.name):
                st.session_state.df = pd.read_csv(file)
                st.session_state.current_file = file.name
                st.session_state.messages = []
                with st.spinner("Generating dataset insights …"):
                    try:
                        st.session_state.insights = DataInsightAgent(st.session_state.df)
                    except Exception as e:
                        st.error(f"Error generating insights: {str(e)}")
            elif "insights" not in st.session_state:
                # Recover insights if session state was partially cleared
                with st.spinner("Generating dataset insights …"):
                    try:
                        st.session_state.insights = DataInsightAgent(st.session_state.df)
                    except Exception as e:
                        st.error(f"Error generating insights: {str(e)}")

        if "df" in st.session_state:
            st.markdown("### Dataset Insights")
            if "insights" in st.session_state and st.session_state.insights:
                st.dataframe(st.session_state.df.head())
                st.markdown(st.session_state.insights)
                current_config_left = get_current_config()
                st.markdown(f"*<span style='color: grey; font-style: italic;'>Generated with {current_config_left.MODEL_PRINT_NAME}</span>*", unsafe_allow_html=True)
            else:
                st.warning("No insights available.")
        else:
            st.info("Upload a CSV to begin chatting with your data.")

    # ------------------------------------------------------------------
    # Right panel — chat interface
    # ------------------------------------------------------------------
    with right:
        st.header("Chat with your data")
        if "df" in st.session_state:
            current_config_right = get_current_config()
            st.markdown(f"*<span style='color: grey; font-style: italic;'>Using {current_config_right.MODEL_PRINT_NAME}</span>*", unsafe_allow_html=True)
        if "messages" not in st.session_state:
            st.session_state.messages = []

        clear_col1, clear_col2 = st.columns([9, 1])
        with clear_col2:
            if st.button("Clear chat"):
                st.session_state.messages = []
                st.session_state.plots = []
                st.rerun()

        # Render conversation history; plots are stored separately by index
        # to avoid serialisation issues with matplotlib objects in session state
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"], unsafe_allow_html=True)
                    if msg.get("plot_index") is not None:
                        idx = msg["plot_index"]
                        if 0 <= idx < len(st.session_state.plots):
                            st.pyplot(st.session_state.plots[idx], use_container_width=False)

        if "df" in st.session_state:
            if user_q := st.chat_input("Ask about your data…"):
                st.session_state.messages.append({"role": "user", "content": user_q})

                with st.spinner("Working …"):
                    # Pass the last 3 user turns as context for follow-up question resolution
                    recent_user_turns = [m["content"] for m in st.session_state.messages if m["role"] == "user"][-3:]
                    context_text = "\n".join(recent_user_turns[:-1]) if len(recent_user_turns) > 1 else None

                    # Run the full agent pipeline
                    code, should_plot_flag, code_thinking = CodeGenerationAgent(user_q, st.session_state.df, context_text)
                    result_obj = ExecutionAgent(code, st.session_state.df, should_plot_flag)
                    raw_thinking, reasoning_txt = ReasoningAgent(user_q, result_obj)
                    reasoning_txt = reasoning_txt.replace("`", "")

                # Build the assistant message with three collapsible sections:
                # 🧠 Reasoning (model chain-of-thought), explanation text, and View code
                is_plot = isinstance(result_obj, (plt.Figure, plt.Axes))
                plot_idx = None
                if is_plot:
                    fig = result_obj.figure if isinstance(result_obj, plt.Axes) else result_obj
                    st.session_state.plots.append(fig)
                    plot_idx = len(st.session_state.plots) - 1

                thinking_html = ""
                if raw_thinking:
                    thinking_html = (
                        '<details class="thinking">'
                        '<summary>🧠 Reasoning</summary>'
                        f'<pre>{raw_thinking}</pre>'
                        '</details>'
                    )

                explanation_html = reasoning_txt

                code_html = (
                    '<details class="code">'
                    '<summary>View code</summary>'
                    '<pre><code class="language-python">'
                    f'{code}'
                    '</code></pre>'
                    '</details>'
                )

                assistant_msg = f"{thinking_html}{explanation_html}\n\n{code_html}"

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_msg,
                    "plot_index": plot_idx
                })
                st.rerun()


if __name__ == "__main__":
    main()
