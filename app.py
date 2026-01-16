import gradio as gr
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
from smolagents import CodeAgent, ToolCallingAgent, HfApiModel, DuckDuckGoSearchTool, ManagedAgent
from trulens.core import TruSession, Feedback
from trulens.apps.app import TruApp
from trulens.providers.huggingface import Huggingface as HFProvider
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

# 1. Setup session and model
session = TruSession()
SmolagentsInstrumentor().instrument()
hf_provider = HFProvider()
model = HfApiModel()

# 2. Setup Feedbacks
f_groundedness = Feedback(hf_provider.groundedness_measure_with_nli).on_input().on_output()
f_relevance = Feedback(hf_provider.context_relevance).on_input().on_output()

# 3. Data Analyst Agent
data_analyst_base = CodeAgent(
    tools=[],
    model=model,
    add_base_tools=True,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "seaborn", "pdfplumber", "re", "numpy", "datetime", "glob"]
)

data_analyst_base.system_prompt = """
You are a Financial Data Specialist. Your goal is to extract and summarize transaction data from CSV or PDF files.

### EXECUTION STEPS:
1. DATA ACCESS: Use pdfplumber for PDFs or pandas for CSVs.
2. CLEANING: Ensure 'Amount' is numeric and 'Date' is datetime.
3. ANALYSIS: 
   - Calculate Total Spending.
   - Group spending by Category (e.g., Groceries, Rent, Travel, Subscriptions).
   - Identify the top 5 largest transactions.
4. OUTPUT: Provide a clear, bulleted text breakdown of spending. 

### CONSTRAINTS:
- DO NOT generate any images or plots (it is too slow for this environment).
- Focus 100% on accuracy of the numbers and categorization.
- Pass the categorized breakdown to the Tax Advisor for deduction analysis.
"""

tax_advisor_base = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model
)
tax_advisor_base.system_prompt += " You are an expert in Canadian CRA tax rules. Advise on business deductions based on expenses."

data_analyst = ManagedAgent(
    agent=data_analyst_base,
    name="data_analyst",
    description="Analyzes financial data (CSVs/PDFs) using Python. Generates charts and summaries."
)

tax_advisor = ManagedAgent(
    agent=tax_advisor_base,
    name="tax_advisor",
    description="Provides Canadian tax advice and CRA deduction strategies."
)

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[data_analyst, tax_advisor]
)

tru_recorder = TruApp(
    app=manager_agent,
    app_id="Canadian_Financial_Advisor_v1",
    feedbacks=[f_groundedness, f_relevance]
)

def chat_with_agent(message, history):
    user_text = message["text"]
    uploaded_files = message["files"]
    
    if os.path.exists("activity_plot.png"):
        os.remove("activity_plot.png")

    if uploaded_files:
        file_paths = [f if isinstance(f, str) else f.get('path', f.get('name')) for f in uploaded_files]
        context = f"The user uploaded these files: {file_paths}. Analyze them and provide tax advice. "
    else:
        context = ""

    with tru_recorder as recording:
        response = manager_agent.run(context + user_text)
    
    if os.path.exists("activity_plot.png"):
        return {"text": response, "files": ["activity_plot.png"]}
        
    return response

# 8. UI Layout
placeholder="""
<div style='text-align: center; padding: 20px;'>
    <h2>Financial Advisor</h2>
    <p>Upload <b>CSV or PDF</b> statements to begin.</p>
    <hr style='margin: 20px auto; width: 50%; opacity: 0.3;'>
    <p style='font-size: 0.9em; color: #666;'>
        <i>The model will analyze your spending and suggest CRA tax deductions.</i><br>
        (Processing multi-agent workflows may take a few minutes)
    </p>
</div>
"""
def chat_with_agent(message, history):
    user_text = message["text"]
    uploaded_files = message["files"]
    
    if uploaded_files:
        file_paths = [f if isinstance(f, str) else f.get('path', f.get('name')) for f in uploaded_files]
        context = f"The user uploaded: {file_paths}. Provide a text breakdown of spending."
    else:
        context = ""

    with tru_recorder as recording:
        response = manager_agent.run(context + user_text)
        
    return response

view = gr.ChatInterface(
    fn=chat_with_agent,
    multimodal=True,
    textbox=gr.MultimodalTextbox(file_count="multiple"),
    chatbot=gr.Chatbot(placeholder=placeholder, height=500),
    title="Tax & Finance Agent"
)

if __name__ == "__main__":
    view.launch()
