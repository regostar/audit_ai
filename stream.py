
#!/usr/bin/env python3
"""
k1_analyzer.py

Streamlit app to upload a K-1 PDF, extract its text, send prompts to an LLM,
and render:
  • Mandatory‐fields checklist
  • Mandatory‐fields table
  • Box L bar chart (capital account analysis)
  • Box K1 pie charts (profit, loss, capital)
  • Part III bar chart (income, deductions, other)
  • Additional forms needed (bullet list)
  • Clarifying questions (checklist)

Requirements:
  pip install streamlit pdfplumber openai pandas altair python-dotenv
  Create a .env file next to this script with:
    OPENAI_API_KEY=sk-…
"""

import os
import json
from dotenv import load_dotenv
import streamlit as st
import openai
import pdfplumber
import pandas as pd
import altair as alt

# ——— Load API Key ———————————————————————————————————————————————
load_dotenv()  # loads .env into os.environ
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
    st.stop()

# ——— Streamlit Config ——————————————————————————————————————————  
st.set_page_config(page_title="K-1 Tax Form Analyzer", layout="wide")

# ——— Helpers —————————————————————————————————————————————————————
def call_llm(prompt: str, system: str = "You are a helpful assistant.") -> str:
    """Call OpenAI chat completion with the v1 API."""
    resp = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()

def extract_pdf_text(pdf_file) -> str:
    """Extract all text from the uploaded PDF using pdfplumber."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text += txt + "\n"
    return text

def plot_bar(df: pd.DataFrame, x: str, y: str, title: str):
    """Simple bar chart: negatives in red, positives in blue."""
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(x, title=None),
            y=alt.Y(y, title=None),
            color=alt.condition(
                alt.datum[y] < 0,
                alt.value("red"),
                alt.value("steelblue")
            )
        )
        .properties(width=600, height=300, title=title)
    )
    st.altair_chart(chart, use_container_width=True)



# ——— UI —————————————————————————————————————————————————————————
st.title("📄 K-1 Tax Form Analyzer")

# 1) Upload PDF
uploaded_file = st.file_uploader("Upload your K-1 PDF file", type="pdf")
if not uploaded_file:
    st.info("Please upload a K-1 PDF to get started.")
    st.stop()

# 2) Extract text
with st.spinner("Extracting text from PDF…"):
    pdf_text = extract_pdf_text(uploaded_file)

# 3) Mandatory Fields Checklist
st.header("✅ Mandatory Fields Checklist")
prompt1 = f"""
Below is the text extracted from a K-1 tax form. Identify which standard mandatory fields are present and filled:
- Partner’s name
- Partner’s SSN/EIN
- Partnership’s name
- Partnership’s EIN
- Box A–Q entries
- Box L (Capital Account Analysis)
- Signatures and dates

Output as a Markdown checklist, e.g.:
- [x] Partner’s name
- [ ] Partner’s SSN/EIN
"""
checklist_md = call_llm(prompt1 + "\n\n" + pdf_text)
for line in checklist_md.splitlines():
    if line.startswith("- ["):
        checked = (line[3] == "x")
        label = line[line.find("]")+2 :]
        st.checkbox(label, value=checked)

# 4) Mandatory Fields Table
st.header("📋 Mandatory Fields Table")
prompt2 = """
Extract the values for each mandatory field (same list as above) and output as a Markdown table with columns: Field | Value.
"""
table_md = call_llm(prompt2 + "\n\n" + pdf_text)
st.markdown(table_md)

# 5) Box L – Capital Account Analysis (Bar Chart)
st.header("📊 Capital Account Analysis (Box L)")
prompt3 = """
Extract the four line items under "Box L – Capital Account Analysis" and their amounts.
Return JSON array like:
[
  { "Category": "Beginning Capital",   "Amount": 10000 },
  { "Category": "Contributions",       "Amount": 5000 },
  { "Category": "Withdrawals",         "Amount": -2000 },
  { "Category": "Ending Capital",      "Amount": 13000 }
]
"""
json_l = call_llm(prompt3 + "\n\n" + pdf_text)
try:
    df_l = pd.DataFrame(json.loads(json_l))
    plot_bar(df_l, "Category", "Amount", "Box L – Capital Account Analysis")
except Exception:
    st.error("Failed to parse Box L data:")
    st.code(json_l)

# 6) Box K1 – Partner's Share (Pie Charts)
st.header("🥧 Partner's Share of Profit, Loss & Capital (Box K1)")
prompt4 = f"""
Extract three percentage values from Box K1: Partner's share of Profit, Loss, and Capital.
Return ONLY a JSON array in this exact format, with no additional text:
[
  {{ "Type": "Profit",  "Amount": 100.0 }},
  {{ "Type": "Loss",    "Amount": 100.0 }},
  {{ "Type": "Capital", "Amount": 100.0 }}
]
Important:
1. Amount values must be the actual percentage numbers (e.g., use 100.0 for 100%)
2. Use exactly these Type values: "Profit", "Loss", "Capital"
3. Do not include the % symbol in the Amount values
"""
json_k1 = call_llm(prompt4 + "\n\n" + pdf_text)
try:
    # Try to clean up the response if it contains extra text
    json_str = json_k1.strip()
    if json_str.startswith('```json'):
        json_str = json_str[7:]
    if json_str.startswith('```'):
        json_str = json_str[3:]
    if json_str.endswith('```'):
        json_str = json_str[:-3]
    json_str = json_str.strip()
    
    df_k1 = pd.DataFrame(json.loads(json_str))
    if len(df_k1) != 3 or not all(t in df_k1['Type'].values for t in ['Profit', 'Loss', 'Capital']):
        raise ValueError("Invalid data format - missing required types")
    
    # Ensure we have percentage values
    df_k1['Amount'] = pd.to_numeric(df_k1['Amount'])
    if (df_k1['Amount'] < 0).any() or (df_k1['Amount'] > 100).any():
        raise ValueError("Invalid percentage values - must be between 0 and 100")
    
    # Create a dummy row for each share type to make the pie chart show 100%
    df_k1_viz = pd.DataFrame()
    for _, row in df_k1.iterrows():
        share_data = [
            {'Type': row['Type'], 'Category': 'Share', 'Amount': row['Amount']},
            {'Type': row['Type'], 'Category': 'Remainder', 'Amount': 100 - row['Amount']}
        ]
        df_k1_viz = pd.concat([df_k1_viz, pd.DataFrame(share_data)], ignore_index=True)
    
    cols = st.columns(3)
    for i, type_name in enumerate(['Profit', 'Loss', 'Capital']):
        type_data = df_k1_viz[df_k1_viz['Type'] == type_name]
        share_value = type_data[type_data['Category'] == 'Share']['Amount'].iloc[0]
        with cols[i]:
            chart = (alt.Chart(type_data)
                .mark_arc(innerRadius=50)
                .encode(
                    theta=alt.Theta("Amount:Q", stack=True),
                    color=alt.Color(
                        'Category:N',
                        scale=alt.Scale(domain=['Share', 'Remainder'],
                                      range=['steelblue', '#f0f0f0'])
                    ),
                    tooltip=['Category:N', 'Amount:Q']
                )
                .properties(
                    width=200,
                    height=200,
                    title=alt.TitleParams(
                        text=f"{type_name} Share: {share_value:.1f}%",
                        anchor='middle'
                    )
                )
            )
            st.altair_chart(chart, use_container_width=True)
except Exception:
    st.error("Failed to parse Box K1 data:")
    st.code(json_k1)

# 7) Part III – Income, Deductions & Other Financials (Bar Chart)
st.header("📈 Income, Deductions & Other Financials (Part III)")
prompt5 = """
Extract Part III items (Income, Deductions, Credits, etc.) and output ONLY a JSON array in this exact format:
[
  { "Category": "Ordinary Business Income", "Amount": 12000 },
  { "Category": "Net Rental Income",        "Amount": 5000 },
  { "Category": "Interest Income",         "Amount": 1000 },
  { "Category": "Section 179 Deduction",   "Amount": -3000 },
  { "Category": "Other Deductions",        "Amount": -2000 }
]

Important:
1. Amount values must be numbers (not strings)
2. Use negative values for deductions and losses
3. Return only the JSON array, no other text
"""
json_part3 = call_llm(prompt5 + "\n\n" + pdf_text)
try:
    # Try to clean up the response if it contains extra text
    json_str = json_part3.strip()
    if json_str.startswith('```json'):
        json_str = json_str[7:]
    if json_str.startswith('```'):
        json_str = json_str[3:]
    if json_str.endswith('```'):
        json_str = json_str[:-3]
    json_str = json_str.strip()
    
    df3 = pd.DataFrame(json.loads(json_str))
    
    # Validate the data
    if len(df3) == 0:
        raise ValueError("No data found")
    if not all(col in df3.columns for col in ['Category', 'Amount']):
        raise ValueError("Missing required columns")
    
    # Convert Amount to numeric and sort by absolute value
    df3['Amount'] = pd.to_numeric(df3['Amount'])
    df3 = df3.sort_values('Amount', key=abs, ascending=False)
    
    plot_bar(df3, "Category", "Amount", "Part III Financials")
except Exception as e:
    st.error("Failed to parse Part III data:")
    st.code(json_part3)
    st.error(f"Error: {str(e)}")

# 8) Additional Forms Needed
st.header("📝 Additional Forms Needed")
prompt6 = """
Based on the K-1 data, list any additional tax forms the taxpayer needs to fill out (e.g., Schedule E, Form 8995).
Output as bullet points.
"""
add_forms = call_llm(prompt6 + "\n\n" + pdf_text)
st.text_area("Forms", add_forms, height=150)

# 9) Clarifying Questions for Client
st.header("❓ Clarifying Questions for Client")
prompt7 = """
Given the details of this K-1 form, what follow-up or clarifying questions should we ask the client?
Provide as a Markdown checklist.
"""
questions_md = call_llm(prompt7 + "\n\n" + pdf_text)
st.markdown(questions_md)
