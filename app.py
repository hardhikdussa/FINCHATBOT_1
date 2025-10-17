# Colab cell 2 — write the Streamlit app

import os
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import json

# ---------------------------
# Helper functions
# ---------------------------
def safe_load_generation_pipeline():
    """
    Try to load the desired Granite model. If it's not available or
    OOM, fall back to a small local model (gpt2) for demo purposes.
    """
    model_id = "ibm-granite/granite-3.3-2b-instruct"
    fallback =  "ibm-granite/granite-3.3-2b-instruct"
    try:
        st.sidebar.info(f"Attempting to load model: {model_id}")
        gen = pipeline("text-generation", model=model_id, device=0 if torch.cuda.is_available() else -1)
        st.sidebar.success("Loaded Granite model.")
        return gen, model_id
    except Exception as e:
        st.sidebar.warning(f"Could not load {model_id}: {e}")
        st.sidebar.info(f"Falling back to {fallback} (demo-only).")
        gen = pipeline("text-generation", model=fallback, device=0 if torch.cuda.is_available() else -1)
        return gen, fallback

def build_prompt(user_text, user_type, context=None):
    """Construct an instruction-style prompt for the generative model."""
    persona = "You are a friendly, concise personal finance assistant."
    audience = f"The user is a {user_type}."
    extra = ""
    if context:
        extra = f"Context: {context}\n"
    prompt = (
        f"{persona}\n{audience}\n{extra}"
        f"Provide practical, actionable advice on savings, taxes, or investments.\n"
        f"User: {user_text}\nAssistant:"
    )
    return prompt

def generate_response(pipe, prompt, max_tokens=150):
    """Use the pipeline to generate text — returns string"""
    # Some pipelines accept raw text, others (chat template) need different input.
    try:
        out = pipe(prompt, max_new_tokens=max_tokens, do_sample=False)
        # pipeline "text-generation" returns list of dicts with 'generated_text'
        text = out[0].get("generated_text", "")
        # If the model echoes the prompt, try to strip the prompt prefix
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        return text.strip()
    except Exception as e:
        return f"Error generating response: {e}"

def generate_budget_summary(income, expenses_dict):
    total_expenses = sum(expenses_dict.values())
    savings = income - total_expenses
    breakdown_lines = "\n".join([f"- {k}: {v}" for k, v in expenses_dict.items()])
    summary = (
        f"Monthly Income: {income}\n"
        f"Total Expenses: {total_expenses}\n"
        f"Estimated Monthly Savings: {savings}\n\n"
        f"Breakdown:\n{breakdown_lines}"
    )
    # Simple suggestion
    if savings < 0:
        suggestion = "\nSuggestion: Your expenses exceed income. Reduce non-essential spending and/or increase income."
    else:
        suggestion = "\nSuggestion: Aim to save at least 10-20% of income; consider automating transfers to savings."
    return summary + suggestion

def spending_insights(expenses_dict, income):
    insights = []
    total = sum(expenses_dict.values())
    if total == 0:
        return "No expense data provided."
    for k, v in expenses_dict.items():
        pct = (v / total) * 100
        if pct > 40:
            insights.append(f"{k} is {pct:.0f}% of total expenses — consider whether this can be reduced.")
        elif pct < 5:
            insights.append(f"{k} is only {pct:.0f}% of expenses — that's efficient.")
    # Savings rate insight
    savings = income - total
    if savings / income < 0.10:
        insights.append("Savings rate <10%: consider a higher savings target.")
    else:
        insights.append("Savings rate looks healthy.")
    return "\n".join(insights)

# ---------------------------
# App UI
# ---------------------------
st.set_page_config(page_title="Personal Finance Chatbot", layout="wide")
st.title("💸 Personal Finance Chatbot (Demo)")

# Sidebar
st.sidebar.header("Configuration")
model_pipe, loaded_model = safe_load_generation_pipeline()
st.sidebar.markdown(f"**Model:** `{loaded_model}`")

# User profile
st.sidebar.header("User Profile")
user_type = st.sidebar.selectbox("I am a:", ["Student", "Professional", "Retiree", "Other"])
age = st.sidebar.number_input("Age", min_value=13, max_value=120, value=30)

# Main columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Ask a finance question")
    user_question = st.text_area("Your question (savings, taxes, investments...)", height=120)
    if st.button("Get Advice"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            prompt = build_prompt(user_question, user_type)
            with st.spinner("Generating advice..."):
                answer = generate_response(model_pipe, prompt, max_tokens=200)
            st.markdown("**Chatbot response:**")
            st.write(answer)

    st.markdown("---")
    st.subheader("Budget / Spending Analysis")
    # Simple budget inputs (expandable)
    with st.form("budget_form"):
        income = st.number_input("Monthly Income (₹)", min_value=0.0, value=50000.0, step=1000.0)
        rent = st.number_input("Rent", value=15000.0, step=500.0)
        food = st.number_input("Food", value=8000.0, step=100.0)
        transport = st.number_input("Transport", value=2000.0)
        entertainment = st.number_input("Entertainment", value=3000.0)
        other = st.number_input("Other", value=2000.0)
        submitted = st.form_submit_button("Generate Summary")
        if submitted:
            expenses = {"Rent": rent, "Food": food, "Transport": transport, "Entertainment": entertainment, "Other": other}
            summary = generate_budget_summary(income, expenses)
            st.markdown("**Budget Summary**")
            st.text(summary)
            st.markdown("**Spending Insights**")
            st.text(spending_insights(expenses, income))

with col2:
    st.subheader("Quick prompts & features")
    st.markdown("- Adaptive tone based on user profile (Student / Professional).")
    st.markdown("- Use the textbox above to ask for: `best saving strategy`, `tax-saving options`, `investment ideas`.")
    st.markdown("**Note:** This demo may use a lightweight model if the target Granite model is not accessible.")
    st.markdown("---")
    st.subheader("Optional IBM Watson Integration")
    st.markdown("If you have IBM Watson credentials you can integrate Watson Assistant / Discovery. "
                "Set environment variables `WATSON_APIKEY` and `WATSON_URL` and extend the app.")

    st.markdown("**Debug / Raw**")
    if st.checkbox("Show internal state"):
        st.json({"loaded_model": loaded_model, "user_type": user_type, "age": age})

# ---------------------------
# End app.py
# ---------------------------
