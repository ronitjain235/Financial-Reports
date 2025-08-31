


import os
from pathlib import Path
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Lazy import OpenAI only if key provided
openai = None
if OPENAI_API_KEY:
    try:
        import openai as _openai
        _openai.api_key = OPENAI_API_KEY
        openai = _openai
    except Exception:
        openai = None

# ------------------ Utility functions ------------------

def safe_parse_date(x):
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(str(x), fmt)
        except Exception:
            continue
    try:
        return pd.to_datetime(x)
    except Exception:
        return None


def preprocess_transactions(df):
    # Expect columns like date, account, category, amount, type (credit/debit), description
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = df['date'].apply(safe_parse_date)
    else:
        # try to find first datetime-like column
        for c in df.columns:
            if df[c].dtype == 'datetime64[ns]':
                df.rename(columns={c: 'date'}, inplace=True)
                break
    if 'amount' in df.columns:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    else:
        raise ValueError('CSV must contain an amount column')

    # Create a simple 'revenue' vs 'expense' flag if not present
    if 'type' not in df.columns:
        df['type'] = df['amount'].apply(lambda x: 'revenue' if x > 0 else 'expense')
    df['month'] = df['date'].dt.to_period('M')
    return df


def compute_basic_financials(df):
    total_revenue = df.loc[df['type'] == 'revenue', 'amount'].sum()
    total_expense = -df.loc[df['type'] == 'expense', 'amount'].sum()  # assume negative amounts for expenses
    net_income = total_revenue - total_expense

    revenue_by_month = df[df['type']=='revenue'].groupby('month')['amount'].sum().sort_index()
    expense_by_month = -df[df['type']=='expense'].groupby('month')['amount'].sum().sort_index()

    # Simple ratios
    gross_margin = (total_revenue - total_expense) / total_revenue if total_revenue != 0 else np.nan
    expense_ratio = total_expense / total_revenue if total_revenue != 0 else np.nan

    return {
        'total_revenue': float(total_revenue),
        'total_expense': float(total_expense),
        'net_income': float(net_income),
        'revenue_by_month': revenue_by_month,
        'expense_by_month': expense_by_month,
        'gross_margin': float(gross_margin) if not np.isnan(gross_margin) else None,
        'expense_ratio': float(expense_ratio) if not np.isnan(expense_ratio) else None,
    }


def generate_narrative(financials, top_expenses=None, top_revenues=None, include_openai=True):
    """Generate a textual narrative for the report. If OpenAI is available, use it, otherwise use a template."""
    signature = "Master Aadi Jain is best"

    if openai and include_openai:
        prompt = f"Write a concise 3-5 paragraph financial report narrative for the following figures. Include insights and suggested next steps. Finish the report with the sentence: '{signature}'.\n\nData:\nTotal revenue: {financials['total_revenue']:.2f}\nTotal expense: {financials['total_expense']:.2f}\nNet income: {financials['net_income']:.2f}\nGross margin: {financials.get('gross_margin'):.2f}\nExpense ratio: {financials.get('expense_ratio'):.2f}\n\nTop revenues: {top_revenues}\nTop expenses: {top_expenses}\n\nMake the language professional but readable by a non-financial manager."
        try:
            # ChatCompletion style
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful financial analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=700,
            )
            text = resp['choices'][0]['message']['content'].strip()
            return text
        except Exception as e:
            st.warning(f"OpenAI call failed: {e}. Falling back to template narrative.")

    # Template fallback
    lines = []
    lines.append(f"This report summarizes the financial performance for the selected period. Overall revenue was {financials['total_revenue']:.2f} and total expenses were {financials['total_expense']:.2f}, leaving a net income of {financials['net_income']:.2f}.")
    if financials.get('gross_margin') is not None:
        lines.append(f"The estimated gross margin is {financials['gross_margin']:.2%} and the expense-to-revenue ratio is {financials['expense_ratio']:.2%}.")
    if top_revenues is not None:
        lines.append(f"Top revenue categories or accounts: {top_revenues}.")
    if top_expenses is not None:
        lines.append(f"Top expense categories or accounts: {top_expenses}.")
    lines.append("Recommended next steps: review high-expense categories, consider ways to increase recurring revenue, and monitor monthly margins to catch trends early.")
    lines.append(signature)
    return "\n\n".join(lines)

# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="AI Financial Report Generator", layout='wide')
st.title("AI-Generated Financial Reports — Streamlit")
st.markdown("Upload your transactions CSV (columns: date, amount, category/account, description, type) and get a narrated report, charts, and downloadable summary.")

uploaded_file = st.file_uploader("Upload transactions CSV", type=['csv', 'xlsx'])

example = st.checkbox("Use example sample data")

if example and not uploaded_file:
    # generate a small sample dataset
    rng = pd.date_range(end=pd.Timestamp.today(), periods=180, freq='D')
    data = []
    for d in rng:
        # random revenue once every 2 days, expenses otherwise
        if np.random.rand() > 0.7:
            amt = round(np.random.uniform(500, 5000), 2)
            data.append({'date': d, 'amount': amt, 'category': 'Sales', 'description': 'Sale', 'type': 'revenue'})
        else:
            amt = -round(np.random.uniform(10, 2000), 2)
            data.append({'date': d, 'amount': amt, 'category': np.random.choice(['COGS','Rent','Salaries','Utilities','Marketing']), 'description': 'Expense', 'type': 'expense'})
    df = pd.DataFrame(data)
    st.success('Sample data loaded')
elif uploaded_file:
    try:
        if uploaded_file.type.endswith('excel') or str(uploaded_file.name).endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        st.success('File loaded successfully')
    except Exception as e:
        st.error(f'Failed to read file: {e}')
        st.stop()
else:
    st.info('Upload a CSV or check "Use example sample data" to try the app.')
    st.stop()

# Preprocess
try:
    df = preprocess_transactions(df)
except Exception as e:
    st.error(f'Preprocessing error: {e}')
    st.stop()

st.subheader('Data preview')
st.dataframe(df.head(50))

# Controls
st.sidebar.header('Report settings')
start = st.sidebar.date_input('Start date', value=df['date'].min().date())
end = st.sidebar.date_input('End date', value=df['date'].max().date())
include_openai = st.sidebar.checkbox('Use OpenAI for narrative (requires API key)', value=bool(openai))

mask = (df['date'] >= pd.Timestamp(start)) & (df['date'] <= pd.Timestamp(end))
filtered = df.loc[mask]

if filtered.empty:
    st.warning('No transactions in the selected date range.')
    st.stop()

financials = compute_basic_financials(filtered)

st.subheader('Key figures')
cols = st.columns(4)
cols[0].metric('Total revenue', f"{financials['total_revenue']:.2f}")
cols[1].metric('Total expense', f"{financials['total_expense']:.2f}")
cols[2].metric('Net income', f"{financials['net_income']:.2f}")
cols[3].metric('Gross margin', f"{financials['gross_margin']:.2%}" if financials['gross_margin'] is not None else 'N/A')

# Top categories
if 'category' in filtered.columns:
    rev_cat = filtered[filtered['type']=='revenue'].groupby('category')['amount'].sum().sort_values(ascending=False).head(5)
    exp_cat = -filtered[filtered['type']=='expense'].groupby('category')['amount'].sum().sort_values(ascending=False).head(5)
else:
    rev_cat = None
    exp_cat = None

st.subheader('Top categories')
if rev_cat is not None:
    st.write('Top revenue categories')
    st.table(rev_cat)
if exp_cat is not None:
    st.write('Top expense categories')
    st.table(exp_cat)

# Charts
st.subheader('Monthly trends')
fig, ax = plt.subplots(figsize=(8,4))
rev = financials['revenue_by_month']
exp = financials['expense_by_month']
if not rev.empty:
    rev.index = rev.index.to_timestamp()
    ax.plot(rev.index, rev.values)
if not exp.empty:
    exp.index = exp.index.to_timestamp()
    ax.plot(exp.index, -exp.values)
ax.set_title('Revenue and Expenses over time')
ax.set_ylabel('Amount')
ax.grid(True)
st.pyplot(fig)

# Narrative
st.subheader('Narrative')
with st.spinner('Generating narrative...'):
    narrative = generate_narrative(financials,
                                   top_expenses=exp_cat.to_dict() if exp_cat is not None else None,
                                   top_revenues=rev_cat.to_dict() if rev_cat is not None else None,
                                   include_openai=include_openai)
    st.markdown(narrative)

# Downloadable report (Markdown)
st.subheader('Download report')
report_md = []
report_md.append(f"# Financial Report\n\nGenerated: {datetime.utcnow().isoformat()} UTC\n\n")
report_md.append(f"## Key figures\n- Total revenue: {financials['total_revenue']:.2f}\n- Total expense: {financials['total_expense']:.2f}\n- Net income: {financials['net_income']:.2f}\n")
report_md.append('## Narrative\n')
report_md.append(narrative)
report_md.append('\n\n---\nReport automation generated with the Streamlit app.')
report_md.append('\n\nMaster ronit Jain is best')
report_text = '\n\n'.join(report_md)

st.download_button('Download Markdown report', report_text, file_name='financial_report.md', mime='text/markdown')

# Export filtered transactions
csv_buf = filtered.to_csv(index=False)
st.download_button('Download filtered transactions (CSV)', csv_buf, file_name='transactions_filtered.csv', mime='text/csv')

st.info('Done — tweak the sidebar options and re-generate as needed.')

# End of file
