import streamlit as st
import pandas as pd
from openai import OpenAI
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import matplotlib #needed for plots
import random #mock data
import numpy as np 

#Page Configuration
st.set_page_config(layout="wide")

#Constants 
DATA_PATH = "./"
ITEMS_FILE = os.path.join(DATA_PATH, "items.csv")
KEYWORDS_FILE = os.path.join(DATA_PATH, "keywords.csv")
MERCHANTS_FILE = os.path.join(DATA_PATH, "merchant.csv")
TRANSACTIONS_FILE = os.path.join(DATA_PATH, "transaction_data.csv")
TRANSACTION_ITEMS_FILE = os.path.join(DATA_PATH, "transaction_items.csv")
ANALYSIS_PERIOD_DAYS = 7 # Analyze the last 7 days

#Load Data
@st.cache_data #load cache data(avoid repeat loading)
def load_data():
    #Loads all necessary CSV files into pandas DataFrames
    try:
        items_df = pd.read_csv(ITEMS_FILE)
        keywords_df = pd.read_csv(KEYWORDS_FILE)
        merchants_df = pd.read_csv(MERCHANTS_FILE)
        transactions_df = pd.read_csv(TRANSACTIONS_FILE, low_memory=False)
        transaction_items_df = pd.read_csv(TRANSACTION_ITEMS_FILE)

        #Date Parsing
        date_cols = ['order_time', 'driver_arrival_time', 'driver_pickup_time', 'delivery_time']
        placeholder_values = ['################', '#', 'nan', '']

        for col in date_cols:
            if col in transactions_df.columns:
                 transactions_df[col] = transactions_df[col].astype(str)
                 transactions_df[col].replace(placeholder_values, np.nan, inplace=True)
                 transactions_df[col] = pd.to_datetime(transactions_df[col], dayfirst=False, errors='coerce')
            else:
                 if col == 'order_time':
                     st.error(f"Critical column '{col}' not found in {TRANSACTIONS_FILE}. Cannot proceed.")
                     st.stop()
                 else:
                     st.warning(f"Date column '{col}' not found in {TRANSACTIONS_FILE}. Related calculations might be affected.")

        # Parse merchant join date
        merchant_join_date_col = 'join_date'
        merchant_join_date_format = '%d%m%Y'
        try:
            if merchant_join_date_col in merchants_df.columns:
                 merchants_df[merchant_join_date_col] = pd.to_datetime(merchants_df[merchant_join_date_col], format=merchant_join_date_format, errors='coerce')
            else:
                 st.warning(f"Column '{merchant_join_date_col}' not found in {MERCHANTS_FILE}. Business maturity info unavailable.")
                 merchants_df[merchant_join_date_col] = pd.NaT # Create column if missing
        except Exception as e:
             st.warning(f"Could not parse '{merchant_join_date_col}' with format '{merchant_join_date_format}': {e}. Check format in {MERCHANTS_FILE}.")
             if merchant_join_date_col not in merchants_df.columns: # Avoid error if col missing AND parse failed
                 merchants_df[merchant_join_date_col] = pd.NaT


        # Drop rows where order_time couldn't be parsed
        original_rows = len(transactions_df)
        if 'order_time' in transactions_df.columns:
            transactions_df.dropna(subset=['order_time'], inplace=True)
            rows_dropped = original_rows - len(transactions_df)
            if rows_dropped > 0:
                st.warning(f"Dropped {rows_dropped} rows from transactions due to invalid 'order_time' values after attempting to parse. Please verify the date format in {TRANSACTIONS_FILE}.")
                if original_rows > 0 and (rows_dropped / original_rows > 0.5) : # Check division by zero
                    st.error("Warning: A large number of transaction rows were dropped due to invalid dates. Analysis results may be inaccurate. Please check the 'order_time' format in your CSV.")
        else:
            st.error(f"'order_time' column is missing from {TRANSACTIONS_FILE}. Cannot proceed with analysis.")
            st.stop()


        #Validation and Merging
        if items_df.empty or merchants_df.empty or transactions_df.empty or transaction_items_df.empty:
             st.error("One or more essential data files are empty or failed to load/parse. Cannot proceed.")
             st.stop()

        #Merge merchant names
        if 'merchant_id' in transactions_df.columns and 'merchant_id' in merchants_df.columns:
             if 'merchant_name' not in merchants_df.columns: merchants_df['merchant_name'] = 'Unknown Merchant Name'
             merchants_df['merchant_name'].fillna('Unknown Merchant Name', inplace=True)
             transactions_df = pd.merge(transactions_df, merchants_df[['merchant_id', 'merchant_name']], on='merchant_id', how='left')
             transactions_df['merchant_name'].fillna('Unknown Merchant Name', inplace=True)
        else:
             st.warning("'merchant_id' column missing in transactions or merchants file. Cannot merge merchant names.")
             transactions_df['merchant_name'] = 'Unknown (Merge Failed)'


        return items_df, keywords_df, merchants_df, transactions_df, transaction_items_df
    except FileNotFoundError as e:
        st.error(f"Error loading data file: {e}. Make sure CSV files are in the directory '{os.path.abspath(DATA_PATH)}'.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading or initial processing: {e}")
        st.stop()

#Load data
items_df, keywords_df, merchants_df, transactions_df, transaction_items_df = load_data()


#API Setup
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    st.error("Error: DEEPSEEK_API_KEY environment variable not set.")
    st.stop()
try:
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
except Exception as e:
    st.error(f"Error initializing DeepSeek Client: {e}")
    st.stop()


#Data Processing Function
def process_merchant_data(selected_merchant_id, transactions_df, transaction_items_df, items_df, period_days=7):
    #Analyzes data for a specific merchant over a given period
    required_tx_cols = ['merchant_id', 'order_time', 'order_value', 'order_id']
    if not all(col in transactions_df.columns for col in required_tx_cols):
        return {"error": f"Missing required columns in transaction data ({required_tx_cols})."}
    required_ti_cols = ['order_id', 'item_id']
    if not all(col in transaction_items_df.columns for col in required_ti_cols):
         return {"error": f"Missing required columns in transaction items data ({required_ti_cols})."}
    required_item_cols = ['item_id', 'item_name', 'item_price']
    if not all(col in items_df.columns for col in required_item_cols):
         return {"error": f"Missing required columns in items data ({required_item_cols})."}


    if transactions_df['order_time'].isnull().all():
         return {"error": "No valid order times found after processing."}
    end_date = transactions_df['order_time'].max()
    start_date = end_date - timedelta(days=period_days)

    merchant_tx = transactions_df[
        (transactions_df['merchant_id'] == selected_merchant_id) &
        (transactions_df['order_time'] >= start_date) &
        (transactions_df['order_time'] <= end_date)
    ].copy()

    if merchant_tx.empty:
        return {"error": f"No transaction data found for merchant {selected_merchant_id} between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}."}

    total_sales = merchant_tx['order_value'].sum()
    total_orders = merchant_tx['order_id'].nunique()
    avg_order_value = total_sales / total_orders if total_orders > 0 else 0

    merchant_order_ids = merchant_tx['order_id'].tolist()
    items_sold = transaction_items_df[transaction_items_df['order_id'].isin(merchant_order_ids)].copy()

    best_selling_items = {}
    worst_selling_items = {}
    if not items_sold.empty:
        items_sold_details = pd.merge(items_sold, items_df[['item_id', 'item_name', 'item_price']], on='item_id', how='left')
        items_sold_details['item_name'].fillna('Unknown Item', inplace=True)
        items_sold_details['item_price'].fillna(0, inplace=True)
        item_counts = items_sold_details['item_name'].value_counts()
        if not item_counts.empty:
             best_selling_items = item_counts.head(3).to_dict()
             worst_selling_items = item_counts.tail(3).to_dict()


    merchant_tx['order_week'] = merchant_tx['order_time'].dt.strftime('%Y-W%U')
    weekly_sales_trend = merchant_tx.groupby('order_week')['order_value'].sum().sort_index().to_dict()

    avg_prep_time = None
    pickup_col = 'driver_pickup_time'
    order_t_col = 'order_time'
    if pickup_col in merchant_tx.columns and order_t_col in merchant_tx.columns and pd.api.types.is_datetime64_any_dtype(merchant_tx[pickup_col]) and pd.api.types.is_datetime64_any_dtype(merchant_tx[order_t_col]):
         pickup_time_naive = merchant_tx[pickup_col].dt.tz_localize(None) if merchant_tx[pickup_col].dt.tz is not None else merchant_tx[pickup_col]
         order_time_naive = merchant_tx[order_t_col].dt.tz_localize(None) if merchant_tx[order_t_col].dt.tz is not None else merchant_tx[order_t_col]
         merchant_tx['prep_time_minutes'] = (pickup_time_naive - order_time_naive).dt.total_seconds() / 60
         valid_prep_times = merchant_tx['prep_time_minutes'][(merchant_tx['prep_time_minutes'] >= 0) & (merchant_tx['prep_time_minutes'] < 120)]
         if not valid_prep_times.empty:
              avg_prep_time = valid_prep_times.mean()

    summary = {
        "merchant_id": selected_merchant_id,
        "report_period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        "total_sales": total_sales, "total_orders": total_orders, "avg_order_value": avg_order_value,
        "best_selling_items_count": best_selling_items, "worst_selling_items_count": worst_selling_items,
        "avg_prep_time_minutes": avg_prep_time, "weekly_sales_trend": weekly_sales_trend,
    }
    return summary


#Keyword Insights Function
def get_keyword_insights(keywords_df):
    #Provides general keyword insights.
    required_keyword_cols = ['keyword', 'view', 'order']
    if keywords_df.empty or not all(col in keywords_df.columns for col in required_keyword_cols):
        return {"top_market_keywords": "Keyword data unavailable or missing required columns."}
    try:
        keywords_df['order'] = pd.to_numeric(keywords_df['order'], errors='coerce')
        keywords_df.dropna(subset=['order'], inplace=True)
        keywords_df['view'] = pd.to_numeric(keywords_df['view'], errors='coerce')

        top_keywords = keywords_df.nlargest(5, 'order')[required_keyword_cols].to_dict('records')
        return {"top_market_keywords": top_keywords}
    except Exception as e:
         return {"top_market_keywords": f"Error processing keywords: {e}"}


#AI Insight Generation Function
def get_proactive_insights_from_data(merchant_profile, data_summary, keyword_insights=None):
    #Generates insights using the processed data summary and optional keywords.
    if "error" in data_summary:
         return f"Could not generate insights due to a data processing error: {data_summary['error']}"
    prep_time_str = f"{data_summary.get('avg_prep_time_minutes'):.1f}" if isinstance(data_summary.get('avg_prep_time_minutes'), float) else 'N/A'
    prompt_data_section = f"""
**This Week's Business Data Summary ({data_summary.get('report_period', 'N/A')}):**
*   Total Sales: RM {data_summary.get('total_sales', 0):.2f}
*   Total Orders: {data_summary.get('total_orders', 0)}
*   Avg. Order Value: RM {data_summary.get('avg_order_value', 0):.2f}
*   Best Selling Items (by count): {data_summary.get('best_selling_items_count', {})}
*   Worst Selling Items (by count): {data_summary.get('worst_selling_items_count', {})}
*   Avg. Prep Time: {prep_time_str} minutes
*   Weekly Sales Trend (Week: Amount): {data_summary.get('weekly_sales_trend', {})}
"""
    
    if keyword_insights and isinstance(keyword_insights.get("top_market_keywords"), list):
         prompt_data_section += f"\n*   Top Market Keywords (by orders): {keyword_insights['top_market_keywords']}"
    elif keyword_insights and isinstance(keyword_insights.get("top_market_keywords"), str): # Handle error string
         prompt_data_section += f"\n*   Top Market Keywords: {keyword_insights['top_market_keywords']}"


    join_date_col_in_profile = 'join_date'

    system_prompt = f"""
You are a top Grab platform AI business analyst and consultant.
Analyze the weekly data for '{merchant_profile.get('merchant_name', 'this merchant')}' ({merchant_profile.get('type', 'Unknown type')}) and **proactively** provide **1-2 most important insights and specific, actionable recommendations**.

**Rules:**
1.  **Prioritize:** Focus on critical issues or big opportunities revealed in the data.
2.  **Actionable:** Give concrete steps.
3.  **Personalize:** Tailor advice to the merchant profile (e.g., Type: {merchant_profile.get('type', 'N/A')}, Location: {merchant_profile.get('location', 'N/A')}, Joined: {merchant_profile.get(join_date_col_in_profile, pd.NaT).strftime('%Y-%m-%d') if pd.notnull(merchant_profile.get(join_date_col_in_profile)) else 'N/A'}).
4.  **Data-Driven:** Base everything *only* on the data provided below.
5.  **Clear & Concise:** Use simple language. **Highlight** key points.

{prompt_data_section}

Generate 1-2 most important insights and recommendations:
"""
    try:
        response = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "system", "content": system_prompt}], temperature=0.6, max_tokens=400)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating initial insights: {e}"


#Streamlit App UI Layout


st.title("🚀 Grab MEX AI Assistant")
st.caption("Using data from CSV files for Task 2")

#Initialize session state
if 'selected_merchant_id' not in st.session_state:
    st.session_state.selected_merchant_id = merchants_df['merchant_id'].iloc[0] if not merchants_df.empty else None
if 'current_merchant_id' not in st.session_state:
    st.session_state.current_merchant_id = None
if 'data_summary' not in st.session_state:
    st.session_state.data_summary = None
if 'merchant_profile' not in st.session_state:
    st.session_state.merchant_profile = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'search_term' not in st.session_state:
    st.session_state.search_term = ""


#Merchant Selection with Filtering
merchant_dict = pd.Series(merchants_df.merchant_name.values, index=merchants_df.merchant_id).fillna('Unknown Name').to_dict() if not merchants_df.empty else {}
all_merchant_ids = list(merchant_dict.keys())

if not all_merchant_ids:
    st.error("No merchants found in merchant.csv. Cannot proceed.")
    st.stop()

st.session_state.search_term = st.text_input(
    "Search Merchant Name:",
    value=st.session_state.search_term
)

if st.session_state.search_term:
    search_lower = st.session_state.search_term.lower()
    filtered_merchant_dict = {
        mid: name for mid, name in merchant_dict.items()
        if search_lower in name.lower()
    }
    filtered_merchant_ids = list(filtered_merchant_dict.keys())
    if not filtered_merchant_ids:
        st.warning(f"No merchants found matching '{st.session_state.search_term}'. Displaying all merchants.")
        filtered_merchant_ids = all_merchant_ids
        filtered_merchant_dict = merchant_dict
else:
    filtered_merchant_ids = all_merchant_ids
    filtered_merchant_dict = merchant_dict


if st.session_state.selected_merchant_id not in filtered_merchant_ids:
     st.session_state.selected_merchant_id = filtered_merchant_ids[0] if filtered_merchant_ids else None

selected_merchant_id = st.selectbox(
    "Select Merchant:",
    options=filtered_merchant_ids,
    format_func=lambda mid: f"{filtered_merchant_dict.get(mid, 'Unknown Name')} ({mid})" if mid else "No Selection",
    key='selected_merchant_id',
    disabled=not filtered_merchant_ids
)


#Process Data and Update State on Selection Change
if selected_merchant_id and (st.session_state.current_merchant_id != selected_merchant_id or st.session_state.data_summary is None):
    st.session_state.data_summary = process_merchant_data(
        selected_merchant_id, transactions_df, transaction_items_df, items_df, period_days=ANALYSIS_PERIOD_DAYS
    )
    merchant_join_date_col = 'join_date'
    profile_series = merchants_df[merchants_df['merchant_id'] == selected_merchant_id].iloc[0] if selected_merchant_id in merchants_df['merchant_id'].values else None
    st.session_state.merchant_profile = profile_series.to_dict() if profile_series is not None else {"merchant_name": "Unknown Merchant", "merchant_id": selected_merchant_id, merchant_join_date_col: pd.NaT}

    st.session_state.current_merchant_id = selected_merchant_id
    st.session_state.messages = []


#Add Initial Insight Message
if selected_merchant_id and not st.session_state.messages:
    if st.session_state.data_summary and st.session_state.merchant_profile:
        if "error" not in st.session_state.data_summary:
            initial_insight = get_proactive_insights_from_data(
                st.session_state.merchant_profile,
                st.session_state.data_summary
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Hello {st.session_state.merchant_profile.get('merchant_name', 'Merchant')}! Here's your AI analysis for the last {ANALYSIS_PERIOD_DAYS} days ({st.session_state.data_summary.get('report_period', 'N/A')}). Summary on the left.\n\n**Key Insights & Suggestions:**\n\n{initial_insight}\n\nAsk me anything else!"
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Hello {st.session_state.merchant_profile.get('merchant_name', 'Merchant')}. I couldn't process your data for the selected period. Error: {st.session_state.data_summary['error']}"
            })


#Page Layout and Display
if st.session_state.data_summary and st.session_state.merchant_profile:
    left_column, right_column = st.columns([1, 2])

    #Sidebar (Left Column)
    with left_column:
        st.header("📊 Business Overview")
        profile = st.session_state.merchant_profile
        summary = st.session_state.data_summary

        if "error" in summary:
             st.error(summary["error"])
        else:
            st.subheader(f"Merchant: {profile.get('merchant_name', selected_merchant_id)}")
            join_date_col_in_profile = 'join_date'
            join_date_val = profile.get(join_date_col_in_profile)
            join_date_str = join_date_val.strftime('%Y-%m-%d') if pd.notnull(join_date_val) else 'N/A'
            st.markdown(f"**Joined:** {join_date_str} | **City ID:** {profile.get('city_id', 'N/A')}")
            st.markdown(f"**Report Period:** {summary.get('report_period', 'N/A')}")

            st.metric("Total Sales", f"RM {summary.get('total_sales', 0):.2f}")
            st.metric("Total Orders", summary.get('total_orders', 0))
            st.metric("Avg. Order Value", f"RM {summary.get('avg_order_value', 0):.2f}")
            prep_time = summary.get('avg_prep_time_minutes', 'N/A')
            st.metric("Avg. Prep Time (mins)", f"{prep_time:.1f}" if isinstance(prep_time, float) else 'N/A')

            st.subheader("Items Performance (by Count)")
            st.write("**Top Selling:**")
            st.json(summary.get('best_selling_items_count', {}))
            st.write("**Least Selling:**")
            st.json(summary.get('worst_selling_items_count', {}))

            st.subheader("📈 Sales Trend (Recent Weeks)")
            try:
                trend_data = summary.get('weekly_sales_trend', {})
                if trend_data:
                    sorted_weeks = sorted(trend_data.keys())
                    sales = [trend_data[week] for week in sorted_weeks]
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(sorted_weeks, sales, marker='o', linestyle='-', color='g')
                    ax.set_xlabel("Week")
                    ax.set_ylabel("Sales (RM)")
                    ax.grid(True, linestyle='--', alpha=0.6)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("No weekly sales trend data available for this period.")
            except Exception as e:
                st.error(f"Could not draw sales chart: {e}")

            st.markdown("---")
            st.caption("Data processed from provided CSV files.")


    #Chat Interface (Right Column)
    with right_column:
        st.header("💬 Chat with AI Assistant")
        for message in st.session_state.get("messages", []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_prompt := st.chat_input(f"Ask AI about the data..."):
            if "messages" not in st.session_state: st.session_state.messages = []
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"): st.markdown(user_prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty(); message_placeholder.markdown("🤔 Thinking...")
                if st.session_state.data_summary and "error" not in st.session_state.data_summary:
                    prep_time_str_qa = f"{st.session_state.data_summary.get('avg_prep_time_minutes'):.1f}" if isinstance(st.session_state.data_summary.get('avg_prep_time_minutes'), float) else 'N/A'
                    # Include necessary fields in the QA prompt context
                    prompt_data_section_qa = f"""
**Current Data Summary ({st.session_state.data_summary.get('report_period', 'N/A')}):**
*   Total Sales: RM {st.session_state.data_summary.get('total_sales', 0):.2f}
*   Total Orders: {st.session_state.data_summary.get('total_orders', 0)}
*   Avg. Order Value: RM {st.session_state.data_summary.get('avg_order_value', 0):.2f}
*   Best Selling Items (by count): {st.session_state.data_summary.get('best_selling_items_count', {})}
*   Worst Selling Items (by count): {st.session_state.data_summary.get('worst_selling_items_count', {})}
*   Avg. Prep Time: {prep_time_str_qa} minutes
*   Weekly Sales Trend (Week: Amount): {st.session_state.data_summary.get('weekly_sales_trend', {})}
"""
                    # Include profile info in QA prompt too for personalization potential
                    profile_info_qa = f"Merchant Name: {st.session_state.merchant_profile.get('merchant_name', 'N/A')}, Type: {st.session_state.merchant_profile.get('type', 'N/A')}" # Add more if needed

                    qa_system_prompt = f"""
You are a helpful Grab merchant AI assistant talking to {profile_info_qa}.
Answer the merchant's **specific questions** based **only** on the data summary provided below.

**Rules:**
1.  **Use Only Provided Data.**
2.  **Answer Directly.**
3.  **Admit Limitations** if the answer isn't in the data.
4.  **Cite Data (If Possible).**

{prompt_data_section_qa}

Answer the merchant's question:
"""
                    messages_for_api = [{"role": "system", "content": qa_system_prompt},{"role": "user", "content": user_prompt}]
                    try:
                        response = client.chat.completions.create(model="deepseek-chat",messages=messages_for_api,temperature=0.3,max_tokens=300)
                        ai_response = response.choices[0].message.content.strip()
                        message_placeholder.markdown(ai_response)
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    except Exception as e:
                        error_message = f"Sorry, I encountered an error calling the AI: {e}"
                        message_placeholder.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                else:
                    error_message = "I cannot answer questions as there was an issue processing the merchant data."
                    message_placeholder.warning(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

else:
     st.info("Select a merchant from the dropdown to view their analysis.")