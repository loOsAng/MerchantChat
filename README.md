# MerchantChat
这是一个 Streamlit 应用，用于分析商家数据并通过 AI（DeepSeek）提供洞察


# Grab MEX AI Assistant 🚀

A Streamlit web application designed to provide AI-powered business insights and an interactive Q&A interface for Grab merchant data. This tool helps merchants understand their performance, identify trends, and receive actionable recommendations.

## ✨ Features

*   **Comprehensive Data Loading & Preprocessing**:
    *   Loads data from multiple CSV files: `items.csv`, `keywords.csv`, `merchant.csv`, `transaction_data.csv`, `transaction_items.csv`.
    *   Robust date parsing for various timestamp columns with error handling.
    *   Data validation and merging to create a cohesive dataset for analysis.
*   **Interactive Merchant Dashboard**:
    *   Select a specific merchant using a searchable dropdown list.
    *   Displays key performance indicators (KPIs) for the selected merchant over the last 7 days (configurable):
        *   Total Sales (RM)
        *   Total Orders
        *   Average Order Value (RM)
        *   Average Preparation Time (minutes)
    *   Highlights Top 3 Best Selling and Worst 3 Selling items (by quantity).
    *   Visualizes weekly sales trends using a line chart.
*   **AI-Powered Insights & Chat (via DeepSeek API)**:
    *   **Proactive Analysis**: Automatically generates 1-2 key insights and actionable recommendations tailored to the selected merchant's profile and recent performance data.
    *   **Interactive Q&A**: Allows merchants to ask follow-up questions about their data in a chat interface. The AI assistant answers based on the provided data summary.
*   **User-Friendly Interface**:
    *   Clean, two-column layout: Business Overview on the left, AI Chat on the right.
    *   Real-time updates when a new merchant is selected.
    *   Error messages and warnings for data issues or API failures.

## 🛠️ Technologies Used

*   **Python 3.x**
*   **Streamlit**: For building the interactive web application.
*   **Pandas**: For data manipulation and analysis.
*   **Matplotlib**: For generating charts.
*   **OpenAI Python Client (configured for DeepSeek API)**: For AI-powered insights and chat.
*   **OS, Datetime, Numpy, Random**: Standard Python libraries for utility functions.

## ⚙️ Setup and Installation

1.  **Clone the repository**:
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create and activate a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    (It's good practice to have a `requirements.txt` file. If you don't, list the main ones)
    ```bash
    pip install streamlit pandas openai matplotlib numpy
    ```
    *(If you create a `requirements.txt` file from your environment, you can just do `pip install -r requirements.txt`)*

4.  **Prepare Data Files**:
    *   Ensure the following CSV files are present in the root directory (or update `DATA_PATH` in the script):
        *   `items.csv`
        *   `keywords.csv` (Note: `get_keyword_insights` is defined but not currently used in `get_proactive_insights_from_data`)
        *   `merchant.csv`
        *   `transaction_data.csv`
        *   `transaction_items.csv`
    *   Verify that the date formats in your CSVs match the parsing logic in `load_data()`.

5.  **Set up Environment Variables**:
    *   You need a DeepSeek API key. Set it as an environment variable:
        ```bash
        # On Linux/macOS
        export DEEPSEEK_API_KEY="your_deepseek_api_key"
        # On Windows (Command Prompt)
        set DEEPSEEK_API_KEY="your_deepseek_api_key"
        # On Windows (PowerShell)
        $env:DEEPSEEK_API_KEY="your_deepseek_api_key"
        ```
        Alternatively, you can hardcode it for testing, but this is not recommended for version control.

## 🚀 Running the Application

Once the setup is complete, run the Streamlit app:

```bash
streamlit run your_script_name.py
