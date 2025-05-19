import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os
from data_cleaning import clean_data, suggest_columns_to_drop
from report_generator import generate_cleaning_report_pdf
from ai_utils import sentiment_analysis

# Ensure folders exist
os.makedirs("cleaned_datasets", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Initialize session state
if 'cleaned_files' not in st.session_state:
    st.session_state.cleaned_files = []

if 'last_report_path' not in st.session_state:
    st.session_state.last_report_path = None

if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None

if 'cleaning_summary' not in st.session_state:
    st.session_state.cleaning_summary = None

# --- App Title ---
st.title("📊 AI-Powered Data Cleaning & Visualization Tool")

# --- Layout with Tabs ---
tabs = st.tabs(["📥 Upload & Clean Data", "📊 Visualize Data", "🧠 AI Insights", "📜 History & Reports"])

# --- Tab 1: Upload & Clean Data ---
with tabs[0]:
    st.header("📥 Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("### 📖 Dataset Preview")
        st.dataframe(df.head())

        st.markdown("### 🗑️ Auto-Suggested Columns to Drop")
        suggestions = suggest_columns_to_drop(df)
        if suggestions:
            suggested_columns = [col for col, _ in suggestions]
            for col, reason in suggestions:
                st.info(f"**{col}** → {reason}")

            columns_to_drop = st.multiselect(
                "Select columns to drop (suggested by system, editable):",
                options=df.columns.tolist(),
                default=suggested_columns,
                help="You can add or remove columns before cleaning."
            )
        else:
            st.warning("No columns were auto-suggested for dropping.")
            columns_to_drop = st.multiselect(
                "Select columns to drop manually:",
                options=df.columns.tolist(),
                help="Select columns you think should be removed."
            )

        if st.button("✨ Clean Data"):
            with st.spinner("Cleaning your data..."):
                cleaned_df, original_df, columns_dropped, missing_value_actions = clean_data(df, columns_to_drop)

                st.success("✅ Data cleaned successfully!")

                # Save cleaned dataset to session state
                st.session_state.cleaned_df = cleaned_df

                # Prepare description log for later display
                cleaning_summary = {
                    "columns_dropped": columns_dropped,
                    "missing_value_actions": missing_value_actions
                }
                st.session_state.cleaning_summary = cleaning_summary

                # Save cleaned CSV file
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                cleaned_filename = f"cleaned_datasets/cleaned_dataset_{timestamp}.csv"
                cleaned_df.to_csv(cleaned_filename, index=False)
                st.session_state.cleaned_files.append(cleaned_filename)
                st.info(f"💾 Cleaned dataset saved as `{cleaned_filename}`")

                # Save cleaning log
                log_filename = f"logs/data_cleaning_log_{timestamp}.log"
                with open(log_filename, 'w') as log_file:
                    log_file.write(f"Data Cleaning Log ({timestamp})\n")
                    log_file.write(f"Original Dataset Shape: {original_df.shape}\n")
                    log_file.write(f"Cleaned Dataset Shape: {cleaned_df.shape}\n")
                    log_file.write(f"Columns Dropped: {columns_dropped}\n")
                    log_file.write(f"Missing Value Actions: {missing_value_actions}\n")

                # Generate PDF report
                pdf_filename = f"cleaned_datasets/data_cleaning_report_{timestamp}.pdf"
                generate_cleaning_report_pdf(original_df, cleaned_df, columns_dropped, missing_value_actions, pdf_filename)
                st.session_state.last_report_path = pdf_filename

    # If cleaning was done, show results and summary
    if st.session_state.cleaned_df is not None:
        st.markdown("### 📊 Full Cleaned Data")
        st.dataframe(st.session_state.cleaned_df)

        # 📥 Download button for Cleaned CSV
        csv = st.session_state.cleaned_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Cleaned Dataset as CSV",
            data=csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

        # 📖 Description of Data Cleaning
        with st.expander("📖 What we did during Data Cleaning"):
            st.markdown("""
            **Why Data Cleaning is Important:**
            - Raw datasets often contain missing values, duplicates, irrelevant or inconsistent columns.
            - Cleaning ensures your data is reliable for analytics, visualizations, and AI/ML models.

            **Steps Performed:**
            - **Columns Dropped**: Removed user-selected and suggested irrelevant columns.
            - **Missing Values**:
              - Numeric columns filled with **median**.
              - Categorical columns filled with **most frequent (mode)**.
            - **Categorical Encoding**: Non-numeric columns converted to numerical codes for AI-readiness.
            """)

            summary = st.session_state.cleaning_summary
            if summary:
                if summary["columns_dropped"]:
                    st.write(f"**Columns Dropped:** {summary['columns_dropped']}")
                else:
                    st.write("No columns were dropped.")

                if summary["missing_value_actions"]:
                    st.write("**Missing Value Handling:**")
                    for col, action in summary["missing_value_actions"].items():
                        st.write(f"- {col}: {action}")
                else:
                    st.write("No missing values found.")

# --- Tab 2: Visualize Data ---
with tabs[1]:
    st.header("📊 Visualize Your Cleaned Data")

    if st.session_state.cleaned_df is None:
        st.info("👉 Upload and clean a dataset first to visualize data here.")
    else:
        vis_option = st.selectbox("Select visualization type", ["None", "Histogram", "Boxplot", "Correlation Heatmap"])

        if vis_option != "None":
            target_df = st.session_state.cleaned_df

            if vis_option in ["Histogram", "Boxplot"]:
                numeric_cols = target_df.select_dtypes(include=["number"]).columns.tolist()
                if not numeric_cols:
                    st.warning("⚠️ No numeric columns available for this visualization.")
                else:
                    col = st.selectbox("Select numeric column", numeric_cols)
                    fig, ax = plt.subplots(figsize=(8, 4))

                    if vis_option == "Histogram":
                        sns.histplot(target_df[col], kde=True, ax=ax)
                    else:
                        sns.boxplot(y=target_df[col], ax=ax)

                    st.pyplot(fig)

            elif vis_option == "Correlation Heatmap":
                corr = target_df.corr()
                if corr.empty:
                    st.warning("⚠️ Not enough numeric data to compute correlation.")
                else:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
                    st.pyplot(fig)

# --- Tab 3: AI Insights ---
with tabs[2]:
    st.header("🧠 AI Insights: Sentiment Analysis")

    text_input = st.text_area("Enter text for sentiment analysis", height=150)

    if st.button("📝 Analyze Sentiment"):
        if text_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing sentiment..."):
                result = sentiment_analysis(text_input)

            if "label" in result:
                label = result["label"]
                confidence = result["confidence"]
                st.markdown("### 📊 Sentiment Analysis Result")
                st.success(f"The overall sentiment is **{label}** with a confidence of **{confidence}%**.")
            elif "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                st.error("❌ Could not retrieve a valid sentiment result. Please try again.")

# --- Tab 4: History & Reports ---
with tabs[3]:
    st.header("📜 History & Reports")

    if st.session_state.cleaned_files:
        st.markdown("### 📁 Previously Cleaned Datasets")
        for file in st.session_state.cleaned_files:
            st.write(f"- `{file}`")

        if st.session_state.last_report_path:
            with open(st.session_state.last_report_path, "rb") as f:
                report_data = f.read()

            st.download_button(
                label="📄 Download Last Cleaning Report (PDF)",
                data=report_data,
                file_name=os.path.basename(st.session_state.last_report_path),
                mime="application/pdf"
            )
    else:
        st.info("📌 No cleaned datasets yet. Clean some data first!")
