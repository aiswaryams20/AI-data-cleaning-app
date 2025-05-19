from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

def generate_cleaning_report_pdf(original_df, cleaned_df, columns_dropped, missing_value_actions, file_path):
    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter
    textobject = c.beginText(40, height - 50)
    textobject.setFont("Helvetica-Bold", 16)

    # Title
    textobject.textLine("📊 Data Cleaning Summary Report")
    textobject.setFont("Helvetica", 12)
    textobject.textLine("=" * 80)
    textobject.textLine(f"Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    textobject.textLine("")

    # Introduction
    textobject.textLine("📝 What is Data Cleaning?")
    textobject.textLine("Data cleaning is the process of identifying and fixing problems in a dataset.")
    textobject.textLine("It ensures the data is reliable, consistent, and ready for analysis or machine learning.")
    textobject.textLine("In this report, we summarize the cleaning steps performed on your dataset.")
    textobject.textLine("")

    # Dataset Shapes
    textobject.setFont("Helvetica-Bold", 13)
    textobject.textLine("📏 Dataset Overview")
    textobject.setFont("Helvetica", 12)
    textobject.textLine(f"Original Dataset Shape: {original_df.shape}")
    textobject.textLine(f"Cleaned Dataset Shape:  {cleaned_df.shape}")
    textobject.textLine("")

    # Columns Dropped
    textobject.setFont("Helvetica-Bold", 13)
    textobject.textLine("🗑️ Columns Dropped")
    textobject.setFont("Helvetica", 12)
    if columns_dropped:
        textobject.textLine("The following columns were removed as they were either irrelevant or unhelpful:")
        for col in columns_dropped:
            textobject.textLine(f"- {col}")
    else:
        textobject.textLine("No columns were dropped from this dataset.")
    textobject.textLine("")

    # Missing Values Handled
    textobject.setFont("Helvetica-Bold", 13)
    textobject.textLine("🔍 Missing Values Handled")
    textobject.setFont("Helvetica", 12)
    if missing_value_actions:
        textobject.textLine("Missing values were handled as follows:")
        for col, action in missing_value_actions.items():
            textobject.textLine(f"- {col}: {action}")
    else:
        textobject.textLine("No missing values were found in the dataset.")
    textobject.textLine("")

    # Summary of Numeric Columns
    numeric_cols = cleaned_df.select_dtypes(include=['number'])
    if not numeric_cols.empty:
        textobject.setFont("Helvetica-Bold", 13)
        textobject.textLine("📊 Numeric Columns Summary")
        textobject.setFont("Helvetica", 12)
        textobject.textLine(f"Total Numeric Columns: {numeric_cols.shape[1]}")
        summary = numeric_cols.describe().round(2)
        for col in summary.columns:
            textobject.textLine(f"\n{col}:")
            textobject.textLine(f"  Min: {summary[col]['min']}")
            textobject.textLine(f"  Max: {summary[col]['max']}")
            textobject.textLine(f"  Mean: {summary[col]['mean']}")
            textobject.textLine(f"  Std Dev: {summary[col]['std']}")
    else:
        textobject.textLine("No numeric columns available for summary.")
    textobject.textLine("")

    # Final Note
    textobject.setFont("Helvetica-Bold", 13)
    textobject.textLine("✅ Final Remarks")
    textobject.setFont("Helvetica", 12)
    textobject.textLine("The dataset has been cleaned and is now ready for further analysis or modeling.")
    textobject.textLine("Thank you for using our Data Cleaning Tool!")
    textobject.textLine("")

    c.drawText(textobject)
    c.showPage()
    c.save()

    return file_path
