import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(df, columns_to_drop):
    original_df = df.copy()
    columns_dropped = []
    missing_value_actions = {}

    # Drop user-selected columns
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
    columns_dropped.extend([col for col in columns_to_drop if col in original_df.columns])

    # Handle missing values
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    non_numeric_cols = df.select_dtypes(exclude=['int64', 'float64']).columns

    # Fill numeric columns with median
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            missing_value_actions[col] = f"Filled with median ({median_value})"

    # Fill non-numeric columns with mode
    for col in non_numeric_cols:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode().iloc[0]
            df[col] = df[col].fillna(mode_value)
            missing_value_actions[col] = f"Filled with mode ({mode_value})"

    # Encode categorical variables
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = label_encoder.fit_transform(df[col])

    return df, original_df, columns_dropped, missing_value_actions


def suggest_columns_to_drop(df, missing_threshold=0.95):
    suggestions = []

    for col in df.columns:
        # Unique values equal to number of rows → likely an ID column
        if df[col].nunique() == df.shape[0]:
            suggestions.append((col, "Unique values in every row (likely an ID)"))

        # Columns with high missing value ratio
        missing_ratio = df[col].isnull().sum() / df.shape[0]
        if missing_ratio >= missing_threshold:
            suggestions.append((col, f"{missing_ratio * 100:.1f}% missing values"))

        # Columns with only one unique value
        if df[col].nunique() == 1:
            suggestions.append((col, "Only one unique value"))

    return suggestions
