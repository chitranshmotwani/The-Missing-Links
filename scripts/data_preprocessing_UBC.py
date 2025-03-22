import pandas as pd
import numpy as np

# Load datasets
df_2013_2024 = pd.read_csv("../data/UBC_2013_2024.csv")
df_2024 = pd.read_csv("../data/UBC_2024.csv")

# Combine datasets
df_combined = pd.concat([df_2013_2024, df_2024], ignore_index=True)

# Function to clean and combine the dataset
def clean_and_combine_data(df):
    # Combine Finished and Completed into one column
    df["Finished"] = df["Finished"] + df["Completed"]
    df.drop(columns=["Completed"], inplace=True)

    # Fix illogical values in sequential columns
    # Ensure: Signed up >= Applied >= Accepted >= Approved >= Realized >= Finished
    # df["Applied"] = df[["Signed up", "Applied"]].min(axis=1)
    # df["Accepted"] = df[["Applied", "Accepted"]].min(axis=1)
    # df["Approved"] = df[["Accepted", "Approved"]].min(axis=1)
    # df["Realized"] = df[["Approved", "Realized"]].min(axis=1)
    # df["Finished"] = df[["Realized", "Finished"]].min(axis=1)

    # Clean GPA (cap at 4.33 and floor at 0)
    df["GPA"] = df["GPA"].clip(0, 4.33)

    # Modify randomly generated columns to create realistic trends
    # Gender: Assign gender based on academic background
    male_dominant = [
        "Computer sciences", "Chemical engineering", "Mathematics", "Electronics engineering",
        "Mechanical engineering", "Systems and Computing Engineering", "Software development and programming",
        "Computer engineering", "Electrical engineering", "Aerospace engineering", "Industrial engineering",
        "Material engineering", "Bioengineering", "Physics", "Chemistry"
    ]
    female_dominant = [
        "Education", "Psychology", "Biology", "Literature", "Sociology", "Languages", "Graphic design",
        "Media Arts", "Theatre", "Social Work", "Nursing", "Health Science", "Public relations", "Religion"
    ]
    df["Gender"] = np.where(
        df["Backgrounds"].isin(male_dominant), 
        "Male", 
        np.where(
            df["Backgrounds"].isin(female_dominant),
            "Female", 
            np.random.choice(["Male", "Female"])  # Random for other backgrounds
        )
    )

    # Funding: Assign funding based on GPA and motivation
    df["Funding"] = np.where(
        (df["GPA"] > 3.5),
        "Yes", 
        "No"
    )

    return df

# Clean the combined dataset
df_combined_cleaned = clean_and_combine_data(df_combined)

# Group by Backgrounds and aggregate columns
def aggregate_data(df):
    # Sum numerical columns
    numerical_columns = ["Signed up", "Applied", "Accepted", "Approved", "Realized", "Finished"]
    df_aggregated = df.groupby("Backgrounds", as_index=False)[numerical_columns].sum()

    # For categorical columns, take the mode (most frequent value)
    categorical_columns = [
        "Gender", "GPA", "Funding"
    ]
    for col in categorical_columns:
        mode_values = df.groupby("Backgrounds")[col].agg(lambda x: x.mode()[0]).reset_index()
        df_aggregated = df_aggregated.merge(mode_values, on="Backgrounds", how="left")

    return df_aggregated

# Aggregate the cleaned dataset
df_aggregated = aggregate_data(df_combined_cleaned)

# Save the cleaned combined dataset
df_aggregated.to_csv("../data/cleaned/UBC_Aggregated_Cleaned.csv", index=False)

print("Combined cleaned dataset saved as 'UBC_Aggregated_Cleaned.csv'!")