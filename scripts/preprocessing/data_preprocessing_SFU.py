import pandas as pd
import numpy as np

# Load datasets
df_2013_2024 = pd.read_csv("../data/SFU_2013_2024.csv")
df_2024 = pd.read_csv("../data/SFU_2024.csv")

# Combine datasets
df_combined = pd.concat([df_2013_2024, df_2024], ignore_index=True)

# Function to clean and combine the dataset
def clean_and_combine_data(df):
    # Combine Finished and Completed into one column
    df["Finished"] = df["Finished"] + df["Completed"]
    df.drop(columns=["Completed"], inplace=True)

    # Fix illogical values in sequential columns
    # Ensure: Signed up >= Applied >= Accepted >= Approved >= Realized >= Finished
    df["Applied"] = df[["Signed up", "Applied"]].min(axis=1)
    df["Accepted"] = df[["Applied", "Accepted"]].min(axis=1)
    df["Approved"] = df[["Accepted", "Approved"]].min(axis=1)
    df["Realized"] = df[["Approved", "Realized"]].min(axis=1)
    df["Finished"] = df[["Realized", "Finished"]].min(axis=1)

    # Clean GPA (cap at 4.33 and floor at 0)
    df["GPA"] = df["GPA"].clip(0, 4.33)

    # Clean Length of Exchange (cap at 12 months, minimum 1 month)
    df["Length of Exchange"] = df["Length of Exchange"].clip(1, 12)

    # Clean English Proficiency (cap at 100, minimum 0)
    df["English Proficiency"] = df["English Proficiency"].clip(0, 100)

    # Clean Number of Destinations (minimum 1, maximum 5)
    df["Number of Destinations"] = df["Number of Destinations"].clip(1, 5)

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

    # GPA: Assign higher GPAs to students with more destinations and funding
    df["GPA"] = np.where(
        (df["Number of Destinations"] > 3) & (df["Funding"] == "Yes"),
        df["GPA"] * 1.1,  # Increase GPA by 10% for these students
        df["GPA"]
    ).clip(0, 4.33)  # Ensure GPA stays within bounds

    # Funding: Assign funding based on GPA and motivation
    df["Funding"] = np.where(
        (df["GPA"] > 3.5) & (df["Motivation"] == "Career"),
        "Yes", 
        "No"
    )

    # Length of Exchange: Assign longer exchanges to students with higher English proficiency
    df["Length of Exchange"] = np.where(
        df["English Proficiency"] > 80,
        df["Length of Exchange"] + 2,  # Add 2 months for high proficiency
        df["Length of Exchange"]
    ).clip(1, 12)  # Ensure length stays within bounds

    # Motivation: Assign motivation based on academic background
    career_motivated = [
        "Business administration", "Marketing", "Economics", "Finance", "Accounting", "Law",
        "International Trade", "Banking", "Public administration", "Entrepreneurship", "Human Resources"
    ]
    cultural_motivated = [
        "Arts", "Literature", "History", "Languages", "Religion", "Theatre", "Media Arts", "Graphic design"
    ]
    df["Motivation"] = np.where(
        df["Backgrounds"].isin(career_motivated),
        "Career", 
        np.where(
            df["Backgrounds"].isin(cultural_motivated),
            "Cultural", 
            "Personal Growth"
        )
    )

    # Number of Destinations: Assign more destinations to students with prior international experience
    df["Number of Destinations"] = np.where(
        df["Prior International Experience"] == "Yes",
        df["Number of Destinations"] + 1,  # Add 1 destination
        df["Number of Destinations"]
    ).clip(1, 5)  # Ensure number stays within bounds

    # English Proficiency: Assign higher proficiency to students with longer exchanges
    df["English Proficiency"] = np.where(
        df["Length of Exchange"] > 6,
        df["English Proficiency"] + 10,  # Add 10 points
        df["English Proficiency"]
    ).clip(0, 100)  # Ensure proficiency stays within bounds

    # Prior International Experience: Assign "Yes" to students with higher GPAs
    df["Prior International Experience"] = np.where(
        df["GPA"] > 3.0,
        "Yes", 
        "No"
    )

    # SFU Campus: Assign campus based on academic background
    surrey_campus = [
        "Computer sciences", "Software development and programming", "Computer engineering",
        "Electrical engineering", "Aerospace engineering", "Industrial engineering", "Material engineering",
        "Bioengineering", "Chemical engineering", "Electronics engineering", "Systems and Computing Engineering"
    ]
    burnaby_campus = [
        "Business administration", "Marketing", "Economics", "Finance", "Accounting", "Law",
        "International Trade", "Banking", "Public administration", "Entrepreneurship", "Human Resources"
    ]
    df["SFU Campus"] = np.where(
        df["Backgrounds"].isin(surrey_campus),
        "Surrey", 
        np.where(
            df["Backgrounds"].isin(burnaby_campus),
            "Burnaby", 
            "Vancouver"
        )
    )

    # Co-op Before Exchange: Assign "Yes" to students with higher GPAs and career motivation
    df["Co-op Before Exchange"] = np.where(
        (df["GPA"] > 3.0) & (df["Motivation"] == "Career"),
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
        "Gender", "GPA", "Funding", "Length of Exchange", "Motivation", 
        "Number of Destinations", "English Proficiency", "Prior International Experience", 
        "SFU Campus", "Co-op Before Exchange"
    ]
    for col in categorical_columns:
        mode_values = df.groupby("Backgrounds")[col].agg(lambda x: x.mode()[0]).reset_index()
        df_aggregated = df_aggregated.merge(mode_values, on="Backgrounds", how="left")

    return df_aggregated

# Aggregate the cleaned dataset
df_aggregated = aggregate_data(df_combined_cleaned)

# Save the cleaned combined dataset
df_aggregated.to_csv("../data/cleaned/SFU_Aggregated_Cleaned.csv", index=False)

print("Combined cleaned dataset saved as 'data/cleaned/SFU_Aggregated_Cleaned.csv'!")