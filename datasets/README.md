# Datasets for SLAI

## Folder Structure
- `synthetic_data.csv` - Example user interaction data.
- `validation_data.csv` - Evaluation dataset.
- `/loaders/dataset_loader.py` - Loads datasets into Pandas DataFrames.

## Expected Columns
| Column   | Description                   |
|----------|-------------------------------|
| user_id  | Unique user identifier        |
| gender   | Sensitive attribute           |
| age      | Demographic attribute         |
| action   | User action (click, ignore)   |
| reward   | Feedback reward (0 or 1)      |
