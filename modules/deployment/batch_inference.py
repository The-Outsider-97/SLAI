import pandas as pd
from modules.deployment.model_deployer import ModelDeployer

def run_batch_predictions(csv_path, output_path, model_path):
    deployer = ModelDeployer(model_path)
    deployer.load()

    df = pd.read_csv(csv_path)
    predictions = []

    for _, row in df.iterrows():
        input_data = row.values.tolist()
        result = deployer.predict(input_data)
        predictions.append(result.get("prediction", -1))

    df["prediction"] = predictions
    df.to_csv(output_path, index=False)
    print(f"Batch predictions saved to: {output_path}")

if __name__ == "__main__":
    run_batch_predictions(
        csv_path="data/batch_input.csv",
        output_path="data/batch_output.csv",
        model_path="models/random_forest_model.pkl"
    )
