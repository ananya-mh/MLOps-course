# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import argparse
import os


if __name__ == "__main__":
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Argument parsing for hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()

    # Train a Random Forest classifier
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Accuracy calculation (y_pred vs y_test)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy on test set: {accuracy:.4f}")

    # Save model to output folder
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "iris_model.pkl")
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")
    print("The model training was successful")
