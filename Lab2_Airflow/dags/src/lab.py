import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os


def load_data():
    """
    Loads the Mall Customers dataset from a CSV file, serializes it, and returns the serialized data.
    """
    # Adjust the file path if your dataset is in a different folder
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/Mall_Customers.csv"))
    serialized_data = pickle.dumps(df)
    return serialized_data


def data_preprocessing(data):
    """
    Deserializes data, performs preprocessing (selects relevant features, scales them),
    and returns serialized processed data.
    """
    df = pickle.loads(data)
    df = df.dropna()

    # Select numerical features for clustering
    clustering_data = df[["Annual Income (k$)", "Spending Score (1-100)"]]

    # Scale features to range [0,1]
    scaler = MinMaxScaler()
    clustering_scaled = scaler.fit_transform(clustering_data)

    # Serialize scaled data
    clustering_serialized_data = pickle.dumps(clustering_scaled)
    return clustering_serialized_data


def build_save_model(data, filename):
    """
    Builds a KMeans clustering model, saves it to a file, and returns SSE values for the elbow method.
    """
    df = pickle.loads(data)

    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }

    sse = []
    for k in range(1, 11):  # 1–10 clusters are usually enough for this dataset
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)

    # Save the last trained model
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump(kmeans, f)

    return sse


def load_model_elbow(filename, sse):
    """
    Loads a saved KMeans model, determines the optimal number of clusters (using the elbow method),
    and predicts the cluster for a new test sample.
    """
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(output_path, "rb"))

    # Example test data — can be replaced by your own test CSV
    test_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))

    # Determine optimal cluster count
    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    print(f"Optimal number of clusters: {kl.elbow}")

    # Predict cluster for test data
    predictions = loaded_model.predict(test_data)
    return predictions[0]
