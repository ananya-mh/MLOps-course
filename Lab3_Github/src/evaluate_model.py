import pickle, os, json, random
from sklearn.metrics import f1_score
import joblib, glob, sys
import argparse
from sklearn.datasets import make_classification

sys.path.insert(0, os.path.abspath('..'))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    
    # Access the timestamp
    timestamp = args.timestamp
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(project_root, "models")

    model_version = f'model_{timestamp}_dt_model'
    model_path = os.path.join(models_dir, f'{model_version}.joblib')

    try:
        model = joblib.load(model_path)
    except:
        raise ValueError('Failed to catching the latest model')
        
    try:
        # Check if the file exists within the folder
        X, y = make_classification(
                            n_samples=random.randint(0, 2000),
                            n_features=6,
                            n_informative=3,
                            n_redundant=0,
                            n_repeated=0,
                            n_classes=2,
                            random_state=0,
                            shuffle=True,
                        )
    except:
        raise ValueError('Failed to catching the data')
    
    y_predict = model.predict(X)
    metrics = {"F1_Score":f1_score(y, y_predict)}
    
    # Save metrics to a JSON file

    metrics_dir = os.path.join(os.path.dirname(__file__), "../metrics/")
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_path = os.path.join(metrics_dir, f"{timestamp}_metrics.json")
    with open(metrics_path, 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)
               
    
