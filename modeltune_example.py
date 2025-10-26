#Example if you want to use Ray Tune for hyperparameter tuning with Ultralytics YOLO
from ray import tune
import ray
from ultralytics import YOLO
import os

ctx = ray.init(include_dashboard=True, dashboard_host="127.0.0.1")
print("Ray dashboard URL:", ctx.dashboard_url)

    # Load YOLO model weights
model = YOLO('yolo11n.pt')

# Absolute path to your data.yaml file (update path accordingly)
data_path = os.path.abspath("Data/data.yaml")

# Define the hyperparameter search space for tuning
search_space = {
    "lr0": tune.uniform(1e-4, 1e-2),     # learning rate
    "momentum": tune.uniform(0.8, 0.95)  # SGD momentum
    # Add more hyperparameters as needed
}

# Run hyperparameter tuning with Ray using Ultralytics YOLO's tune method
result = model.tune(
    data=data_path,        # Dataset YAML path
    space=search_space,    # Hyperparameter search space
    epochs=10,             # Number of training epochs
    use_ray=True           # Enable Ray Tune for hyperparameter optimization
)
best_result = result.get_best_result(metric="metrics/mAP50(B)", mode="max")


print("Best config:", best_result.config)

print("Best mAP:", best_result.metrics["metrics/mAP50(B)"])


checkpoint = getattr(best_result, "checkpoint", None)
if checkpoint:
    print("Best checkpoint path:", checkpoint.path)
else:
    print("No checkpoint found — only config available.")
