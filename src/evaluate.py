import os
import sys
dir_store = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.utils.store_utils import StorageHandler
from src.utils.anomaly_detection import AnomalyDetector1D
store_handler = StorageHandler(dir_store=dir_store)
import warnings
warnings.filterwarnings('ignore')

target = "Class"
exclude_cols = ["Class"]
metric_type = "acc"
metric_opt = {
    "acc": "accuracy_max"
}
d = {
    "batch_normalization": True,
    "layer_normalization": False,
    "num_epochs": 50,
    "verbose": 2,
    "mini_batch_size": 3000,
    "learning_rate": 0.001,
    "beta1": 0.5,
    "layers": [32, 16, 8],
    "use_bias": False,
    "kernel_initializer": "he_uniform",
    "bias_initializer": "zeros",
    "activation": "leaky_relu",
    "activation_config": {"alpha": 0.2},
    "dropout": False
}


def evaluate():

    X_train = store_handler.read("X_train.p")
    X_test_Legit = store_handler.read("X_test_Legit.p")
    X_test_Fraud = store_handler.read("X_test_Fraud.p")

    result = AnomalyDetector1D(config=d).train(
        X_train, X_test_Legit=X_test_Legit, X_test_Fraud=X_test_Fraud, validation=True
    )

    pred_test_Legit = result['pred_test_Legit']
    pred_test_Fraud = result['pred_test_Fraud']

    store_handler.store(pred_test_Legit, "pred_test_Legit.p")
    store_handler.store(pred_test_Fraud, "pred_test_Fraud.p")





if __name__ == '__main__':
    evaluate()
