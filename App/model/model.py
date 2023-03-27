import numpy as np
from pathlib import Path
# import logging
import joblib


# logging.basicConfig(level=logging.DEBUG)

# logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve(strict=True).parent


with open(f"{BASE_DIR}/Tenserflow.joblib", "rb") as f:
    model_pipeline_loaded = joblib.load(f)

classes = {
    '10': 'Topalak',
    '0': 'Ayrik_Otu',
    '5': 'Kanyas',
    '9': 'Sirken',
    '6': 'Kopek_Uzumu',
    '8': 'Seytan_Elmasi',
    '2': 'Domuz_Pitragi',
    '4': 'Isirgan_Otu',
    '7': 'Semiz_Otu',
    '3': 'Ebegumeci',
    '1': 'Bahce_Sutlegeni'
}


def get_class_name(predicted_class):
    return classes[str(predicted_class)]


def multivariate_gaussian(X, mu, var):
    k = len(mu)

    if var.ndim == 1:
        var = np.diag(var)

    X = X - mu
    p = (2 * np.pi)**(-k/2) * np.linalg.det(var)**(-0.5) * \
        np.exp(-0.5 * np.sum(np.matmul(X, np.linalg.pinv(var)) * X, axis=1))

    return p


def isitAnamoly(x):
    epsilon = 1.0403232915170298e-12
    mu = np.array([-0.00428515, -0.00084384,  0.00388045, -0.00638636, -0.01134534,
                  -0.00560243,  0.00046675])
    var = np.array([[0.99109507, 0., 0., 0., 0.,
                     0., 0.],
                    [0., 0.98915215, 0., 0., 0.,
                     0., 0.],
                    [0., 0., 0.98942418, 0., 0.,
                     0., 0.],
                    [0., 0., 0., 0.98465041, 0.,
                     0., 0.],
                    [0., 0., 0., 0., 0.99732631,
                     0., 0.],
                    [0., 0., 0., 0., 0.,
                     1.00040413, 0.],
                    [0., 0., 0., 0., 0.,
                     0., 0.98981666]])
    if (np.ndim(x) == 1):
        x = x.reshape(1, -1)
    p = multivariate_gaussian(x, mu, var)
    return p < epsilon, (1-p)


def predict_pipeline(data: dict) -> str:
    # Preprocess the input data
    input_data = np.array(list(data.values())).reshape(1, -1)

    normalized_input = model_pipeline_loaded.named_steps["min_max_scaler"].transform(
        input_data)
    scaled_input = model_pipeline_loaded.named_steps["scaler"].transform(
        normalized_input)
    is_anomaly, probability = isitAnamoly(scaled_input)
    if (is_anomaly):
        return "Anomaly", probability
    # Make predictions
    pred = model_pipeline_loaded.predict(input_data)

    # Get the class name
    class_name = get_class_name(pred.argmax(axis=1)[0])
    print(pred)
    Confidence = pred.max(axis=1)[0]

    return class_name, Confidence
