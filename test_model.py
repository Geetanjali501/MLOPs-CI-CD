
from model import train_model
from sklearn.datasets import load_iris

def test_model_accuracy():
    model, accuracy = train_model()
    assert 0 <= accuracy <= 1
    assert accuracy >= 0.80

def test_model_prediction():
    model, accuracy = train_model()
    iris = load_iris()
    prediction = model.predict([iris.data[0]])
    assert prediction[0] in [0, 1, 2]
