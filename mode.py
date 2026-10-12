
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def train_model():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Model: Decision Tree Classifier")
    print("Dataset: Iris")
    print(f"Model Accuracy: {accuracy:.2%}")
    print("Sample Prediction:", model.predict([X_test[0]]))

    with open("accuracy.txt", "w") as file:
        file.write(f"{accuracy:.2%}")

    return model, accuracy

if __name__ == "__main__":
    train_model()
