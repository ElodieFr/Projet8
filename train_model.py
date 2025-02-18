import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Création d'un jeu de données
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement d'un modèle simple
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Enregistrer le modèle avec MLflow
mlflow.sklearn.save_model(model, "models/model")

print("✅ Modèle MLflow enregistré dans 'models/model'")