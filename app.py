"" 
import mlflow
import mlflow.pyfunc
import pandas as pd
import streamlit as st
import os
import plotly.graph_objects as go

# Définition des chemins locaux
DATA_PATH = "data/"
MODEL_PATH = "models/model"

# Fonction pour vérifier si un fichier existe
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        st.error(f"⚠️ Fichier introuvable : {file_path}")
        return False
    return True

# Chargement des données avec gestion des erreurs
@st.cache_data()
def load_test_data():
    """Charge les données de test en local"""
    file_path = os.path.join(DATA_PATH, "application_test.csv")
    if check_file_exists(file_path):
        return pd.read_csv(file_path)
    return None

@st.cache_data()
def load_test_data_description():
    """Charge la description des données en local"""
    file_path = os.path.join(DATA_PATH, "HomeCredit_columns_description.csv")
    if check_file_exists(file_path):
        return pd.read_csv(file_path)
    return None

@st.cache_data()
def retrieve_feature_names():
    """Charge les noms des features en local"""
    file_path = os.path.join(DATA_PATH, "feature_names.txt")
    if check_file_exists(file_path):
        with open(file_path, "r") as f:
            return f.read().splitlines()
    return []

@st.cache_resource()
def load_model():
    """Charge le modèle MLflow en local"""
    if check_file_exists(MODEL_PATH):
        try:
            return mlflow.sklearn.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"⚠️ Erreur lors du chargement du modèle : {str(e)}")
            return None
    return None

# Chargement des données et du modèle
model = load_model()
feature_names = retrieve_feature_names()
customer_data = load_test_data()
customer_data_description = load_test_data_description()
optimal_threshold = 0.636364

# Vérifier si les fichiers sont bien chargés avant de continuer
if customer_data is None or model is None:
    st.error("⚠️ Impossible de charger les données ou le modèle. Vérifiez que tous les fichiers sont disponibles.")
    st.stop()  # Arrête l'exécution de Streamlit

# Fonction de prédiction
def make_prediction(input_data, model, optimal_threshold):
    """Effectue une prédiction et renvoie la probabilité et l'étiquette"""
    input_df = pd.DataFrame([input_data])

    try:
        probability_class1 = model.predict_proba(input_df)[:, 1][0]
        prediction_label = "Refusé" if probability_class1 >= optimal_threshold else "Accepté"
        return probability_class1, prediction_label
    except Exception as e:
        st.error(f"⚠️ Erreur lors de la prédiction : {str(e)}")
        return None, None

# Fonction pour afficher une jauge avec Plotly
def gauge_chart(value, threshold=0.636364):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Probabilité de défaut de paiement"},
        gauge={'axis': {'range': [0, 1]},
               'bar': {'color': "red" if value >= threshold else "green"},
               'steps': [
                   {'range': [0, threshold], 'color': "green"},
                   {'range': [threshold, 1], 'color': "red"}]}
    ))
    st.plotly_chart(fig)

# Interface Streamlit
def main():
    st.title("📊 Credit Scoring Dashboard")

    st.header("🔍 Sélection du client")
    sk_id_curr = st.number_input("Entrez un SK_ID_CURR:", 
                                 min_value=int(customer_data['SK_ID_CURR'].min()), 
                                 max_value=int(customer_data['SK_ID_CURR'].max()))

    if sk_id_curr in customer_data['SK_ID_CURR'].values:
        client_data = customer_data[customer_data['SK_ID_CURR'] == sk_id_curr].iloc[0].to_dict()
        st.write(f"📋 **Informations du client :**")
        st.json(client_data)

        # Prédiction
        prob, label = make_prediction(client_data, model, optimal_threshold)

        if prob is not None:
            gauge_chart(prob, optimal_threshold)
            st.success(f"**Résultat de la prédiction : {label}**")
        else:
            st.error("⚠️ Une erreur s'est produite lors de la prédiction.")

    else:
        st.warning("⚠️ Client non trouvé. Veuillez entrer un ID valide.")

if __name__ == "__main__":
    main()