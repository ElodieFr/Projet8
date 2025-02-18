import mlflow
import mlflow.pyfunc
import pandas as pd
import streamlit as st
import os
import plotly.graph_objects as go
import chardet

# Définition des chemins locaux
DATA_PATH = "data/"
MODEL_PATH = "models/model"
OPTIMAL_THRESHOLD = 0.636364

# Vérification de l'existence des fichiers
@st.cache_data()
def check_file_exists(file_path):
    return os.path.exists(file_path)

# Détection automatique de l'encodage
@st.cache_data(ttl=600)
def detect_encoding(file_path):
    if not check_file_exists(file_path):
        return "utf-8"  # Encodage par défaut si le fichier est absent
    
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read(100000))  # Analyse un grand échantillon
    return result.get("encoding", "utf-8")  # Définit UTF-8 par défaut en cas d'échec

# Chargement des données CSV avec gestion d'encodage
@st.cache_data(ttl=600)
def load_csv_data(filename):
    file_path = os.path.join(DATA_PATH, filename)
    
    if not check_file_exists(file_path):
        st.error(f"❌ Fichier introuvable : {filename}")
        return None

    try:
        encoding = detect_encoding(file_path)
        df = pd.read_csv(file_path, encoding=encoding, on_bad_lines="skip")  # Correction ici
        return df
    except Exception as e:
        st.error(f"⚠️ Erreur lors du chargement {filename} : {str(e)}")
        return None

@st.cache_data(ttl=600)
def load_feature_names():
    file_path = os.path.join(DATA_PATH, "feature_names.txt")
    if check_file_exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().splitlines()
    return []

@st.cache_resource()
def load_model():
    if check_file_exists(MODEL_PATH):
        try:
            return mlflow.sklearn.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"⚠️ Erreur lors du chargement du modèle : {str(e)}")
    return None

import urllib.request

# URL du fichier CSV (remplace par ton lien réel)
url = "https://mon-site.com/application_test.csv"
destination = os.path.join("data", "application_test.csv")

# Vérifier si le fichier existe, sinon le télécharger
if not os.path.exists(destination):
    os.makedirs("data", exist_ok=True)  # Crée le dossier 'data' s'il n'existe pas
    try:
        urllib.request.urlretrieve(url, destination)
        print("✅ Fichier téléchargé avec succès :", destination)
    except Exception as e:
        print(f"⚠️ Erreur lors du téléchargement du fichier : {str(e)}")

# Chargement des ressources
model = load_model()
feature_names = load_feature_names()
customer_data = load_csv_data("application_test.csv")
customer_data_description = load_csv_data("HomeCredit_columns_description.csv")

# Vérifier si les fichiers essentiels sont bien chargés
if customer_data is None:
    st.error("⚠️ Le fichier `application_test.csv` est introuvable ou corrompu.")
if customer_data_description is None:
    st.error("⚠️ Le fichier `HomeCredit_columns_description.csv` est introuvable ou corrompu.")
if model is None:
    st.error("⚠️ Le modèle est introuvable ou corrompu.")
if customer_data is None or model is None:
    st.stop()

# 🔥 Encodage des variables catégoriques avant prédiction
if customer_data is not None:
    # Identifier les colonnes de type objet (catégoriques)
    categorical_columns = customer_data.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_columns:
        customer_data = pd.get_dummies(customer_data, columns=categorical_columns)
        st.write("✅ Encodage des variables catégoriques effectué :", categorical_columns)

# Fonction de prédiction sécurisée
def make_prediction(input_data, model, threshold):
    input_df = pd.DataFrame([input_data])
    try:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[:, 1][0]
            return prob, "Refusé" if prob >= threshold else "Accepté"
        else:
            st.error("⚠️ Le modèle ne supporte pas `predict_proba`")
    except Exception as e:
        st.error(f"⚠️ Erreur lors de la prédiction : {str(e)}")
        return None, None

# Affichage d'une jauge avec Plotly
def gauge_chart(value, threshold):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Probabilité de défaut de paiement"},
        gauge={'axis': {'range': [0, 1]},
               'bar': {'color': "red" if value >= threshold else "green"},
               'steps': [
                   {'range': [0, threshold], 'color': "green"},
                   {'range': [threshold, 1], 'color': "red"}] }
    ))
    st.plotly_chart(fig)

# Interface principale
def main():
    st.title("📊 Credit Scoring Dashboard")
    st.header("🔍 Sélection du client")

    # Vérifier si customer_data contient bien la colonne SK_ID_CURR
    if 'SK_ID_CURR' not in customer_data.columns:
        st.error("⚠️ Erreur : La colonne `SK_ID_CURR` est absente des données.")
        st.stop()

    sk_id_curr = st.number_input("Entrez un SK_ID_CURR:",
                                 min_value=int(customer_data['SK_ID_CURR'].min()),
                                 max_value=int(customer_data['SK_ID_CURR'].max()))

    if sk_id_curr in customer_data['SK_ID_CURR'].values:
        client_data = customer_data[customer_data['SK_ID_CURR'] == sk_id_curr].iloc[0].to_dict()
        st.write("📋 **Informations du client :**")
        st.json(client_data)
        
        # Prédiction
        prob, label = make_prediction(client_data, model, OPTIMAL_THRESHOLD)
        if prob is not None:
            gauge_chart(prob, OPTIMAL_THRESHOLD)
            st.success(f"**Résultat de la prédiction : {label}**")
    else:
        st.warning("⚠️ Client non trouvé. Veuillez entrer un ID valide.")

    # Mode debug pour vérifier les fichiers
    if st.checkbox("🔍 Mode Debug"):
        st.write("Fichiers disponibles dans 'data/':", os.listdir(DATA_PATH) if check_file_exists(DATA_PATH) else "Dossier introuvable")
        st.write("Fichiers disponibles dans 'models/':", os.listdir("models/") if check_file_exists("models/") else "Dossier introuvable")
        st.write("Feature Names:", feature_names)
        st.write("Aperçu des données clients:", customer_data.head() if customer_data is not None else "Données introuvables")

if __name__ == "__main__":
    main()
