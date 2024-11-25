
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from itertools import combinations
from collections import defaultdict
import xlsxwriter
from io import BytesIO
import operator
from typing import Annotated, List, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.error("⛔ Accès refusé. Veuillez vous authentifier sur la page Presentation.")
    st.stop()  # Arrête l'exécution du script ici

# === Charger les données ===
@st.cache_data
def load_bpd_data_exploded():
    return pd.read_csv("data/bpd_exploded.csv")

@st.cache_data
def load_bpd_data():
    return pd.read_csv("data/bpd.csv")

bpd = load_bpd_data()
bpd_analytics = load_bpd_data_exploded()

# Sidebar filter for groupe selection
selected_groupe = st.sidebar.multiselect(
    "Filtre par groupe:",
    options=["All"] + sorted(bpd["Groupe"].unique()), 
    default=["All"]
)

# Sidebar filter for entity selection
selected_entities = st.sidebar.multiselect(
    "Filtre par entité:",
    options=["All"] + sorted(bpd_analytics["Entity"].unique()), 
    default=["All"]
)

# Sidebar filter for entity selection
selected_bpd = st.sidebar.multiselect(
    "Filtre par banque publique de développement & autres:",
    options=["All"] + sorted(bpd_analytics["BPD Names"].unique()), 
    default=["All"]
)

# Initialiser le DataFrame filtré

# Filtrer par groupe
if "All" not in selected_groupe:
    bpd = bpd[bpd["Groupe"].isin(selected_groupe)]
    bpd_analytics = bpd_analytics[bpd_analytics["Groupe"].isin(selected_groupe)]

# Filtrer par entité
if "All" not in selected_entities:
    bpd = bpd[bpd["Entity"].isin(selected_entities)]
    bpd_analytics = bpd_analytics[bpd_analytics["Entity"].isin(selected_groupe)]

# Filtrer par entité
if "All" not in selected_bpd:
    bpd_analytics = bpd_analytics[bpd_analytics["BPD Names"].isin(selected_bpd)]

# Filtrer par banque publique de développement
if "All" not in selected_bpd:
    def filter_bpd(cell):
        """
        Vérifie si au moins une des BPD sélectionnées est présente dans la cellule.
        """
        if pd.isna(cell):  # Gérer les valeurs NaN
            return False
        cell_bpd = [name.strip() for name in cell.split(",")]
        return any(bpd in cell_bpd for bpd in selected_bpd)
    
    bpd = bpd[bpd["BPD Names"].apply(filter_bpd)]


if bpd.empty:
    st.warning("Aucun résultat trouvé pour les filtres sélectionnés.")

# Afficher les données filtrées dans Streamlit
st.sidebar.write(f"Nombre de lignes après filtrage : {len(bpd)}")


# === Configuration du modèle OpenAI ===
@st.cache_resource
def load_model():
  return ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=st.secrets["openai_api_key"])

llm = load_model()

# === Définir les chaînes Map et Reduce ===
# Prompt pour l'étape Map (résumé par BPD)
map_template = "Write a concise summary of the following: {context}."
map_prompt = ChatPromptTemplate([("human", map_template)])
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# Prompt pour l'étape Reduce (résumé global)
reduce_template = """
The following is a set of summaries:
{docs}
Take these and distill it into a final, consolidated summary
of the main themes.
"""
reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# === Initialiser les données de session ===
if "summaries" not in st.session_state:
    st.session_state["summaries"] = None  # Pour stocker les résumés par BPD
if "global_summary" not in st.session_state:
    st.session_state["global_summary"] = None  # Pour stocker le résumé global


# === Calculs ===

# 1. Pourcentage des BPD  
top_bpd = (
    bpd_analytics.groupby(["BPD Names"])
    .size()
    .reset_index(name="Counts")
    .sort_values(by="Counts", ascending=False)
)

top_10_bpd_names = (
    top_bpd.groupby("BPD Names")["Counts"]
    .sum()
    .nlargest(10)
    .index
)
top_bpd["Pourcentage"] = top_bpd["Counts"].transform(lambda x: (x / x.sum()) * 100)
top_bpd_filtered = top_bpd[top_bpd["BPD Names"].isin(top_10_bpd_names)]
top_bpd_filtered.reset_index(drop=True, inplace=True)
top_bpd_filtered.sort_values(by="Pourcentage", ascending=False, inplace=True)
top_bpd_filtered["BPD Names"] = top_bpd_filtered.reset_index().apply(
    lambda row: f"{row.name} - {row['BPD Names']}",
    axis=1
)

# 2. Pourcentage des BPD par groupe
top_bpd = (
    bpd_analytics.groupby(["BPD Names", "Groupe"])
    .size()
    .reset_index(name="Counts")
    .sort_values(by="Counts", ascending=False)
)

top_10_bpd_names = (
    top_bpd.groupby("BPD Names")["Counts"]
    .sum()
    .nlargest(10)
    .index
)

top_bpd_filtered_group = top_bpd[top_bpd["BPD Names"].isin(top_10_bpd_names)]
top_bpd_filtered_group["Pourcentage"] = top_bpd_filtered_group.groupby("Groupe")["Counts"].transform(lambda x: (x / x.sum()) * 100)
top_bpd_filtered_group.reset_index(drop=True, inplace=True)
top_bpd_filtered_group.sort_values(by="Pourcentage", ascending=False, inplace=True)


# === Visualisations ===
st.title("📊 Analyse des Banques Publiques de Développements (BPD)")
st.markdown("---")
# === Création des onglets ===
tab1, tab2, tab3 = st.tabs(["Visualisation", "Résumé", "Table"])

# === Onglet 1: Visualisation ===
with tab1:
  row1 = st.columns(2)
  row2 = st.columns(2)

  # Graphique 1 : Pourcentage des BPD par numéro
  with row1[0]:
      st.subheader("Répartition des top 10 BPDs")
      st.bar_chart(top_bpd_filtered, x='BPD Names', y='Pourcentage', use_container_width=True,x_label="BPD Names",horizontal = True)

  # Graphique 2 : Pourcentage des BPD par Groupe et numéro
  with row1[1]:
      st.subheader("Répartition des top 10 BPDs par Groupe")
      st.bar_chart(
          top_bpd_filtered_group, 
          x='BPD Names', 
          y='Pourcentage', 
          use_container_width=True, 
          color='Groupe', 
          x_label="BPD Names"
      )


# === Streamlit Tab for Summarization ===
with tab2:
    st.markdown("## 📋 Résumé des Analyses des BPDs")
    
    # Préparation du texte pour le contexte
    context = []
    for _, row in bpd.iterrows():
        context.append(f"Rôle des Banques Publiques de Développement dans le fichier {row['Entity']} :\n{row['Role Summary']}\n\n")
    
    # Concaténer les textes pour former le contexte global
    concatenated_context = "\n".join(context)
    
    # Bouton pour générer le résumé
    if st.button("Générer le résumé"):
        # Réinitialiser les résumés dans la session state
        st.session_state["summaries"] = None
        st.session_state["global_summary"] = None

        # Générer le résumé global à partir du contexte
        def generate_global_summary(context_text):
            """
            Produit un résumé global du contexte donné.
            """
            return reduce_chain.run({"docs": context_text})

        # Générer le résumé global
        st.session_state["global_summary"] = generate_global_summary(concatenated_context)

    # Afficher le résumé global
    if st.session_state["global_summary"] is not None:
        st.subheader("Résumé global")
        st.write(st.session_state["global_summary"])
    else:
        st.info("Cliquez sur le bouton pour générer le résumé.")


with tab3:
    st.subheader("Données brutes")
    
    # Préparation des données brutes pour l'affichage et le téléchargement
    bpd.reset_index(drop=True, inplace=True)

    st.table(bpd)
