
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
def load_actions_data():
    return pd.read_csv("data/addis_ababa.csv")

actions = load_actions_data()

# Sidebar filter for groupe selection
selected_groupe = st.sidebar.multiselect(
    "Filtre par groupe:",
    options=["All"] + sorted(actions["Groupe"].unique()), 
    default=["All"]
)

# Sidebar filter for entity selection
selected_entities = st.sidebar.multiselect(
    "Filtre par entité:",
    options=["All"] + sorted(actions["Entity"].unique()), 
    default=["All"]
)

# Sidebar filter for Actions aera selection
selected_actions = st.sidebar.multiselect(
    "Filtre par Action aeras:",
    options=["All"] + sorted(actions["Action Area"].unique()), 
    default=["All"]
)

# Initialiser le DataFrame filtré

# Filtrer par groupe
if "All" not in selected_groupe:
    actions = actions[actions["Groupe"].isin(selected_groupe)]

# Filtrer par entité
if "All" not in selected_entities:
    actions = actions[actions["Entity"].isin(selected_entities)]

# Filtrer par actions number
if "All" not in selected_actions:
    actions = actions[actions["Action Area"].isin(selected_actions)]

if actions.empty:
    st.warning("Aucun résultat trouvé pour les filtres sélectionnés.")

# Afficher les données filtrées dans Streamlit
st.sidebar.write(f"Nombre de lignes après filtrage : {len(actions)}")


# === Configuration du modèle OpenAI ===
@st.cache_resource
def load_model():
  return ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=st.secrets["openai_api_key"])

llm = load_model()

# === Définir les chaînes Map et Reduce ===
# Prompt pour l'étape Map (résumé par actions)
map_template = "Write a concise summary of the following: {context}."
map_prompt = ChatPromptTemplate([("human", map_template)])
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# Prompt pour l'étape Reduce (résumé global)
reduce_template = """
Voici un ensemble de résumés :
{docs}
Prenez ces résumés et distillez-les en un résumé final consolidé 
des principaux thèmes abordés.
"""
reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# === Initialiser les données de session ===
if "summaries" not in st.session_state:
    st.session_state["summaries"] = None  # Pour stocker les résumés par actions
if "global_summary" not in st.session_state:
    st.session_state["global_summary"] = None  # Pour stocker le résumé global


# === Calculs ===

# 1. Pourcentage des actions count par actions number
actions_count_sum = actions.groupby('Action Area')['Mention count'].sum()
total_actions_count = actions['Mention count'].sum()
actions_count_percentage = (actions_count_sum / total_actions_count) * 100
df_actions_count_percentage = actions_count_percentage.reset_index()
df_actions_count_percentage.rename(columns={'Mention count': 'Pourcentage'}, inplace=True)
df_actions_count_percentage.sort_values(by='Pourcentage', ascending=False, inplace=True)

# 2. Pourcentage des actions count par actions number et par Groupe
actions_count_grouped = actions.groupby(['Groupe', 'Action Area'])['Mention count'].sum().reset_index()
actions_count_grouped['Pourcentage'] = actions_count_grouped.groupby('Groupe')['Mention count'].transform(lambda x: (x / x.sum()) * 100)

# 3. Nombre total d'actions count par Entity (les 10 principales)
entity_actions_count = actions.groupby('Entity')['Mention count'].sum().sort_values(ascending=False).head(10).reset_index()
entity_actions_count["Entity"] = entity_actions_count.reset_index().apply(
    lambda row: f"{row.name} - {row['Entity']}",
    axis=1
)


# === Visualisations ===
st.title("📊 Analyse des Actions aeras Addis Abeba")
st.markdown("---")
# === Création des onglets ===
tab1, tab2, tab3 = st.tabs(["Visualisation", "Résumé", "Table"])

# === Onglet 1: Visualisation ===
with tab1:
  row1 = st.columns(2)
  row2 = st.columns(2)

  # Graphique 1 : Pourcentage des actions par numéro
  with row1[0]:
      st.subheader("Répartition des actions")
      st.bar_chart(df_actions_count_percentage, x='Action Area', y='Pourcentage', use_container_width=True,x_label="Actions aeras")

  # Graphique 2 : Pourcentage des actions par Groupe et numéro
  with row1[1]:
      st.subheader("Répartition des actions par Groupe")
      st.bar_chart(
          actions_count_grouped, 
          x='Action Area', 
          y='Pourcentage', 
          use_container_width=True, 
          color='Groupe', 
          x_label="N° actions"
      )

  # Graphique 3 : Nombre total d'actions par Entity
  with row2[0]:
      st.subheader("Top 10 contributeur")
      # Utiliser Matplotlib pour garantir un tri correct
      st.bar_chart(
      entity_actions_count, 
      x='Entity', 
      y='Mention count', 
      use_container_width=True, 
      x_label="Contributeur",
      y_label="Nombre d'occurence actions",
      horizontal=True
      )

  with row2[1]:
      if len(actions["Action Area"].unique()) >1:
        # Charger les données d'actions (simulé ici comme actions_clean pour un exemple rapide)
        actions_clean = actions[["Action Area", "Mention count", "Entity"]]

        # Calculer le pourcentage de 'actions count' par 'actions number' et par 'Entity'
        actions_clean['Percentage'] = actions_clean.groupby('Entity')['Mention count'].transform(lambda x: (x / x.sum()) * 100)

        # Filtrer pour garder les 5 premiers pourcentages par 'Entity'
        top5_actionss_per_entity = actions_clean.groupby('Entity').apply(lambda x: x.nlargest(5, 'Percentage')).reset_index(drop=True)

        # Calculer le pourcentage global des 'actions count'
        total_actions_count_filtered = top5_actionss_per_entity['Mention count'].sum()
        overall_actions_counts = top5_actionss_per_entity.groupby('Action Area')['Mention count'].sum().reset_index()
        overall_actions_counts['Overall Percentage'] = (overall_actions_counts['Mention count'] / total_actions_count_filtered) * 100

        # Identifier les 5 'actions numbers' les plus représentés globalement
        top5_actionss_global = overall_actions_counts.nlargest(5, 'Overall Percentage')['Action Area'].tolist()

        # Filtrer le DataFrame pour inclure uniquement les Top 5 actionss
        filtered_data = top5_actionss_per_entity[top5_actionss_per_entity['Action Area'].isin(top5_actionss_global)]

        # Calculer les co-occurrences
        co_occurrence = defaultdict(int)
        entity_actions = filtered_data.groupby('Entity')['Action Area'].apply(list)

        for actions_ in entity_actions:
            unique_actions = sorted(set(actions_))
            for pair in combinations(unique_actions, 2):
                co_occurrence[pair] += 1

        # Convertir en DataFrame
        co_occurrence_df = pd.DataFrame([
            {'actions_1': pair[0], 'actions_2': pair[1], 'Weight': weight}
            for pair, weight in co_occurrence.items()
        ])
        co_occurrence_df['Weight'] = co_occurrence_df['Weight'].apply(lambda x: x / co_occurrence_df['Weight'].sum() * 100)

        # Créer le graphe de réseau avec NetworkX
        G = nx.Graph()

        # Ajouter les nœuds
        for actions_ in top5_actionss_global:
            G.add_node(actions_)

        # Ajouter les arêtes avec poids
        for _, row in co_occurrence_df.iterrows():
            G.add_edge(row['actions_1'], row['actions_2'], weight=row['Weight'])

        # Calculer les positions des nœuds
        pos = nx.spring_layout(G, k=0.5, seed=42)

        # Extraire les coordonnées des nœuds et des arêtes
        edge_x = []
        edge_y = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        # Création du graphique avec Plotly
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=[str(node) for node in G.nodes()],
            textposition="top center",
            marker=dict(
                size=[G.degree[node] * 5 for node in G.nodes()],  # Taille basée sur le degré
                color='skyblue',
                line=dict(width=2)
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace])

        # Configuration du layout
        fig.update_layout(
            title="Graphe de Réseau Interactif des 5 actions les Plus Représentés",
            titlefont_size=16,
            showlegend=False,
            margin=dict(b=0, l=0, r=0, t=40),
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        )

        # Affichage avec Streamlit
        st.plotly_chart(fig, use_container_width=True)


# === Streamlit Tab for Summarization ===
with tab2:
    st.markdown("## 📋 Résumé des Analyses des Action aeras")
    # Réinitialiser les résumés dans la session state
    st.session_state["summaries"] = None
    st.session_state["global_summary"] = None
    # Prendre le top 5 des actionss les plus fréquents
    # top_5_actions = (
    #     actions.groupby("Action Area")["Mention count"]
    #     .sum()
    #     .index.tolist()
    # )

    # Filtrer uniquement les actionss dans le top 5
    # actions_filtered = actions[actions["Action Area"].isin(top_5_actions)]
    # Préparation des données par actions
    actions_grouped = actions[~actions["Summary"].isna()]
    actions_grouped = actions_grouped.groupby("Action Area")  # `actions` est le DataFrame chargé
    contents = [
        " ".join(group["Summary"].tolist()) for _, group in actions_grouped
    ]

    # Bouton pour générer le résumé
    if st.button("Générer le résumé"):
        

        # Définir l'état initial
        initial_state = {"contents": contents}

        # Fonction pour exécuter le graphe synchrone et collecter les résumés
        def generate_summaries(state):
            """
            Exécute le graphe de manière synchrone et collecte les résultats.
            """
            summaries = []
            total_steps = len(state["contents"])
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, content in enumerate(state["contents"]):
                # Exécution du résumé pour un contenu
                step_result = map_chain.run({"context": content})
                summaries.append(step_result)

                # Mise à jour de la barre de progression
                progress_bar.progress((i + 1) / total_steps)
                status_text.text(f"Résumé généré pour {i + 1}/{total_steps} actions.")

            # Nettoyer la barre de progression
            progress_bar.empty()
            status_text.empty()

            return summaries

        # Générer les résumés par actions
        st.session_state["summaries"] = generate_summaries(initial_state)

        # Générer le résumé global
        def generate_global_summary(summaries):
            """
            Combine les résumés pour produire un résumé global.
            """
            combined_summaries = "\n".join(summaries)
            result = reduce_chain.run({"docs": combined_summaries})
            return result

        st.session_state["global_summary"] = generate_global_summary(st.session_state["summaries"])

    # Afficher les résumés stockés dans la session state
    if st.session_state["summaries"] is not None:
        st.subheader("Résumé par action aeras")
        for Action_Area, summary in zip(actions.groupby("Action Area").groups.keys(), st.session_state["summaries"]):
            st.write(f"**actions {Action_Area}** : {summary}")

    if st.session_state["global_summary"] is not None:
        st.subheader("Résumé global")
        st.write(st.session_state["global_summary"])
    else:
        st.info("Cliquez sur le bouton pour générer les résumés.")

with tab3:
    st.subheader("Données brutes")
    
    # Préparation des données brutes pour l'affichage et le téléchargement
    actions.reset_index(drop=True, inplace=True)

    st.table(actions)
