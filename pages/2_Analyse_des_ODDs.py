
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
def load_odd_data():
    return pd.read_csv("data/odd.csv")

odd = load_odd_data()

# Sidebar filter for groupe selection
selected_groupe = st.sidebar.multiselect(
    "Filtre par groupe:",
    options=["All"] + sorted(odd["Groupe"].unique()), 
    default=["All"]
)

# Sidebar filter for entity selection
selected_entities = st.sidebar.multiselect(
    "Filtre par entité:",
    options=["All"] + sorted(odd["Entity"].unique()), 
    default=["All"]
)

# Sidebar filter for ODD number selection
selected_odds = st.sidebar.multiselect(
    "Filtre par ODD:",
    options=["All"] + sorted(odd["ODD number"].unique()), 
    default=["All"]
)

# Initialiser le DataFrame filtré

# Filtrer par groupe
if "All" not in selected_groupe:
    odd = odd[odd["Groupe"].isin(selected_groupe)]

# Filtrer par entité
if "All" not in selected_entities:
    odd = odd[odd["Entity"].isin(selected_entities)]

# Filtrer par ODD number
if "All" not in selected_odds:
    odd = odd[odd["ODD number"].isin(selected_odds)]

if odd.empty:
    st.warning("Aucun résultat trouvé pour les filtres sélectionnés.")

# Afficher les données filtrées dans Streamlit
st.sidebar.write(f"Nombre de lignes après filtrage : {len(odd)}")


# === Configuration du modèle OpenAI ===
@st.cache_resource
def load_model():
  return ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=st.secrets["openai_api_key"])

llm = load_model()

# === Définir les chaînes Map et Reduce ===
# Prompt pour l'étape Map (résumé par ODD)
map_template = "Write a concise summary of the following: {context}."
map_prompt = ChatPromptTemplate([("human", map_template)])
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# Prompt pour l'étape Reduce (résumé global)
reduce_template = """
Voici un ensemble de résumés :
{docs}
Prenez ces résumés et distillez-les en un résumé final consolidé 
des principaux thèmes abordés. Le résumé final doit être en Français.
"""
reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# === Initialiser les données de session ===
if "summaries" not in st.session_state:
    st.session_state["summaries"] = None  # Pour stocker les résumés par ODD
if "global_summary" not in st.session_state:
    st.session_state["global_summary"] = None  # Pour stocker le résumé global


# === Calculs ===

# 1. Pourcentage des ODD count par ODD number
odd_count_sum = odd.groupby('ODD number')['ODD count'].sum()
total_odd_count = odd['ODD count'].sum()
odd_count_percentage = (odd_count_sum / total_odd_count) * 100
df_odd_count_percentage = odd_count_percentage.reset_index()
df_odd_count_percentage.rename(columns={'ODD count': 'Pourcentage'}, inplace=True)
df_odd_count_percentage.sort_values(by='Pourcentage', ascending=False, inplace=True)

# 2. Pourcentage des ODD count par ODD number et par Groupe
odd_count_grouped = odd.groupby(['Groupe', 'ODD number'])['ODD count'].sum().reset_index()
odd_count_grouped['Pourcentage'] = odd_count_grouped.groupby('Groupe')['ODD count'].transform(lambda x: (x / x.sum()) * 100)

# 3. Nombre total d'ODD count par Entity (les 10 principales)
entity_odd_count = odd.groupby('Entity')['ODD count'].sum().sort_values(ascending=False).head(10).reset_index()
entity_odd_count["Entity"] = entity_odd_count.reset_index().apply(
    lambda row: f"{row.name} - {row['Entity']}",
    axis=1
)


# === Visualisations ===
st.title("📊 Analyse des Objectifs de Développement Durable (ODD)")
st.markdown("---")
# === Création des onglets ===
tab1, tab2, tab3 = st.tabs(["Visualisation", "Résumé", "Table"])

# === Onglet 1: Visualisation ===
with tab1:
  row1 = st.columns(2)
  row2 = st.columns(2)

  # Graphique 1 : Pourcentage des ODD par numéro
  with row1[0]:
      st.subheader("Répartition des ODDs ")
      st.bar_chart(df_odd_count_percentage, x='ODD number', y='Pourcentage', use_container_width=True,x_label="N° ODD")

  # Graphique 2 : Pourcentage des ODD par Groupe et numéro
  with row1[1]:
      st.subheader("Répartition des ODD par Groupe")
      st.bar_chart(
          odd_count_grouped, 
          x='ODD number', 
          y='Pourcentage', 
          use_container_width=True, 
          color='Groupe', 
          x_label="N° ODD"
      )

  # Graphique 3 : Nombre total d'ODD par Entity
  with row2[0]:
      st.subheader("Top 10 contributeur")
      # Utiliser Matplotlib pour garantir un tri correct
      st.bar_chart(
      entity_odd_count, 
      x='Entity', 
      y='ODD count', 
      use_container_width=True, 
      x_label="Contributeur",
      y_label="Nombre d'occurence ODD",
      horizontal=True
      )

  with row2[1]:
      if len(odd["ODD number"].unique()) >1:
        # Charger les données d'ODD (simulé ici comme odd_clean pour un exemple rapide)
        odd_clean = odd[["ODD number", "ODD count", "Entity"]]

        # Assurer que 'ODD number' est entier
        odd_clean['ODD number'] = odd_clean['ODD number'].astype(int)

        # Calculer le pourcentage de 'ODD count' par 'ODD number' et par 'Entity'
        odd_clean['Percentage'] = odd_clean.groupby('Entity')['ODD count'].transform(lambda x: (x / x.sum()) * 100)

        # Filtrer pour garder les 5 premiers pourcentages par 'Entity'
        top5_odds_per_entity = odd_clean.groupby('Entity').apply(lambda x: x.nlargest(5, 'Percentage')).reset_index(drop=True)

        # Calculer le pourcentage global des 'ODD count'
        total_odd_count_filtered = top5_odds_per_entity['ODD count'].sum()
        overall_odd_counts = top5_odds_per_entity.groupby('ODD number')['ODD count'].sum().reset_index()
        overall_odd_counts['Overall Percentage'] = (overall_odd_counts['ODD count'] / total_odd_count_filtered) * 100

        # Identifier les 5 'ODD numbers' les plus représentés globalement
        top5_odds_global = overall_odd_counts.nlargest(5, 'Overall Percentage')['ODD number'].tolist()

        # Filtrer le DataFrame pour inclure uniquement les Top 5 ODDs
        filtered_data = top5_odds_per_entity[top5_odds_per_entity['ODD number'].isin(top5_odds_global)]

        # Calculer les co-occurrences
        co_occurrence = defaultdict(int)
        entity_odds = filtered_data.groupby('Entity')['ODD number'].apply(list)

        for odds in entity_odds:
            unique_odds = sorted(set(odds))
            for pair in combinations(unique_odds, 2):
                co_occurrence[pair] += 1

        # Convertir en DataFrame
        co_occurrence_df = pd.DataFrame([
            {'ODD_1': pair[0], 'ODD_2': pair[1], 'Weight': weight}
            for pair, weight in co_occurrence.items()
        ])
        co_occurrence_df['Weight'] = co_occurrence_df['Weight'].apply(lambda x: x / co_occurrence_df['Weight'].sum() * 100)

        # Créer le graphe de réseau avec NetworkX
        G = nx.Graph()

        # Ajouter les nœuds
        for odd_ in top5_odds_global:
            G.add_node(odd_)

        # Ajouter les arêtes avec poids
        for _, row in co_occurrence_df.iterrows():
            G.add_edge(row['ODD_1'], row['ODD_2'], weight=row['Weight'])

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
            title="Graphe de Réseau Interactif des 5 ODDs les Plus Représentés",
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
    st.markdown("## 📋 Résumé des Analyses des top 5 ODDs")
    # Réinitialiser les résumés dans la session state
    st.session_state["summaries"] = None
    st.session_state["global_summary"] = None
    # Prendre le top 5 des ODDs les plus fréquents
    top_5_odds = (
        odd.groupby("ODD number")["ODD count"]
        .sum()
        .nlargest(5)
        .index.tolist()
    )

    # Filtrer uniquement les ODDs dans le top 5
    odd_filtered = odd[odd["ODD number"].isin(top_5_odds)]
    # Préparation des données par ODD
    odd_grouped = odd_filtered.groupby("ODD number")  # `odd` est le DataFrame chargé
    contents = [
        " ".join(group["Summary"].tolist()) for _, group in odd_grouped
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
                status_text.text(f"Résumé généré pour {i + 1}/{total_steps} ODDs.")

            # Nettoyer la barre de progression
            progress_bar.empty()
            status_text.empty()

            return summaries

        # Générer les résumés par ODD
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
        st.subheader("Résumé par ODD")
        for odd_number, summary in zip(odd_filtered.groupby("ODD number").groups.keys(), st.session_state["summaries"]):
            st.write(f"**ODD {int(odd_number)}** : {summary}")

    if st.session_state["global_summary"] is not None:
        st.subheader("Résumé global")
        st.write(st.session_state["global_summary"])
    else:
        st.info("Cliquez sur le bouton pour générer les résumés.")

with tab3:
    st.subheader("Données brutes")
    
    # Préparation des données brutes pour l'affichage et le téléchargement
    odd.reset_index(drop=True, inplace=True)
    odd["ODD count"] = odd["ODD count"].astype(int)
    odd["ODD number"] = odd["ODD number"].astype(int)
    

    st.table(odd)
