import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import random


def generate_planted_partition_graph(n, n_com, p_in, p_out, seed=None):
    """Génère un graphe à communautés plantées.

    Args:
        n: nombre total de sommets
        n_com: nombre de communautés
        p_in: probabilité d'arête entre deux sommets d'une même communauté
        p_out: probabilité d'arête entre deux sommets de communautés différentes
        seed: graine aléatoire

    Returns:
        Un graphe networkx, et un dictionnaire associant chaque sommet à sa
        communauté de référence.
    """
    raise NotImplementedError


def load_real_graph(name):
    """Charge un graphe réel depuis un fichier local.

    Args:
        name: identifiant du jeu de données ("amazon", "cora", etc.)

    Returns:
        Le graphe et la partition de référence si elle est disponible.
    """
    raise NotImplementedError

def load_cora(directed=False):
    """Charge le jeu de données Cora.

    Télécharge l'archive si nécessaire, extrait les fichiers, construit le
    graphe et renvoie la partition de référence.

    Args:
        directed: si True, renvoie un graphe orienté

    Returns:
        (G, ground_truth) : graphe NetworkX et dictionnaire
        {sommet: étiquette de classe}
    """
    raise NotImplementedError

def girvan_newman_communities(G, n_communities=None):
    """Détection de communautés par Girvan-Newman.

    Args:
        G: graphe networkx
        n_communities: nombre de communautés visé. Si None, on s'arrête au
            maximum de modularité.

    Returns:
        Liste de listes de sommets.
    """
    raise NotImplementedError


def louvain_communities(G, resolution=1.0):
    """Détection de communautés par l'algorithme de Louvain.

    Args:
        G: graphe networkx
        resolution: paramètre de résolution de la modularité

    Returns:
        Liste de listes de sommets.
    """
    raise NotImplementedError


def label_propagation_communities(G, max_iter=100):
    """Détection de communautés par propagation d'étiquettes.

    Args:
        G: graphe networkx
        max_iter: nombre maximal d'itérations

    Returns:
        Liste de listes de sommets.
    """
    raise NotImplementedError


def exact_modularity_maximization(G):
    """Recherche exacte de la partition de modularité maximale.

    Réservé aux très petites instances (n <= 10).
    """
    raise NotImplementedError


def modularity(G, communities, resolution=1.0):
    """Calcule la modularité d'une partition.

    Args:
        G: graphe networkx
        communities: liste de listes de sommets
        resolution: paramètre de résolution

    Returns:
        La modularité, valeur flottante.
    """
    raise NotImplementedError


def plot_communities(G, communities, title="", with_labels=False):
    """Visualise une partition en communautés.

    Args:
        G: graphe networkx
        communities: liste de listes de sommets
        title: titre du graphique
    """
    raise NotImplementedError


def benchmark_communities(G, algorithm, repeats=5):
    """Exécute un algorithme plusieurs fois et renvoie temps moyen et modularité.

    Args:
        G: graphe networkx
        algorithm: fonction prenant G en argument et renvoyant une partition
        repeats: nombre de répétitions

    Returns:
        Un couple (temps moyen, modularité moyenne).
    """
    raise NotImplementedError