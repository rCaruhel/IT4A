import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import random

from networkx.classes import Graph


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
    random.seed(seed)
    g = nx.Graph()
    nbr_sommet_par_com = n/n_com
    for num_com in range(1,n_com+1) :
        # Création des sommets pour une communauté
        for i in range(1,int(nbr_sommet_par_com)+1):
            g.add_node(f"{num_com}-{i}")


        # Liaison des sommets entre eux
        for i in range(1,int(nbr_sommet_par_com)+1):
            # parcourir les sommets 2 fois ne sert à rien
            # une fois tous les sommets liés au premier, il ne sert à rien de le revérifier
            for j in range (i+1,int(nbr_sommet_par_com)+1):
                if i == j : break
                if g.has_edge(f"{num_com}-{i}", f"{num_com}-{j}"): break
                if random.random() < p_in:
                    g.add_edge(f"{num_com}-{i}", f"{num_com}-{j}")

    # Maintenant que toutes les communautés sont construites,
    # Il faut relier les sommets de chaque commu entre eux
    # On boucle sur chaque communauté
    for i in range(1,n_com):
        # On boucle sur chaque sommet de la communauté
        for j in range(1,int(nbr_sommet_par_com)+1):
            # on boucle sur les communautés restantes
            for k in range(i+1,n_com+1):
                # Dans la communauté sélectionnée, on boucle sur chaque sommet
                for l in range(1,int(nbr_sommet_par_com)+1):
                    if g.has_edge(f"{i}-{j}", f"{k}-{l}"): break

                    if random.random() < p_out:
                        g.add_edge(f"{i}-{j}", f"{k}-{l}")


    return g


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
    m = G.number_of_edges()
    a = nx.adjacency_matrix(G)
    Q = 0
    G = Graph(G)
    for node_i in G.nodes():
        deg_i = G.degree[node_i]
        for node_j in G.nodes():
            if node_j == node_i : break
            deg_j = G.degree[node_j]
            Q += (a[node_i][node_j]-((deg_j*deg_i)/(2*m)))

    return Q/(2*m)


def plot_communities(G, communities, title="", with_labels=False):
    """Visualise une partition en communautés
    Args:
        G: graphe networkx
        communities: liste de listes de sommets
        title: titre du graphique
        with_labels: afficher ou non les étiquettes des sommets
    """
    # 1. Associer à chaque sommet l'indice de sa communauté
    comm_map = {}
    for comm_idx, community in enumerate(communities):
        for sommet in community:
            comm_map[sommet] = comm_idx

    # 2. Récupérer l'identifiant de communauté dans l'ordre de G.nodes()
    # (valeur par défaut -1 si un sommet n'est dans aucune communauté)
    node_colors = [comm_map.get(n, -1) for n in G.nodes()]

    # 3. Calculer une disposition stable pour le dessin
    pos = nx.spring_layout(G, seed=42)

    # 4. Dessiner le graphe avec une colormap
    plt.figure(figsize=(8, 6))
    if title:
        plt.title(title)

    nx.draw(
        G,
        pos,
        node_color=node_colors,
        cmap=plt.cm.tab10,  # tab10 ou rainbow pour des couleurs distinctes
        with_labels=with_labels,
        node_size=300,
        edge_color="gray",
        alpha=0.9
    )
    plt.show()


def benchmark_communities(G, algorithm, repeats=5):
    """Exécute un algorithme plusieurs fois et renvoie temps moyen et modularité.

    Args:
        G: graphe networkx
        algorithm: fonction prenant G en argument et renvoyant une partition
        repeats: nombre de répétitions

    Returns:
        Un couple (temps moyen, modularité moyenne).
    """
    tps_moy = 0
    mod_moy = 0
    for _ in range(repeats):
        start_time = time.time()
        mod_moy = algorithm(G)
        tps_moy += start_time - time.time()


    return tps_moy / repeats , mod_moy/repeats