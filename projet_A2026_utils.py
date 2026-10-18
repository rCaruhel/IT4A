from collections import Counter

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import random
import os

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

    ground_truth = {}
    random.seed(seed)
    g = nx.Graph()
    nodes_per_comm = int(n / n_com)
    for comm_idx in range(1, n_com + 1):
        # Création des sommets pour une communauté
        for i in range(1, nodes_per_comm + 1):
            node_id = f"{comm_idx}-{i}"
            g.add_node(node_id)
            ground_truth[node_id] = comm_idx

        # Liaison des sommets entre eux au sein de la communauté
        for i in range(1, nodes_per_comm + 1):
            for j in range(i + 1, nodes_per_comm + 1):
                if random.random() < p_in:
                    g.add_edge(f"{comm_idx}-{i}", f"{comm_idx}-{j}")

    # Liaison des sommets entre communautés différentes
    for c1 in range(1, n_com):
        for i in range(1, nodes_per_comm + 1):
            for c2 in range(c1 + 1, n_com + 1):
                for j in range(1, nodes_per_comm + 1):
                    if g.has_edge(f"{c1}-{i}", f"{c2}-{j}"):
                        continue
                    if random.random() < p_out:
                        g.add_edge(f"{c1}-{i}", f"{c2}-{j}")

    return g, ground_truth


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
    import urllib.request
    import tarfile

    DATA_URL  = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
    DATA_BASE = os.path.join(os.path.dirname(__file__), "data")
    DATA_DIR  = os.path.join(DATA_BASE, "cora")
    CONTENT_FILE = os.path.join(DATA_DIR, "cora.content")
    CITES_FILE   = os.path.join(DATA_DIR, "cora.cites")

    # --- Téléchargement et extraction si nécessaire ---
    if not os.path.isfile(CONTENT_FILE) or not os.path.isfile(CITES_FILE):
        os.makedirs(DATA_DIR, exist_ok=True)
        archive_path = os.path.join(DATA_DIR, "cora.tgz")
        print(f"Téléchargement de Cora depuis {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, archive_path)
        print("Extraction de l'archive ...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=DATA_BASE)
        print("Données Cora disponibles.")

    # --- Lecture de cora.content : noeuds et partition de référence ---
    # Format : <paper_id> TAB <feat_1> TAB ... TAB <feat_1433> TAB <class_label>
    ground_truth = {}   # {paper_id (str): class_label (str)}
    with open(CONTENT_FILE, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            node_id     = parts[0]
            class_label = parts[-1]     # dernier champ = catégorie
            ground_truth[node_id] = class_label

    # --- Construction du graphe ---
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(ground_truth.keys())

    # --- Lecture de cora.cites : arêtes ---
    # Format : <cited_id> TAB <citing_id>
    with open(CITES_FILE, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            if u in ground_truth and v in ground_truth:
                G.add_edge(u, v)

    return G, ground_truth

def girvan_newman_communities(G, n_communities=None):
    """Détection de communautés par Girvan-Newman.

    Args:
        G: graphe networkx
        n_communities: nombre de communautés visé. Si None, on s'arrête au
            maximum de modularité.

    Returns:
        Liste de listes de sommets.
    """
    # Copie de travail non orientée (Girvan-Newman suppose un graphe non orienté)
    g = Graph(G)

    def _get_communities(graph):
        """Retourne la liste des composantes connexes sous forme de listes."""
        return [list(c) for c in nx.connected_components(graph)]

    # ------------------------------------------------------------------
    # Boucle principale de Girvan-Newman
    # ------------------------------------------------------------------
    best_communities = _get_communities(g)
    best_q = modularity(G, best_communities)

    while g.number_of_edges() > 0:
        current_communities = _get_communities(g)

        # Arrêt si le nombre de communautés cible est atteint
        if n_communities is not None and len(current_communities) >= n_communities:
            return current_communities

        # Supprimer l'arête de betweenness maximale
        bet = nx.edge_betweenness_centrality(g)
        u, v = max(bet, key=bet.get)
        g.remove_edge(u, v)

        # Mise à jour du meilleur partitionnement (cas n_communities=None)
        if n_communities is None:
            new_communities = _get_communities(g)
            q = modularity(G, new_communities)
            if q > best_q:
                best_q = q
                best_communities = new_communities

    return best_communities if n_communities is None else _get_communities(g)


def louvain_communities(G, resolution=1.0):
    """Détection de communautés par l'algorithme de Louvain.

    Args:
        G: graphe networkx
        resolution: paramètre de résolution de la modularité

    Returns:
        Liste de listes de sommets.
    """
    if G.number_of_nodes() == 0:
        return []

    # Graphe courant (initialisé à partir de G non pondéré, poids = 1.0)
    current_graph = nx.Graph()
    for u in G.nodes():
        current_graph.add_node(u)
    for u, v in G.edges():
        current_graph.add_edge(u, v, weight=1.0)

    node_to_orig = {u: [u] for u in G.nodes()}
    m = G.number_of_edges()
    if m == 0:
        return [[u] for u in G.nodes()]

    while True:
        # =====================================================================
        # PHASE 1 : Optimisation locale de la modularité
        # =====================================================================

        ### création d'une communauté par sommet
        node2comm = {u: i for i, u in enumerate(current_graph.nodes())}
        degrees = dict(current_graph.degree(weight="weight"))
        tot_deg = {i: degrees[u] for i, u in enumerate(current_graph.nodes())}

        improved = False
        moving = True

        while moving:
            moving = False
            for u in list(current_graph.nodes()):
                c_u = node2comm[u]
                k_u = degrees[u]

                # Poids des liens reliant u aux communautés voisines
                comm_weights = {}
                for v, data in current_graph[u].items():
                    if v != u:  # On ignore la boucle réflexive sur u
                        c_v = node2comm[v]
                        comm_weights[c_v] = comm_weights.get(c_v, 0.0) + data.get("weight", 1.0)

                # Retrait temporaire de u de sa communauté
                tot_deg[c_u] -= k_u
                w_u_old = comm_weights.get(c_u, 0.0)
                base_gain = (w_u_old / m) - resolution * (tot_deg[c_u] * k_u) / (2.0 * m * m)

                best_comm = c_u
                best_delta_q = 0.0

                for c in set(comm_weights.keys()) | {c_u}:
                    w_u_c = comm_weights.get(c, 0.0)
                    gain = (w_u_c / m) - resolution * (tot_deg[c] * k_u) / (2.0 * m * m)
                    delta_q = gain - base_gain

                    if delta_q > best_delta_q + 1e-12:
                        best_delta_q = delta_q
                        best_comm = c

                # Réinsertion de u dans la meilleure communauté
                tot_deg[best_comm] += k_u
                node2comm[u] = best_comm

                if best_comm != c_u:
                    moving = True
                    improved = True

        comm_groups = {}
        for u, c in node2comm.items():
            comm_groups.setdefault(c, []).append(u)

        # Condition d'arrêt : aucun déplacement ou aucune contraction supplémentaire
        if not improved or len(comm_groups) == current_graph.number_of_nodes():
            break

        # =====================================================================
        # PHASE 2 : Agrégation (super-graphe contracté)
        # =====================================================================
        new_graph = nx.Graph()
        new_node_to_orig = {}
        old_to_new = {}

        for new_id, (_, members) in enumerate(comm_groups.items()):
            new_graph.add_node(new_id)
            orig_members = []
            for u in members:
                orig_members.extend(node_to_orig[u])
                old_to_new[u] = new_id
            new_node_to_orig[new_id] = orig_members

        for u, v, data in current_graph.edges(data=True):
            w = data.get("weight", 1.0)
            cu, cv = old_to_new[u], old_to_new[v]
            if new_graph.has_edge(cu, cv):
                new_graph[cu][cv]["weight"] += w
            else:
                new_graph.add_edge(cu, cv, weight=w)

        current_graph = new_graph
        node_to_orig = new_node_to_orig

    return list(node_to_orig.values())


def label_propagation_communities_simple(G, max_iter=100):
    """Détection de communautés par propagation d'étiquettes.

    Args:
        G: graphe networkx
        max_iter: nombre maximal d'itérations

    Returns:
        Liste de listes de sommets.
    """
    if G.number_of_nodes() == 0:
        return []

    nodes = list(G.nodes())
    labels = {u: i for i, u in enumerate(nodes)}

    for _ in range(max_iter):
        random.shuffle(nodes)
        changed = False

        for u in nodes:
            neighbors = list(G.neighbors(u))
            if not neighbors:
                continue

            counts = Counter(labels[v] for v in neighbors)
            max_freq = max(counts.values())
            best_labels = [lbl for lbl, freq in counts.items() if freq == max_freq]

            if labels[u] in best_labels and len(best_labels) == 1:
                continue

            new_label = random.choice(best_labels)
            if new_label != labels[u]:
                labels[u] = new_label
                changed = True

        if not changed:
            break

    communities = {}
    for u, lbl in labels.items():
        communities.setdefault(lbl, []).append(u)

    return list(communities.values())


def label_propagation_communities_stabilized(G, max_iter=100):
    """Détection de communautés par propagation d'étiquettes.

    Args:
        G: graphe networkx
        max_iter: nombre maximal d'itérations

    Returns:
        Liste de listes de sommets.
    """
    if G.number_of_nodes() == 0:
        return []

    nodes = list(G.nodes())
    labels = {u: i for i, u in enumerate(nodes)}

    degrees = dict(G.degree())

    for _ in range(max_iter):
        changed = False

        # Stabilisation par ordre décroissant des degrés,
        # avec mélange aléatoire pour les sommets de même degré
        current_nodes = list(nodes)
        random.shuffle(current_nodes)
        current_nodes.sort(key=lambda u: -degrees[u])

        for u in current_nodes:
            neighbors = list(G.neighbors(u))
            if not neighbors:
                continue

            counts = Counter(labels[v] for v in neighbors)
            max_freq = max(counts.values())
            best_labels = [lbl for lbl, freq in counts.items() if freq == max_freq]

            # Si l'étiquette actuelle fait déjà partie des majoritaires, on la conserve
            if labels[u] in best_labels:
                continue

            # En cas d'égalité stricte entre nouvelles étiquettes, choix aléatoire
            new_label = random.choice(best_labels)
            if new_label != labels[u]:
                labels[u] = new_label
                changed = True

        if not changed:
            break

    communities = {}
    for u, lbl in labels.items():
        communities.setdefault(lbl, []).append(u)


    return list(communities.values())

def exact_modularity_maximization(G):
    """Recherche exacte de la partition de modularité maximale.

    Réservé aux très petites instances (n <= 10).
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return []

    # Création d'une communauté par sommet
    best_communities = [[u] for u in nodes]

    # Calcul de la modularité initiale
    best_q = modularity(G, best_communities)

    # Procédure récursive de génération de toutes les partitions d'ensemble (Bell numbers)
    def _backtrack(node_idx, current_communities):
        nonlocal best_q, best_communities

        # Si tous les sommets ont été assignés
        if node_idx == n:
            q = modularity(G, current_communities)
            if q > best_q:
                best_q = q
                best_communities = [list(c) for c in current_communities]
            return

        u = nodes[node_idx]

        # 1. Ajouter le sommet u à une communauté existante
        for comm in current_communities:
            comm.append(u)
            _backtrack(node_idx + 1, current_communities)
            comm.pop()

        # 2. Créer une nouvelle communauté contenant uniquement u
        current_communities.append([u])
        _backtrack(node_idx + 1, current_communities)
        current_communities.pop()

    # Initialisation avec le premier sommet dans la première communauté
    _backtrack(1, [[nodes[0]]])

    return best_communities


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
    if m == 0:
        return 0.0

    # Associer chaque sommet à son indice de communauté
    node_to_comm = {}
    for comm_idx, comm in enumerate(communities):
        for u in comm:
            node_to_comm[u] = comm_idx

    n_c = len(communities)
    # L[c] = nb d'arêtes internes à la communauté c  (O(E))
    # d[c] = somme des degrés des sommets de c        (O(V))
    L = [0] * n_c
    d = [0] * n_c
    degrees = dict(G.degree())

    for u, v in G.edges():
        cu, cv = node_to_comm.get(u, -1), node_to_comm.get(v, -1)
        if cu == cv and cu != -1:
            L[cu] += 1

    for u in G.nodes():
        c = node_to_comm.get(u, -1)
        if c != -1:
            d[c] += degrees[u]

    return sum(L[c] / m - resolution * (d[c] / (2 * m)) ** 2 for c in range(n_c))


def plot_communities(G, communities, title="", with_labels=False):
    """Visualise une partition en communautés
    Args:
        G: graphe networkx
        communities: liste de listes de sommets
        title: titre du graphique
        with_labels: afficher ou non les étiquettes des sommets
    """
    # 1. Associer à chaque sommet l'indice de sa communauté
    node_to_comm = {}
    for comm_idx, community in enumerate(communities):
        for u in community:
            node_to_comm[u] = comm_idx

    # 2. Récupérer l'identifiant de communauté dans l'ordre de G.nodes()
    # (valeur par défaut -1 si un sommet n'est dans aucune communauté)
    node_colors = [node_to_comm.get(n, -1) for n in G.nodes()]

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
    tps_total = 0.0
    mod_total = 0.0
    for _ in range(repeats):
        start_time = time.time()
        comms = algorithm(G)
        tps_total += time.time() - start_time
        mod_total += modularity(G, comms)
    return tps_total / repeats, mod_total / repeats