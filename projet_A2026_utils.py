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

    som_comm = {}
    random.seed(seed)
    g = nx.Graph()
    nbr_sommet_par_com = n/n_com
    for num_com in range(1,n_com+1) :
        # Création des sommets pour une communauté
        for i in range(1,int(nbr_sommet_par_com)+1):
            g.add_node(f"{num_com}-{i}")
            som_comm[f"{num_com}-{i}"] = num_com


        # Liaison des sommets entre eux
        for i in range(1,int(nbr_sommet_par_com)+1):
            # parcourir les sommets 2 fois ne sert à rien
            # une fois tous les sommets liés au premier, il ne sert à rien de le revérifier
            for j in range (i+1,int(nbr_sommet_par_com)+1):
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
                    if g.has_edge(f"{i}-{j}", f"{k}-{l}"): continue

                    if random.random() < p_out:
                        g.add_edge(f"{i}-{j}", f"{k}-{l}")


    return g,som_comm


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
    from collections import deque

    # Copie de travail non orientée (Girvan-Newman suppose un graphe non orienté)
    g = Graph(G)

    # ------------------------------------------------------------------
    # Fonctions internes
    # ------------------------------------------------------------------

    def _edge_betweenness(graph):
        """Calcule la betweenness de chaque arête (algorithme de Brandes)."""
        bet = {e: 0.0 for e in graph.edges()}

        for s in graph.nodes():
            # --- Parcours en largeur depuis s ---
            stack = []
            queue = deque([s])
            pred  = {w: [] for w in graph.nodes()}   # prédécesseurs sur chemins courts
            sigma = dict.fromkeys(graph.nodes(), 0)  # nb de plus courts chemins depuis s
            dist  = dict.fromkeys(graph.nodes(), -1) # distance
            sigma[s] = 1
            dist[s]  = 0

            while queue:
                v = queue.popleft()
                stack.append(v)
                for w in graph.neighbors(v):
                    if dist[w] < 0:                    # w découvert pour la première fois
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    if dist[w] == dist[v] + 1:         # v est sur un plus court chemin vers w
                        sigma[w] += sigma[v]
                        pred[w].append(v)

            # --- Remontée : accumulation des contributions ---
            delta = dict.fromkeys(graph.nodes(), 0.0)
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    # fraction de chemins passant par l'arête (v, w)
                    c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                    edge = (v, w) if graph.has_edge(v, w) else (w, v)
                    if edge in bet:
                        bet[edge] += c
                    delta[v] += c

        # Graphe non orienté : chaque paire (s, t) et (t, s) comptée deux fois
        return {e: b / 2.0 for e, b in bet.items()}

    def _get_communities(graph):
        """Retourne la liste des composantes connexes sous forme de listes."""
        return [list(c) for c in nx.connected_components(graph)]

    # ------------------------------------------------------------------
    # Boucle principale de Girvan-Newman
    # ------------------------------------------------------------------
    best_comms = _get_communities(g)
    best_q     = modularity(G, best_comms)

    while g.number_of_edges() > 0:
        current_comms = _get_communities(g)

        # Arrêt si le nombre de communautés cible est atteint
        if n_communities is not None and len(current_comms) >= n_communities:
            return current_comms

        # Supprimer l'arête de betweenness maximale
        bet    = _edge_betweenness(g)
        u, v   = max(bet, key=bet.get)
        g.remove_edge(u, v)

        # Mise à jour du meilleur partitionnement (cas n_communities=None)
        if n_communities is None:
            new_comms = _get_communities(g)
            q = modularity(G, new_comms)
            if q > best_q:
                best_q     = q
                best_comms = new_comms

    return best_comms if n_communities is None else _get_communities(g)


def louvain_communities(G, resolution=1.0):
    """Détection de communautés par l'algorithme de Louvain (implémentation complète)."""
    if G.number_of_edges() == 0:
        return [[n] for n in G.nodes()]

    # 1. Préparation d'un graphe pondéré (poids = 1.0 par défaut)
    current_graph = nx.Graph()
    for u in G.nodes():
        current_graph.add_node(u)
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        if current_graph.has_edge(u, v):
            current_graph[u][v]["weight"] += w
        else:
            current_graph.add_edge(u, v, weight=w)

    # Dictionnaire gardant la trace des sommets originaux contenus dans chaque super-sommet
    node_to_orig = {n: [n] for n in G.nodes()}

    # Poids total m du graphe (une boucle compte double dans les degrés, donc poids complet dans m)
    m = current_graph.size(weight="weight")

    while True:
        # =====================================================================
        # PHASE 1 : Optimisation locale de la modularité
        # =====================================================================
        # Au départ, chaque noeud est dans sa propre communauté
        node2comm = {n: i for i, n in enumerate(current_graph.nodes())}
        degrees = dict(current_graph.degree(weight="weight"))

        # Somme des degrés par communauté (Sigma_tot)
        tot_deg = {i: degrees[n] for i, n in enumerate(current_graph.nodes())}

        improved_in_pass = False
        moving = True

        while moving:
            moving = False
            for u in list(current_graph.nodes()):
                c_u = node2comm[u]
                k_u = degrees[u]

                # Calcul des poids reliant u aux communautés voisines : k_{u, in}
                comm_weights = {}
                for v, data in current_graph[u].items():
                    if v != u:  # On ignore la boucle interne à u lui-même
                        c_v = node2comm[v]
                        comm_weights[c_v] = comm_weights.get(c_v, 0.0) + data.get("weight", 1.0)

                # Retirer temporairement u de sa communauté actuelle c_u
                tot_deg[c_u] -= k_u
                w_u_old = comm_weights.get(c_u, 0.0)

                # Coût de référence pour remettre u dans son ancienne communauté c_u
                base_gain = (w_u_old / m) - resolution * (tot_deg[c_u] * k_u) / (2.0 * (m ** 2))

                best_comm = c_u
                best_delta_q = 0.0

                # Tester toutes les communautés voisines (ainsi que c_u)
                candidate_comms = set(comm_weights.keys()) | {c_u}
                for c in candidate_comms:
                    w_u_c = comm_weights.get(c, 0.0)
                    gain = (w_u_c / m) - resolution * (tot_deg[c] * k_u) / (2.0 * (m ** 2))
                    delta_q = gain - base_gain

                    if delta_q > best_delta_q + 1e-12:
                        best_delta_q = delta_q
                        best_comm = c

                # Placer u dans la meilleure communauté trouvée
                tot_deg[best_comm] += k_u
                node2comm[u] = best_comm

                if best_comm != c_u:
                    moving = True
                    improved_in_pass = True

        # Regrouper les noeuds par communauté trouvée
        comm_groups = {}
        for n, c in node2comm.items():
            comm_groups.setdefault(c, []).append(n)

        # Condition d'arrêt : aucun mouvement ou aucune réduction du nombre de sommets
        if not improved_in_pass or len(comm_groups) == current_graph.number_of_nodes():
            break

        # =====================================================================
        # PHASE 2 : Agrégation (contraction des communautés en super-sommets)
        # =====================================================================
        new_graph = nx.Graph()
        new_node_to_orig = {}
        old_to_new_id = {}

        for new_id, (_, nodes) in enumerate(comm_groups.items()):
            new_graph.add_node(new_id)
            orig_list = []
            for n in nodes:
                orig_list.extend(node_to_orig[n])
                old_to_new_id[n] = new_id
            new_node_to_orig[new_id] = orig_list

        for u, v, data in current_graph.edges(data=True):
            w = data.get("weight", 1.0)
            cu, cv = old_to_new_id[u], old_to_new_id[v]
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

    deg = dict(G.degree())

    for _ in range(max_iter):
        changed = False

        # Stabilisation par ordre décroissant des degrés,
        # avec mélange aléatoire pour les sommets de même degré
        current_nodes = list(nodes)
        random.shuffle(current_nodes)
        current_nodes.sort(key=lambda u: -deg[u])

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
    best_partition = [[u] for u in nodes]

    # calcul de la modularité
    best_q = modularity(G, best_partition)

    # Procédure récursive de génération de toutes les partitions d'ensemble (Bell numbers)
    def _backtrack(node_idx, current_partition):
        nonlocal best_q, best_partition

        # si le sommet est le dernier sommet du graphe
        if node_idx == n:
            q = modularity(G, current_partition)
            if q > best_q:
                best_q = q
                best_partition = [list(c) for c in current_partition]
            return


        u = nodes[node_idx]

        # 1. Ajouter le sommet u à une communauté existante
        for comm in current_partition:
            comm.append(u)
            _backtrack(node_idx + 1, current_partition)
            comm.pop()

        # 2. Créer une nouvelle communauté contenant uniquement u
        current_partition.append([u])
        _backtrack(node_idx + 1, current_partition)
        current_partition.pop()

    # Initialisation avec le premier sommet dans la première communauté
    _backtrack(1, [[nodes[0]]])

    return best_partition


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
    comm_of = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            comm_of[node] = idx

    n_c = len(communities)
    # L[c] = nb d'arêtes internes à la communauté c  (O(E))
    # d[c] = somme des degrés des sommets de c        (O(V))
    L = [0] * n_c
    d = [0] * n_c
    deg = dict(G.degree())

    for u, v in G.edges():
        cu, cv = comm_of.get(u, -1), comm_of.get(v, -1)
        if cu == cv and cu != -1:
            L[cu] += 1

    for u in G.nodes():
        c = comm_of.get(u, -1)
        if c != -1:
            d[c] += deg[u]

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
    tps_total = 0.0
    mod_total = 0.0
    for _ in range(repeats):
        start_time = time.time()
        comms = algorithm(G)
        tps_total += time.time() - start_time
        mod_total += modularity(G, comms)
    return tps_total / repeats, mod_total / repeats