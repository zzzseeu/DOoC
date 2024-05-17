import typing
import networkx as nx
from collections import defaultdict
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag


def load_gene_mapping(file_path: str) -> dict:
    res = {}

    with open(file_path) as f:
        for line in f:
            line = line.rstrip().split()
            res[line[1]] = int(line[0])

    return res


def load_ontology(file_name: str, gene2id_mapping: dict) -> typing.Sequence:
    dg = nx.DiGraph()
    term_direct_gene_map = defaultdict(set)

    term_size_map, gene_set = {}, set()

    file_handle = open(file_name)
    for line in file_handle:
        line = line.rstrip().split()
        if line[2] == "default":
            dg.add_edge(line[0], line[1])
            continue

        if line[1] not in gene2id_mapping:
            continue
        if line[0] not in term_direct_gene_map:
            term_direct_gene_map[line[0]] = set()

        term_direct_gene_map[line[0]].add(gene2id_mapping[line[1]])
        gene_set.add(line[1])
    file_handle.close()

    print("There are", len(gene_set), "genes")

    leaves = []
    for term in dg.nodes():
        term_gene_set = set()
        if term in term_direct_gene_map:
            term_gene_set = term_direct_gene_map[term]

        deslist = nxadag.descendants(dg, term)

        for child in deslist:
            if child in term_direct_gene_map:
                term_gene_set = term_gene_set | term_direct_gene_map[child]

        if len(term_gene_set) == 0:
            raise ValueError(f"There is empty terms, please delete term: {term}")

        term_size_map[term] = len(term_gene_set)

        if dg.in_degree(term) == 0:
            leaves.append(term)

    ug = dg.to_undirected()
    connected_subg_list = list(nxacc.connected_components(ug))

    print("There are", len(leaves), "roots:", leaves[0])
    print("There are", len(dg.nodes()), "terms")
    print("There are", len(connected_subg_list), "connected componenets")

    if len(leaves) > 1:
        raise ValueError(
            "There are more than 1 root of ontology. Please use only one root."
        )

    if len(connected_subg_list) > 1:
        raise ValueError(
            "There are more than connected components. Please connect them."
        )

    return dg, leaves[0], term_size_map, term_direct_gene_map
