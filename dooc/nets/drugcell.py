import os
import typing
from dataclasses import dataclass
from collections import defaultdict
import torch
from torch import nn
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag


def _load_gene_mapping(file_path: str) -> dict:
    res = {}

    with open(file_path) as f:
        for line in f:
            line = line.rstrip().split()
            res[line[1]] = int(line[0])

    return res


def _load_ontology(file_name: str, gene2id_mapping: dict) -> typing.Sequence:
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


@dataclass
class DrugcellConfig:
    d_model: int
    gene_dim: int
    num_hiddens_genotype: int
    gene2ind_path: str
    ont_path: str


class Drugcell(nn.Module):
    """GNN for mutations embeddings.

    reference: https://github.com/idekerlab/DrugCell/

    """

    DEFAULT_CONFIG = DrugcellConfig(
        d_model=768,
        gene_dim=3008,
        num_hiddens_genotype=6,
        gene2ind_path=os.path.join(os.path.dirname(__file__), "../data", "gene2ind.txt"),
        ont_path=os.path.join(os.path.dirname(__file__), "../data", "drugcell_ont.txt"),
    )

    def __init__(self, conf: DrugcellConfig = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.conf = conf

        dg, dg_root, term_size_map, term_direct_gene_map = self._get_params()
        self.dg, self.dg_root = dg, dg_root
        self.term_size_map, self.term_direct_gene_map = (
            term_size_map,
            term_direct_gene_map,
        )

        self._cal_term_dim()
        self._contruct_direct_gene_layer()
        self._construct_nn_graph()
        self._construct_final_layer()

    def _contruct_direct_gene_layer(self):
        """
        build a layer for forwarding gene that are directly annotated with the term
        """
        for term, gene_set in self.term_direct_gene_map.items():
            if len(gene_set) == 0:
                raise ValueError(f"There are no directed asscoiated genes for {term}")

            # if there are some genes directly annotated with the term, add a layer taking in all genes and forwarding out only those genes
            self.add_module(
                term + "_direct_gene_layer",
                nn.Linear(self.conf.gene_dim, len(gene_set)),
            )

    def _construct_nn_graph(self):
        """
        start from bottom (leaves), and start building a neural network using the given ontology
            adding modules --- the modules are not connected yet
        """
        self.term_layer_list = []  # term_layer_list stores the built neural network
        self.term_neighbor_map = {}
        # term_neighbor_map records all children of each term
        for term in self.dg.nodes():
            self.term_neighbor_map[term] = []
            for child in self.dg.neighbors(term):
                self.term_neighbor_map[term].append(child)

        while True:
            leaves = [n for n in self.dg.nodes() if self.dg.out_degree(n) == 0]
            # leaves = [n for n,d in self.dg.out_degree().items() if d==0]
            # leaves = [n for n,d in self.dg.out_degree() if d==0]

            if len(leaves) == 0:
                break

            self.term_layer_list.append(leaves)

            for term in leaves:

                # input size will be #chilren + #genes directly annotated by the term
                input_size = 0

                for child in self.term_neighbor_map[term]:
                    input_size += self.term_dim_map[child]

                if term in self.term_direct_gene_map:
                    input_size += len(self.term_direct_gene_map[term])

                # term_hidden is the number of the hidden variables in each state
                term_hidden = self.term_dim_map[term]

                self.add_module(
                    term + "_linear_layer", nn.Linear(input_size, term_hidden)
                )
                self.add_module(term + "_batchnorm_layer", nn.BatchNorm1d(term_hidden))

            self.dg.remove_nodes_from(leaves)

    def _construct_final_layer(self):
        """
        add modules for final layer
        """
        self.add_module(
            "final_linear_layer",
            nn.Linear(self.conf.num_hiddens_genotype, self.conf.d_model),
        )

    def _cal_term_dim(self):
        """
        calculate the number of values in a state (term)
        term_size_map is the number of all genes annotated with the term
        """
        self.term_dim_map = {}
        for term, term_size in self.term_size_map.items():
            num_output = self.conf.num_hiddens_genotype

            # log the number of hidden variables per each term
            num_output = int(num_output)
            self.term_dim_map[term] = num_output

    def _get_params(self):
        gene2id_mapping = _load_gene_mapping(self.conf.gene2ind_path)
        dg, dg_root, term_size_map, term_direct_gene_map = _load_ontology(
            self.conf.ont_path, gene2id_mapping
        )
        return dg, dg_root, term_size_map, term_direct_gene_map

    def load_ckpt(self, *ckpt_files: str) -> None:
        self.load_state_dict(
            torch.load(ckpt_files[0], map_location=torch.device("cpu")), strict=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        omit drug layer, cmp to origin drugcell
        """
        x = x.float()
        x_dim = x.dim()
        x = x.unsqueeze(0) if x_dim == 1 else x
        gene_input = x.narrow(1, 0, self.conf.gene_dim)

        # define forward function for genotype dcell #############################################
        term_gene_out_map = {}

        for term, _ in self.term_direct_gene_map.items():
            term_gene_out_map[term] = self._modules[term + "_direct_gene_layer"](
                gene_input
            )

        term_nn_out_map = {}

        for _, layer in enumerate(self.term_layer_list):

            for term in layer:

                child_input_list = []

                for child in self.term_neighbor_map[term]:
                    child_input_list.append(term_nn_out_map[child])

                if term in self.term_direct_gene_map:
                    child_input_list.append(term_gene_out_map[term])

                child_input = torch.cat(child_input_list, 1)

                term_nn_out = self._modules[term + "_linear_layer"](child_input)

                tanh_out = torch.tanh(term_nn_out)
                term_nn_out_map[term] = self._modules[term + "_batchnorm_layer"](
                    tanh_out
                )

        out = self._modules['final_linear_layer'](term_nn_out_map[self.dg_root])
        if x_dim == 1:
            out = out.squeeze(0)
        return out
