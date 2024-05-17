import os
import torch
from torch import nn
from dataclasses import dataclass
from dooc.utils import load_gene_mapping, load_ontology


@dataclass
class GeneGNNConfig:
    gene_dim: int
    drug_dim: int
    num_hiddens_genotype: int
    num_hiddens_drug: list
    num_hiddens_final: int
    gene2ind_path: str
    ont_path: str


class GeneGNN(nn.Module):
    """GNN for mutations embeddings.

    reference: https://github.com/idekerlab/DrugCell/

    """

    DEFAULT_CONFIG = GeneGNNConfig(
        gene_dim=3008,
        drug_dim=2048,
        num_hiddens_genotype=6,
        num_hiddens_drug=[100, 50, 6],
        num_hiddens_final=6,
        gene2ind_path=os.path.join(os.path.dirname(__file__), "data", "gene2ind.txt"),
        ont_path=os.path.join(os.path.dirname(__file__), "data", "drugcell_ont.txt"),
    )

    def __init__(self, conf: GeneGNNConfig = DEFAULT_CONFIG) -> None:
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
        self._construct_nn_drug()
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

    def _construct_nn_drug(self):
        """
        add modules for fully connected neural networks for drug processing
        """
        input_size = self.conf.drug_dim

        for i in range(len(self.conf.num_hiddens_drug)):
            self.add_module(
                "drug_linear_layer_" + str(i + 1),
                nn.Linear(input_size, self.conf.num_hiddens_drug[i]),
            )
            self.add_module(
                "drug_batchnorm_layer_" + str(i + 1),
                nn.BatchNorm1d(self.conf.num_hiddens_drug[i]),
            )
            self.add_module(
                "drug_aux_linear_layer1_" + str(i + 1),
                nn.Linear(self.conf.num_hiddens_drug[i], 1),
            )
            self.add_module("drug_aux_linear_layer2_" + str(i + 1), nn.Linear(1, 1))

            input_size = self.conf.num_hiddens_drug[i]

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
                self.add_module(term + "_aux_linear_layer1", nn.Linear(term_hidden, 1))
                self.add_module(term + "_aux_linear_layer2", nn.Linear(1, 1))

            self.dg.remove_nodes_from(leaves)

    def _construct_final_layer(self):
        """
        add modules for final layer
        """
        final_input_size = (
            self.conf.num_hiddens_genotype + self.conf.num_hiddens_drug[-1]
        )
        self.add_module(
            "final_linear_layer",
            nn.Linear(final_input_size, self.conf.num_hiddens_final),
        )
        self.add_module(
            "final_batchnorm_layer", nn.BatchNorm1d(self.conf.num_hiddens_final)
        )
        self.add_module(
            "final_aux_linear_layer", nn.Linear(self.conf.num_hiddens_final, 1)
        )
        self.add_module("final_linear_layer_output", nn.Linear(1, 1))

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
        gene2id_mapping = load_gene_mapping(self.conf.gene2ind_path)
        dg, dg_root, term_size_map, term_direct_gene_map = load_ontology(
            self.conf.ont_path, gene2id_mapping
        )
        return dg, dg_root, term_size_map, term_direct_gene_map

    def load_ckpt(self, *ckpt_files: str) -> None:
        self.load_state_dict(
            torch.load(ckpt_files[0], map_location=torch.device("cpu")), strict=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        removed drug layer
        """

        gene_input = x.narrow(1, 0, self.conf.gene_dim)
        # drug_input = x.narrow(1, self.conf.gene_dim, self.conf.drug_dim)

        # define forward function for genotype dcell #############################################
        term_gene_out_map = {}

        for term, _ in self.term_direct_gene_map.items():
            term_gene_out_map[term] = self._modules[term + "_direct_gene_layer"](
                gene_input
            )

        term_nn_out_map = {}
        aux_out_map = {}

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
                aux_layer1_out = torch.tanh(
                    self._modules[term + "_aux_linear_layer1"](term_nn_out_map[term])
                )
                aux_out_map[term] = self._modules[term + "_aux_linear_layer2"](
                    aux_layer1_out
                )

        return term_nn_out_map[self.dg_root]
