#    SimQN: a discrete-event simulator for the quantum networks
#    Copyright (C) 2021-2022 Lutong Chen, Jian Li, Kaiping Xue
#    University of Science and Technology of China, USTC.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import itertools
from qns.entity.node.app import Application
from qns.entity.qchannel.qchannel import QuantumChannel
from qns.entity.node.node import QNode
from typing import Dict, List, Optional, Tuple
from qns.network.topology import Topology
from qns.utils.rnd import get_weighted_choice


class BarabasiAlbertTopology(Topology):
    """
    BarabasiAlbertTopology is a random topology generator based on the Barabasi-Albert model.
    Each new QNode added has `new_nodes_egdes` to existing QNodes.
    The probability of a new QNode connecting to an existing QNode is proportional to the degree of the existing QNode.
    """
    def __init__(self, nodes_number, new_nodes_egdes: int,
                 nodes_apps: List[Application] = [],
                 qchannel_args: Dict = {}, cchannel_args: Dict = {},
                 memory_args: Optional[List[Dict]] = {}):
        """
        Args:
            nodes_number: the number of Qnodes
            new_nodes_egdes: the number of edges of a new node, must be greater than 0 and less than nodes_number
        """
        super().__init__(nodes_number, nodes_apps, qchannel_args, cchannel_args, memory_args)
        self.new_nodes_egdes = new_nodes_egdes

    def build(self) -> Tuple[List[QNode], List[QuantumChannel]]:
        # check config
        if self.new_nodes_egdes < 1 or self.new_nodes_egdes >= self.nodes_number:
            raise ValueError("new_nodes_egdes must be greater than 0 and less than nodes_number")
        nl: List[QNode] = []
        ll: List[QuantumChannel] = []
        # generate initial QNodes and QuantumChannels
        for i in range(self.new_nodes_egdes):
            n = QNode(f"n{i+1}")
            nl.append(n)
        initial_edges = list(itertools.combinations(nl, 2))
        for n1, n2 in initial_edges:
            qc = QuantumChannel(name=f"l{n1}-{n2}", **self.qchannel_args)
            ll.append(qc)
            n1.add_qchannel(qc)
            n2.add_qchannel(qc)
        # generate new QNodes following Barabasi-Albert model
        for i in range(self.new_nodes_egdes, self.nodes_number):
            n = QNode(f"n{i+1}")
            # deal with boundary conditions
            if self.new_nodes_egdes == 1 and i == 1:
                n1 = nl[0]
                nl.append(n1)
                qc = QuantumChannel(name=f"l{n1}-{n}", **self.qchannel_args)
                ll.append(qc)
                n1.add_qchannel(qc)
                n.add_qchannel(qc)
                continue
            # add new_node_egdes edges for the new node
            weighted_choice = [len(n_i.qchannels) for n_i in nl]
            choice_list = get_weighted_choice(nl, weighted_choice, self.new_nodes_egdes)
            for n_i in choice_list:
                qc = QuantumChannel(name=f"l{n_i}-{n}", **self.qchannel_args)
                ll.append(qc)
                n.add_qchannel(qc)
                n_i.add_qchannel(qc)
            nl.append(n)
        # QNode configuration
        self._add_apps(nl)
        self._add_memories(nl)
        return nl, ll
