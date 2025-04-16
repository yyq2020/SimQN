from qns.network.network import QuantumNetwork
from qns.network.topology.barabasialberttopo import BarabasiAlbertTopology


def test_barabasialbert_topo():
    topo = BarabasiAlbertTopology(nodes_number=100, new_nodes_egdes=1)
    net = QuantumNetwork(topo)

    print(net.nodes, net.qchannels)
