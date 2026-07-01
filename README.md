# SimQN
[![Pytest](https://github.com/QNLab-USTC/SimQN/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/QNLab-USTC/SimQN/actions/workflows/pytest.yml)
![Flake8](https://github.com/QNLab-USTC/SimQN/actions/workflows/flake8.yml/badge.svg)
[![GitHub Stars](https://img.shields.io/github/stars/QNLab-USTC/SimQN?style=flat)](https://github.com/QNLab-USTC/SimQN)&nbsp;
[![PyPI - Downloads](https://img.shields.io/pypi/dm/qns?style=flat&logo=pypi&label=PyPI)](https://pypi.org/project/qns/)&nbsp;
[![Citations](https://img.shields.io/badge/Citations-66-blue?style=flat-square&logo=googlescholar)](https://scholar.google.com/scholar?cites=17361842933401893020)&nbsp;
[![Academic Usages](https://img.shields.io/badge/Academic-Usages-21-purple?style=flat-square)](https://github.com/QNLab-USTC/SimQN?tab=readme-ov-file#academic-paper-using-simqn-for-simulation)
[![Extensions](https://img.shields.io/badge/Extensions-7-orange?style=flat-square)](https://github.com/QNLab-USTC/SimQN?tab=readme-ov-file#platform-extension)&nbsp;

- [SimQN](#simqn)
  - [News](#News)
  - [Overview](#overview)
  - [Roadmap](#roadmap)
  - [Why choose SimQN?](#why-choose-simqn)
  - [Installation](#installation)
  - [First sight of SimQN](#first-sight-of-simqn)
  - [Get Help](#get-help)
  - [Release History](#release-history)
  - [Platform Extension](#platform-extension)
  - [How to contribute?](#how-to-contribute)
  - [License and Authors](#license-and-authors)
  - [Citation](#citation)

## News

* 🚀 **Version Update:** SimQN v0.2.3 has been released in June 2026.

* 📚 **Academic Papers:** SimQN has been implemented as the simulation platform in **24 peer-reviewed academic papers** (66 citations in total on Google Scholar).

* 🧩 **Platform Extension:** **7 simulation platforms or system extensions** have been built on SimQN, with more in development.

For more details, please refer to [Release History](#release-history), [Platform Extension](#platform-extension), and [Academic Paper Using SimQN for Simulation](#academic-paper-using-simqn-for-simulation).

## Overview

**[Attention]** Most of existing&future studies in [QNLab](https://qnlab-ustc.com/) are evaluated on SimQN platform. If you would like to follow our work or seek an easy-to-use quantum network simulator, you cannot miss SimQN! Please check out the following publication for details. **([Link](https://ieeexplore.ieee.org/abstract/document/10024900/) and [PDF](https://infonetlijian.github.io/homepage/PDF_files/2023-%E3%80%90IEEE%20Network%E3%80%91-SimQN_a_Network-layer_Simulator_for_the_Quantum_Network_Investigation.pdf)).**  

Welcome to SimQN's documentation. SimQN is a discrete-event-based network simulation platform for quantum networks.
SimQN enables large-scale investigations, including QKD protocols, entanglement distribution protocols, routing algorithms, and resource allocation schemas in quantum networks. For example, users can use SimQN to design routing algorithms for better QKD performance. For more information, please refer to the [Documents](https://qnlab-ustc.github.io/SimQN/).

SimQN is a Python3 library for quantum networking simulation. It is designed to be general purpose. It means that SimQN can be used for both QKD networks, entanglement distribution networks, and other kinds of quantum networks' evaluation. The core idea is that SimQN makes no architecture assumption. Since there is currently no recognized network architecture in quantum network investigations, SimQN stays flexible in this aspect.

SimQN provides high performance for large-scale network simulation. SimQN uses [Cython](https://cython.org/) to compile critical code in C/C++ libraries to boost the evaluation. Also, along with the commonly used quantum state-based physical models, SimQN provides a higher-layer fidelity-based entanglement physical model to reduce the computation overhead and bring convenience for users in evaluation. Last but not least, SimQN provides several network auxiliary models for easily building network topologies, producing routing tables, and managing multiple session requests.

We have already reproduced several academic papers using SimQN, and the open-source code can be found at our group [website](https://github.com/QNLab-USTC).


## Roadmap

![Roadmap](https://github.com/QNLab-USTC/QuantumNetworkWebsite/blob/main/static/images/simqn_roadmap.png)

- Currently, we are focusing on developing the 0.2.x version of SimQN, which will include:
  - Useful network utilities, such as more random topology generators, routing algorithms, and session request generators, real topology adaptors, and Multi-path routing algorithms.
  - Representative quantum network protocols, such as Q-CAST routing protocol, PS/PU routing protocol, REPS routing protocol for quantum information networks, and CASCADE error correction protocol for QKD networks.

- The following functions will be included in the future versions:
  - Practical quantum network entities, such as quantum repeaters, quantum switches, and quantum benchmarking devices.
  - Useful network utilities, such as random request traffic generators.
  - Support for Quantum network stack protocols, including KM protocols, routing protocols in QKD networks, and entanglement distribution protocols in quantum information networks.
  - Realization of easy-to-use GUI for SimQN.
  
## Why choose SimQN?

SimQN is designed as a functional and easy-to-use simulator, like [NS3](https://www.nsnam.org/) in classic networks, it provides numerous functions for anyone who wants to simulate a QKD network or entanglement-based network. 

Compared with the existing quantum network simulators, the developers pay more attention to simulation in the network area. Currently, a network simulation can be complicated, as users may have to implement routing algorithms and multiply protocols in different layers to complete a simulation. SimQN aims to break down this problem by providing a modulized quantum node and reusable algorithms and protocols. As a result, users can focus on what they study and reuse other built-in modules. The developers believe this will significantly reduce the burden on our users. As for the physics area, SimQN can also simulate quantum noise, fidelity, and more. Thus, if you focus on the research of the quantum network area, SimQN can be a competitive choice.  

**The main advantages of SimQN can be summarized as follows.**

- Easy-to-use with Python
- High-efficiency simulator core
- Customizable qubit model
- Built-in quantum internet protocol stack
- Mainstream quantum applications support, e.g., QKD, quantum networking, and distributed quantum computing (incoming)
- Periodical updates for academia-related functions and state-of-the-art solutions in the community  (you are welcome to highlight your work and contribute your code on SimQN)
- ...


## Installation

Install and update using `pip`:
```
pip3 install -U qns
```

## First sight of SimQN

Here is an example of using SimQN.

``` Python

    from qns.simulator.simulator import Simulator
    from qns.network.topology import RandomTopology
    from qns.network.protocol.entanglement_distribution import EntanglementDistributionApp
    from qns.network import QuantumNetwork
    from qns.network.route.dijkstra import DijkstraRouteAlgorithm
    from qns.network.topology.topo import ClassicTopology
    import qns.utils.log as log
    import logging

    init_fidelity = 0.99   # the initial entanglement's fidelity
    nodes_number = 150     # the number of nodes
    lines_number = 450     # the number of quantum channels
    qchannel_delay = 0.05  # the delay of quantum channels
    cchannel_delay = 0.05  # the delay of classic channels
    memory_capacity = 50   # the size of quantum memories
    send_rate = 10         # the send rate
    requests_number = 10   # the number of sessions (SD-pairs)

    # generate the simulator
    s = Simulator(0, 10, accuracy=1000000)

    # set the log's level
    log.logger.setLevel(logging.INFO)
    log.install(s)

    # generate a random topology using the parameters above
    # each node will install EntanglementDistributionApp for hop-by-hop entanglement distribution
    topo = RandomTopology(nodes_number=nodes_number,
                          lines_number=lines_number,
                          qchannel_args={"delay": qchannel_delay},
                          cchannel_args={"delay": cchannel_delay},
                          memory_args=[{"capacity": memory_capacity}],
                          nodes_apps=[EntanglementDistributionApp(init_fidelity=init_fidelity)])

    # build the network, with Dijkstra's routing algorithm
    net = QuantumNetwork(topo=topo, classic_topo=ClassicTopology.All, route=DijkstraRouteAlgorithm())

    # build the routing table
    net.build_route()

    # randomly select multiple sessions (SD-pars)
    net.random_requests(requests_number, attr={"send_rate": send_rate})

    # all entities in the network will install the simulator and do initiate works.
    net.install(s)

    # run simulation
    s.run()
```

## Get Help

- This [documentation](https://qnlab-ustc.github.io/SimQN/) may answer most questions.
    - The [tutorial](https://qnlab-ustc.github.io/SimQN/tutorials.html) here presents how to use SimQN.
    - The [API manual](https://qnlab-ustc.github.io/SimQN/modules.html) shows more detailed information.
- Welcome to report bugs at [Github](https://github.com/QNLab-USTC/SimQN).

## Release History

- v0.2.3(Released 2026.06)
  - *New funcation!!!*
  - Simulator Core: StableEventPool with Heap-Based Event Scheduling.
  - Network Utilities: Dijkstra Routing with Binary Heap, ClassicChannel Extension for Reliable Packet Delivery and Ordering, Link-decoherence quantum channel with differentiated link fidelity.

- v0.2.2(Released 2026.01)
  - *New functions!!!*
  - Simulator Core: Hash Bucket-Based Event Scheduler.
  - Network Utilities: Add Real Topology Generator, including AboveNetTopology, AGISTopology and Creat Topology From GML File Download From [The Internet Topology Zoo](https://topology-zoo.org/dataset.html). Update the Dijkstra-based Routing Utility.

- v0.2.1(Released 2025.04)
  - *New functions!!!*
  - Network Utilities: Add Random Topology Generator, including ER model, BA model, and dual-BA model.
  - Applications: Add CASCADE Error Correction and Privacy Amplification Process for BB84 Protocol.

- v0.1.5(Released 2022.09)
  - *New functions!!!*
  - Simulator Core: Cython Optimization, Multi-process Support.
  - Quantum Entitise: Quantum Memory, Delay Model.
  - Tools: Monitor Tools.

- v0.1.4(Released 2022.03)
  - *New functions!!!*
  - Simulator Core: Priority Queue-Based Event Scheduler.
  - Physical Backends: Qubit Model, EPR Model, Quantum Gates.
  - Quantum Entitise: Quantum Node, Quantum Channel.
  - Network Utilities: Topology Generator, Routing Utility.
  - Applications: BB84 Protocol, Entanglement Swapping Protocol.
  - Tools: Rnd Tools.

## Platform Extension

- Josipa et al. enabled SimQN to accept structured data from upstream via the HTTP protocol by utilizing FastAPI, thereby developing a [Docker environment for QKD network simulations](https://repozitorij.fpz.unizg.hr/object/fpz:3666). [2025]
- Majid K et al. integrated SimQN with the open-source OpenAirInterface (OAI) stack to deploy a 5G standalone network capable of supporting quantum operations, proposing the [quantum-inspired RAN (Q-RAN)](https://ieeexplore.ieee.org/abstract/document/11432359), a next-generation framework. [2025]
- Abane et al. developed [MQNS](https://ieeexplore.ieee.org/abstract/document/11334170) based on SimQN v0.1.5, leveraging the simulator core, entities, and tools to enable simulation under dynamic, heterogeneous configurations. [2025]
- [Fidelity-Guaranteed-Entanglement-Routing](https://github.com/infonetlijian/Fidelity-Guaranteed-Entanglement-Routing) ⭐11: Complete simulation code for fidelity-guaranteed entanglement routing in quantum networks. [2025]
- [Load-Balancing-Quantum-Routing-SimQN](https://github.com/MLCL-SYSU/Load-balancing-quantum-routing-SimQN): Implementation of a load balancing quantum routing algorithm using SimQN. [2025]
- [QKDsim](https://github.com/zp2333/QKDsim) ⭐2: A QKD network simulation system implemented based on SimQN. [2025]
- [Q-cast](https://github.com/tanjiesheng/Q-cast): A quantum communication simulation platform built on SimQN, supporting multicast entanglement distribution and routing. [2025]

## How to contribute?
Welcome to contribute through GitHub Issue or Pull Requests. Please refer to the [develop guide](https://qnlab-ustc.github.io/SimQN/develop.html). If you have any questions, you are welcome to contact the developers via e-mail.

## License and Authors

SimQN is an open-source project under [GPLv3](/LICENSE) license. The authors of the paper include:
* Lutong Chen (ertuil), School of Cyber Science and Technology, University of Science and Technology of China, China. elliot.98@outlook.com
* Jian Li(infonetlijian), School of Cyber Science and Technology, University of Science and Technology of China, China. lijian9@ustc.edu.cn
* Kaiping Xue (kaipingxue), School of Cyber Science and Technology, University of Science and Technology of China, China. xue.kaiping@gmail.com & kpxue@ustc.edu.cn
* Nenghai Yu, School of Cyber Science and Technology, University of Science and Technology of China, China.
* Ruidong Li, Institute of Science and Engineering, Kanazawa University, Japan.
* Qibin Sun, School of Cyber Science and Technology, University of Science and Technology of China, China.
* Jun Lu, School of Cyber Science and Technology, University of Science and Technology of China, China.

Other contributors include:
* Zirui Xiao, School of Cyber Science and Technology, University of Science and Technology of China, China.
* Yuqi Yang, School of Cyber Science and Technology, University of Science and Technology of China, China.
* Bing Yang, School of Cyber Science and Technology, University of Science and Technology of China, China.
* Xumin Gao, School of Cyber Science and Technology, University of Science and Technology of China, China.
* Yuxin Chen, School of Information Science and Technology, University of Science and Technology of China, China.
* ShaoChuang Heng, School of Cyber Science and Technology, University of Science and Technology of China, China.
* Jianfeng Niu, School of Cyber Science and Technology, University of Science and Technology of China, China.

## Citation

Please cite this publication if you use SimQN in your research.

```Bibtex
@article{chen2023simqn,
  title={SimQN: A network-layer simulator for the quantum network investigation},
  author={Chen, Lutong and Xue, Kaiping and Li, Jian and Yu, Nenghai and Li, Ruidong and Sun, Qibin and Lu, Jun},
  journal={IEEE Network},
  volume={37},
  number={5},
  pages={182--189},
  year={2023},
  publisher={IEEE},
  doi={10.1109/MNET.130.2200481}
}
```

## Academic Paper using SimQN for Simulation

1. Li Z, Xue K, Li J, et al. DRM-ETP: Deadline-and-Reliability-Matched Entanglement Transport Protocol for Quantum Networks[J]. IEEE/ACM Transactions on Networking, 2026.
2. Li Z, Xue K, Li J, et al. Fidelity-Guaranteed Entanglement Routing in Quantum Networks[J]. IEEE Transactions on Mobile Computing, 2026.
3. Li Z, Xue K, Li J, et al. Multi-Entanglement Routing Design over Quantum Networks[J]. IEEE Transactions on Mobile Computing, 2026.
4. Li Z, Xue K, Li J, et al. Q-DDCA: A QoS-aware Dynamic Distributed Coordination Algorithm for Entanglement Distribution in Quantum Networks[C]//IEEE INFOCOM 2026.
5. Li Z, Xue K, Li J, et al. Redundant Entanglement Provisioning and Selection for Throughput Maximization in Quantum Networks[J]. IEEE/ACM Transactions on Networking, 2026.
6. Li Z, Xue K, Li J, et al. Structural Property of the Optimal Entanglement Policy for Quantum Network Switch: An MDP Approach[J]. IEEE/ACM Transactions on Networking, 2026.
7. Ikken N, Kumar P, Slaoui A, et al. Optimizing multi-hop quantum communication using bidirectional quantum teleportation protocol[J]. arXiv preprint arXiv:2504.07320, 2025.
8. Kumar P, Kar B. ZBR: Zone-based routing in quantum networks with efficient entanglement distribution[J]. Journal of Network and Computer Applications, 2025, 238: 104156.
9. Kumar P, Kar B, Shen S H. Trace-distance based end-to-end entanglement fidelity with information preservation in quantum networks[J]. Journal of Network and Computer Applications, 2025: 104366.
10. Chen Y, Li J, Li Z, et al. An Asynchronous Key Relay Protocol Design for Large-Scale Quantum Key Distribution Networks[J]. IEEE Transactions on Networking, 2025.
11. Zheng P, Li J, Li Z, et al. An efficient and robust resource allocation method for quantum key distribution networks[J]. IEEE Transactions on Network and Service Management, 2025.
12. Yang Y, Li Z, Li J, et al. An On-demand Routing Scheme with QoS Provisioning for QKD Networks[C]//2025 International Conference on Quantum Communications, Networking, and Computing (QCNC). IEEE, 2025: 224-231.
13. Tan J, Li Z, Li J, et al. Distributed Key-on-Demand Protocol Design for Dynamic Quantum Key Distribution Networks[C]//2025 QCNC. IEEE, 2025: 239-243.
14. Li Z, Xue K, Li J, et al. Fidelity-Guaranteed Entanglement Routing in Quantum Networks[M]//Security and Privacy Vision in 6G. Wiley, 2025.
15. Li Z, Xue K, Li J, et al. JARA: A Junbi-Aware Routing Algorithm for Entanglement Distribution in Quantum Networks[J]. IEEE/ACM Transactions on Networking, 2025.
16. Niu J, Li Z, Li J, et al. Path-Based Entanglement Verification in Quantum Networks[C]//2025 QCNC. IEEE, 2025: 213-220.
17. Li Z, Xue K, Li J, et al. Quantum Internet: A New World for Routing[C]//2025 QCNC. IEEE, 2025: 1-8.
18. Li Z, Xue K, Li J, et al. QuNet: Cost vector analysis and multi-path entanglement routing in quantum networks[J]. CCF Transactions on Networking, 2025.
19. Majid K, Agarwal S, Das S, et al. Quantum Network Simulators Integration with Open-Source 5G Mobile Stack[C]//IEEE Global Communications Conference (GLOBECOM), 2025.
20. Wu J, Chen L, Zhang J, et al. A distributed routing protocol based on key reservation in quantum key distribution networks[C]//ICC 2024-IEEE International Conference on Communications. IEEE, 2024: 509-514.
21. Li Z, Xue K, Li J, et al. Quantum BGP with Online Path Selection via Network Benchmarking[C]//IEEE INFOCOM 2024.
22. Chen L, Xue K, Li J, et al. Q-DDCA: Decentralized Dynamic Congestion Avoid Routing in Large-Scale Quantum Networks[J]. IEEE/ACM Transactions on Networking, 2024.
23. Li P, Li W. Traffic-Aware Routing Algorithm in Quantum Network[C]//International Conference on Communication, Computing & Security (ICCCS), 2024.
24. Xiao Z, Li J, Xue K, et al. A connectionless entanglement distribution protocol design in quantum networks[J]. IEEE Network, 2023, 38(1): 131-139.


