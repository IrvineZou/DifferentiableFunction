import os
import sys
import argparse
import re
import torch
import numpy as np
import networkx as nx
from pathlib import Path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from annealer import GymAnnealer
from genetic import GeneticAlgorithm
from randomsearch import RandomSearch

from dmon import DMoN
from processor import *
from clusterd_graph_processor import *
from builder import GraphBuilder
from customEnv import GraphEnv
from constants import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def storage(graph_builder, out_path: str):
    payload = {
        "data_dict": graph_builder.data_dict,
        "node_feature_dict_dict": graph_builder.node_feature_dict_dict,
        "lat_dict_dict": graph_builder.lat_dict_dict,
        "successors_map_tm_dict": graph_builder.successors_map_tm_dict,
        "node_mapping_dict": graph_builder.node_mapping_dict,
        "adj_matrix_dict": graph_builder.adj_matrix_dict,
        "new_G_d":graph_builder.new_G_dict
    }
    out_path = str(Path(out_path))
    torch.save(payload, out_path)
    return out_path

def loads(graphpath):
    ckpt = torch.load(graphpath, map_location="cpu")
    new_G_d = ckpt['new_G_d']
    data_dict = ckpt['data_dict']
    node_feature_dict_dict = ckpt['node_feature_dict_dict']
    lat_dict_dict = ckpt['lat_dict_dict']
    successors_map_tm_dict = ckpt['successors_map_tm_dict']
    node_mapping_dict = ckpt['node_mapping_dict']
    adj_matrix_dict = ckpt['adj_matrix' \
    '_dict']
    return new_G_d,data_dict,node_feature_dict_dict,lat_dict_dict,successors_map_tm_dict,node_mapping_dict,adj_matrix_dict



def main():
    parser = argparse.ArgumentParser(description='Select the optimization algorithm to run')
    parser.add_argument('--alg', type=str, choices=['annealing', 'genetic', 'random', 'simba'], required=True, help='Choose algorithm: annealing, genetic, random, simba')
    parser.add_argument('--iterations', type=int, default=800, help='Number of iterations (for random search and annealing)')
    parser.add_argument('--generations', type=int, default=200, help='Number of generations (for genetic algorithm)')
    parser.add_argument('--population_size', type=int, default=50, help='Population size (for genetic algorithm)')
    parser.add_argument('--mutation_rate', type=float, default=0.1, help='Mutation rate (for genetic algorithm)')
    parser.add_argument('--crossover_rate', type=float, default=0.8, help='Crossover rate (for genetic algorithm)')
    parser.add_argument("-nl", "--node_limit", type=int, default=272, help="Node limit number in cluster")
    args = parser.parse_args()

    all_results = {}
    filen="cluster_path_lstm.pth"
    filen2="dn_epoch_lstm_difi.pth"
    super_graph_dict = {}
    all_graphs_dict = {}
    graphs_datas_dict = {}
    n_clusters = 400

    model = DMoN(256, 256, 64, max_cluster_size=args.node_limit, n_clusters=n_clusters).to(device)
    optimizer1 = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.load_state_dict(torch.load("../Stardust2/models/{}".format(filen))['model_state_dict'])


    #dotprocessor = DotProcessor('.')
    #dotprocessor.read_dot_files()
    #dotprocessor.process_dot_files_0()
    #dotprocessor.process_dot_files_5()
    #dotprocessor.clear_graph5()
    #dotprocessor.clear_graph0()

    #graph0s_f = dotprocessor.graph0s_f
    #graph5s_pydot = dotprocessor.graph5s_pydot
    graphprocessor = GraphProcessor(64, 64, 1, 8)
    #pdb.set_trace()
    graphprocessor.load_state_dict(torch.load("../Stardust2/models/{}".format(filen2))['graphprocessor'])
    #graph_builder = GraphBuilder(G_dict={}, new_G_dict={}, graph0s_f=graph0s_f, graph5s_pydot=graph5s_pydot)
    new_G_d = {} #graph_builder.new_G_dict
    #graph_builder.generate_graph()

    #data_dict = graph_builder.data_dict
    #node_feature_dict_dict = graph_builder.node_feature_dict_dict
    #lat_dict_dict = graph_builder.lat_dict_dict
    #successors_map_tm_dict = graph_builder.successors_map_tm_dict
    #node_mapping_dict = graph_builder.node_mapping_dict
    #adj_matrixs = graph_builder.adj_matrix_dict
    graphpath="../graphs/nmt.pt"
    p = Path(graphpath)
    #pdb.set_trace()
    if p.exists():
        new_G_d,data_dict,node_feature_dict_dict,lat_dict_dict,successors_map_tm_dict,node_mapping_dict,adj_matrixs=loads(graphpath)
    else:
        graph_builder.generate_graph()
        storage(graph_builder,graphpath)
        new_G_d,data_dict,node_feature_dict_dict,lat_dict_dict,successors_map_tm_dict,node_mapping_dict,adj_matrixs=loads(graphpath)

    node_feature_dict_embedded = {}
    critical_path_times_dict = {}
    send_table_dict = {}
    node_table_dict = {}
    topo_dict = {}
    solutions = {}
    levels={}
    #for lon in range(50):
    for name in new_G_d.keys():
            name_cleaned = re.sub(r'^\./', '', name)
            #pdb.set_trace()
            data_dict[name_cleaned]=data_dict[name_cleaned].to(device)
            adj_matrixs[name_cleaned]=adj_matrixs[name_cleaned].to(device)
            with torch.no_grad():
                features_pooled, assignments, loss = model(data_dict[name_cleaned].x, data_dict[name_cleaned].edge_index,
                                                       adj_matrixs[name_cleaned])
            node_cluster_indices = torch.argmax(assignments, dim=1)
            solutions[name]=node_cluster_indices
            indice = node_cluster_indices.detach().cpu().numpy()
            unique_indice = np.unique(indice)
            cluster_index_inversed = {i: unique_indice[i] for i in range(len(unique_indice))}
            cluster_index = {v: k for k, v in cluster_index_inversed.items()}

            node_table = {i: [] for i in range(len(unique_indice))}
            send_table = {i: [] for i in range(len(unique_indice))}

            for idx, node_key in enumerate(node_mapping_dict[name_cleaned].keys()):
                node_table[cluster_index[indice[idx]]].append(node_key)

            node_table_dict[name_cleaned] = node_table

            for key, values in successors_map_tm_dict[name_cleaned].items():
                if values:
                    for value in values:
                        if cluster_index[indice[key]] != cluster_index[indice[value]]:
                            send_table[cluster_index[indice[key]]].append(cluster_index[indice[value]])
            send_table_dict[name_cleaned] = send_table

            critical_path_times = graphprocessor.calculate_class_critical_paths(new_G_d[name], node_table)
            critical_path_times_dict[name_cleaned] = critical_path_times

            critical_path_times = graphprocessor.calculate_class_critical_paths(new_G_d[name], node_table)
            critical_path_times_dict[name_cleaned] = critical_path_times
            #loss.backward()
            #optimizer1.step()
            #optimizer1.zero_grad()
            #print("epoch{}:{}".format(lon,loss))


            feature_table = {i: [] for i in range(len(unique_indice))}
            #for idx, (key, value) in enumerate(node_feature_dict_dict[name_cleaned].items()):
            #    for i in range(features_pooled.shape[0]):
            #        node_feature_dict_embedded[key] = features_pooled.detach()[i]
            #    feature_table[cluster_index[indice[idx]]].append(node_feature_dict_embedded[key])
            for a in feature_table:
                for b in cluster_index:
                    if(cluster_index[b]==a):
                        feature_table[a]=features_pooled[b]

            super_node_features = feature_table #model.aggregate_features_max_pooling(feature_table)

            super_graph = graphprocessor.create_supergraph(node_table, new_G_d[name].edges)
            topological_order = list(nx.topological_sort(super_graph))
            topo_dict[name_cleaned] = topological_order

            for node in super_graph.nodes:
                super_graph.nodes[node]['time'] = critical_path_times[node]
            #pdb.set_trace()
            all_graphs,level,_ = graphprocessor.create_all_graphs(super_graph)
            graphs_datas = {
                graph_name: graphprocessor.create_torch_geometric_data_from_networkx(graph, super_node_features)
                for graph_name, graph in all_graphs.items()
            }

            super_graph_dict[name_cleaned] = super_graph
            all_graphs_dict[name_cleaned] = all_graphs
            graphs_datas_dict[name_cleaned] = graphs_datas
            levels[name_cleaned]=level

    counts=torch.bincount(node_cluster_indices).to(torch.float32)
    print("variance is:",torch.std(counts))
    dicts={}
    print("final policy is:",torch.bincount(node_cluster_indices)[torch.bincount(node_cluster_indices).nonzero().view(-1)])

    for name in new_G_d.keys():
        clustersize=torch.bincount(solutions[name])
        largecluster=(clustersize>300).nonzero().view(-1)
        mask=torch.isin(solutions[name],largecluster)
        originalIndices=(mask==True).nonzero().view(-1)
        print("Final solution:")
        for key in node_table_dict[name]:
            print("Cluster:{}--size:{}".format(key,len(node_table_dict[name][key])))
        if(originalIndices.size(0)==0):
            print("This partition is OK, skip")
            continue
        super_graph_dict[name],all_graphs_dict[name],graphs_datas_dict[name],critical_path_times_dict[name],send_table_dict[name],node_table_dict[name],topo_dict[name],levels[name],_ = recluster(solutions[name],model,data_dict,name,
        cluster_index,node_table_dict[name],send_table_dict[name],
        node_mapping_dict[name],successors_map_tm_dict[name],
        graphprocessor,new_G_d[name])
        print("Final solution:")
        for key in node_table_dict[name]:
            print("Cluster:{}--size:{}".format(key,len(node_table_dict[name][key])))



    env = GraphEnv(
        G_super=super_graph_dict,
        all_graphs=all_graphs_dict,
        graphs_datas=graphs_datas_dict,
        graph_processor=graphprocessor,
        time_map=critical_path_times_dict,
        send_table=send_table_dict,
        node_table=node_table_dict,
        topo_dict=topo_dict,
        num_nodes_limt=hetro_limit,
        level=levels
    )

    current_graph_name = env.current_graph_name if hasattr(env, 'current_graph_name') else 'default_graph'

    if args.alg == 'annealing':
        initial_solution = {node: env.action_space.sample() for node in env.topoorder}
        annealer = GymAnnealer(initial_solution, env, env.topoorder)
        annealer.steps = args.iterations
        annealer.Tmax = 2000.0
        annealer.Tmin = 0.5
        best_state, best_energy = annealer.anneal()
        print("Simulated annealing completed.")
        print(f"Best state: {best_state}")
        print(f"Best energy: {best_energy}")
        results = annealer.results
        all_results[current_graph_name] = results

    elif args.alg == 'genetic':
        ga = GeneticAlgorithm(
            env,
            env.topoorder,
            population_size=args.population_size,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate
        )
        ga.evolve()
        print("Genetic algorithm completed.")
        print(f"Best individual: {ga.best_individual}")
        print(f"Best fitness: {ga.best_fitness}")
        #print(f"Achieved minimum time: {ga.min_time}")
        results = ga.results
        all_results[current_graph_name] = results

    elif args.alg == 'random':
        rs = RandomSearch(env, env.topoorder, iterations=args.iterations)
        rs.search()
        print("Random search completed.")
        print(f"Best individual: {rs.best_individual}")
        print(f"Best fitness: {rs.best_fitness}")
        #print(f"Achieved minimum time: {rs.min_time}")
        results = rs.results
        all_results[current_graph_name] = results

    #elif args.alg == 'simba':
    #    gs = GreedySearch(env, env.topoorder)
    #    gs.search()
    #    print("Greedy search completed.")
    #   print(f"Best individual: {gs.best_individual}")
    #    print(f"Total reward: {gs.total_reward}")
    #    print(f"Achieved minimum time: {gs.min_time}")
    #    results = gs.results
    #    all_results[current_graph_name] = results

if __name__ == "__main__":
    main()
