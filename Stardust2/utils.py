import os
import pandas as pd
import re
import json
import random
from pathlib import Path
import pdb
import csv
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import networkx as nx
from networkx.drawing.nx_pydot import from_pydot

from torch_geometric.nn import GraphSAGE, GATConv,SAGEConv, GCNConv, GINConv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from torch_geometric.utils import to_dense_adj

from networkx.drawing.nx_agraph import read_dot
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import torch
import torch.nn.functional as F
import itertools
from collections import defaultdict, namedtuple
import pydot
import gymnasium as gym
from simanneal import Annealer


device = torch.device("cpu")

import os
import re
import json
from collections import defaultdict
import pandas as pd

def checkTheClusterSize(model,data_dict,adj_matrixs,name):
    model.eval()
    with torch.no_grad():
        _, assignments, _ = model(data_dict[name].x, data_dict[name].edge_index, adj_matrixs[name])
    node_cluster_indices = torch.argmax(assignments, dim=1)
    current = torch.bincount(node_cluster_indices)[torch.bincount(node_cluster_indices).nonzero().view(-1)]
    if(torch.max(current)<544):
        return True
    else:
        return False


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


def plot_loss(file:str, isRL:bool,title="Training", xlabel="Samples", ylabel="Throughput"):
    # Make it 1D on CPU as float
    tensors=torch.load(file)
    ft=[]
    for i in range(len(tensors)):
        if(np.isnan(tensors[i])):
              continue
         #elif(tensors[i]<0):
         #     ft.append(-tensors[i])
         #else:
        ft.append(tensors[i])
    #pdb.set_trace()
    y = ft#loss_tensor.detach().flatten().to("cpu").float().numpy()
    x = range(len(y))
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    if(isRL):
        plt.savefig("RL.jpg")
    else:
        plt.savefig("OptNet.jpg")


def count_cycles_in_content(content):
    for line in content:
        if 'IMA halted at' in line:
            cycles = int(line.split()[-2])
            return cycles
    return 0


def parse_inst(inst_str):
    try:
        inst_str = inst_str.replace("'", '"')
        inst_str = re.sub(r',\s*}', '}', inst_str)
        inst = json.loads(inst_str)
        return inst
    except json.JSONDecodeError:
        return None


def list_directories(path='.'):
    entries = os.listdir(path)
    directories = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]
    return directories


def analyze_graphs(base_dir='.'):

    graph_dirs = list_directories(base_dir)
    all_transfers = []
    all_cycles = []

    for graph_dir in graph_dirs:
        graph_path = os.path.join(base_dir, graph_dir)
        if os.path.isdir(graph_path):
            print(f"Processing graph: {graph_dir}")
            transfers, cycles = analyze_tile_ima_transfers(graph_path, graph_dir)
            all_transfers.extend(transfers)
            all_cycles.extend(cycles)


    df_transfers = pd.DataFrame(all_transfers)
    df_cycles = pd.DataFrame(all_cycles)

    df_transfers.to_csv('data_transfers_summary.csv', index=False)
    df_cycles.to_csv('cycles_count_summary.csv', index=False)

    return {'transfers': df_transfers, 'cycles': df_cycles}

def analyze_tile_ima_transfers(graph_path, graph_name):
    cycle_pattern = re.compile(r'^Cycle (\d+)$')
    exe_pattern   = re.compile(r'^Exe \| Inst: (.*)curr_vec: (\d+) \| Flags: (.*)$')

    # Global across the whole graph:
    addr_owners = defaultdict(set)  # addr -> {tile_ima_key,...}
    addr_users  = defaultdict(set)  # addr -> {tile_ima_key,...}
    tile_ima_cycles = {}            # tile_ima_key -> cycle_count

    for tile_name in os.listdir(graph_path):
        tile_path = os.path.join(graph_path, tile_name)
        if not (os.path.isdir(tile_path) and tile_name.startswith("tile")):
            continue

        tile_id = tile_name
        print(f"  Processing {tile_id} in graph {graph_name}")

        ima_files = [f for f in os.listdir(tile_path)
                     if f.startswith("ima_trace") and f.endswith(".txt")]

        for filename in ima_files:
            ima_id = filename.replace("ima_trace", "").replace(".txt", "")
            filepath = os.path.join(tile_path, filename)

            tile_ima_key = f"{graph_name}_{tile_id}_ima_{ima_id}"
            print(f"Processing IMA {ima_id} from file {filename}")

            #cycle_count = 0  # or track max cycle if you prefer
            max_cycle = -1

            with open(filepath, "r") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue

                    m = cycle_pattern.match(line)
                    #pdb.set_trace()
                    if m:
                        #cycle_count += 1
                        c = int(m.group(1))
                        if c > max_cycle: 
                            max_cycle = c
                            continue

                    m = exe_pattern.match(line)
                    if not m:
                        continue

                    inst_str = m.group(1)
                    # curr_vec = int(m.group(2))  # you don't use curr_vec downstream
                    inst = parse_inst(inst_str)
                    if not inst:
                        continue

                    opcode = inst.get("opcode")
                    if opcode not in ("ld", "st"):
                        continue

                    imm = inst.get("imm")
                    mem_addr = None
                    if isinstance(imm, int):
                        mem_addr = imm
                    elif isinstance(imm, str):
                        try:
                            mem_addr = int(imm, 0)
                        except ValueError:
                            mem_addr = None
                    if mem_addr is None:
                        continue

                    vec_length = int(inst.get("vec", 1))

                    # Update sets directly: NO mem_access list
                    if opcode == "st":
                        owners = addr_owners
                        for i in range(vec_length):
                            owners[mem_addr + i].add(tile_ima_key)
                    else:  # "ld"
                        users = addr_users
                        for i in range(vec_length):
                            users[mem_addr + i].add(tile_ima_key)

            #tile_ima_cycles[tile_ima_key] = cycle_count
            # If you want "total cycles" as last index + 1:
            tile_ima_cycles[tile_ima_key] = max_cycle + 1

    # Aggregate transfers
    data_width = 16
    element_size = data_width // 8  # 2 bytes
    data_transfers = defaultdict(lambda: defaultdict(int))

    for addr, owners in addr_owners.items():
        users = addr_users.get(addr)
        if not users:
            continue
        for owner in owners:
            for user in users:
                if owner != user:
                    data_transfers[owner][user] += element_size

    transfer_records = []
    for src, targets in data_transfers.items():
        for dst, bytes_ in targets.items():
            print(f"Data transferred from {src} to {dst}: {bytes_} bytes")
            transfer_records.append({
                "Graph": graph_name,
                "Source": src,
                "Destination": dst,
                "Data Transferred (bytes)": bytes_,
            })

    cycles_records = [{"Graph": graph_name, "Tile_IMA": k, "Cycle Count": v}
                      for k, v in tile_ima_cycles.items()]

    return transfer_records, cycles_records





def find_files_with_extension(path='.', extension='.dot'):
    entries = os.listdir(path)
    files = [file for file in entries if file.endswith(extension) and os.path.isfile(os.path.join(path, file))]
    return files


def find_keys_by_value(d, value):
    return [k for k, v in d.items() if v == value]

def build_dict(G_dict):
    node_name_to_node_dict = {}
    for name, G in G_dict.items():
        node_name_to_node = {node.get_name().replace("\\n", "\n"): str(node).replace("\\n", "\n") for node in G.get_nodes()}
        node_name_to_node_dict[name] = node_name_to_node
    return node_name_to_node_dict

def remove_cycles(G):
    try:
        while True:
            cycle = nx.find_cycle(G, orientation='original')
            G.remove_edge(cycle[-1][0], cycle[-1][1])
    except nx.NetworkXNoCycle:
        pass
    return G


def pad_features(features, max_len):
    print(features.size(0))
    return torch.cat([features, torch.zeros(max_len - features.size(0))])

def find_longest_path_dag(nodes, edges):
    longest_path_to = {node: 0 for node in nodes}

    predecessor = {node: None for node in nodes}

    for node in nodes:

        for next_node in edges.get(node, []):

            if longest_path_to[next_node] < longest_path_to[node] + 1:
                longest_path_to[next_node] = longest_path_to[node] + 1
                predecessor[next_node] = node

    end_node = max(longest_path_to, key=longest_path_to.get)

    path = []
    while end_node is not None:
        path.append(end_node)
        end_node = predecessor[end_node]

    return path[::-1]

def getKey(dic, value):
    result = set()
    for key in dic:
        for v in dic[key]:
            if v == value:
                result.add(key)
    return result


def reclustering(originalindice,newgraphx,newgraphedge,adj,model,
    node_table_dict,send_table_dict,node_mapping_dict,cluster_index,
    successor_map,originalcluster,graphprocessor,new_G_d):
    maxindex=0
    maxindexo=0
    count=0
    missing=[]
    missingo=[]
    missingoindex=0
    print(cluster_index)
    for key in cluster_index:
        if(cluster_index[key]>maxindex):
            maxindex=cluster_index[key]
        if(key>maxindexo):
            maxindexo=key
        if(cluster_index[key]!=count):
            missing.append(int(count))
            count=cluster_index[key]+1
        else:
            count+=1
    for i in range(maxindexo):
        if(i in cluster_index.keys()):
            continue
        else:
            missingo.append(i)
    with torch.no_grad():
        features_pooled, assignments, loss = model(newgraphx, newgraphedge, adj)
    node_cluster_indices = torch.argmax(assignments, dim=1)


    clusterid=torch.unique(node_cluster_indices)
    if(clusterid.size(0)<2):
        onethird=node_cluster_indices.size(0)//3 
        node_cluster_indices[:onethird]=missingo[missingoindex]
        node_cluster_indices[onethird:int(onethird*2)]=missingo[missingoindex+1] 
        node_cluster_indices[int(onethird*2):]=missingo[missingoindex+2]
        missingoindex+=3
        clusterid=torch.unique(node_cluster_indices)
    for i in range(clusterid.size(0)):
        if(clusterid[i].item() in cluster_index.keys()):
            for j in range(node_cluster_indices.size(0)):
                if(node_cluster_indices[j]==clusterid[i]):
                    node_cluster_indices[j]=missingo[missingoindex]
            missingoindex+=1

    originalcluster[originalindice]=node_cluster_indices
    indice=originalcluster.detach().cpu().numpy()
    unique_indice = np.unique(indice)

    start=0
    for cluster in unique_indice:
        if(cluster in cluster_index.keys()):
            continue
        if(start<len(missing)):
            cluster_index[int(cluster)]=int(missing[start])
            start+=1
        else:
            cluster_index[int(cluster)]=int(maxindex+1)
            maxindex+=1
    #pdb.set_trace()
    cluster_index = dict(sorted(cluster_index.items(), key=lambda x: x[1]))
    print(cluster_index)
    node_table={}
    send_table={}
    for i in cluster_index:
        #if cluster_index[i] in node_table_dict.keys():
        #    node_table[cluster_index[i]] = node_table_dict[cluster_index[i]]
        #    send_table[cluster_index[i]] = send_table_dict[cluster_index[i]]
        #else:
        node_table[cluster_index[i]] = []
        send_table[cluster_index[i]] = []

    #pdb.set_trace()
    for i, item in enumerate(node_mapping_dict.keys()):
        #if(node_mapping_dict[item] in originalindice):
        node_table[cluster_index[indice[i]]].append(item)
    node_table_dict = node_table

    #pdb.set_trace()
    for key, values in successor_map.items():
        if values != []:
            for value in values:
                if cluster_index[indice[key]] != cluster_index[indice[value]]:
                    send_table[cluster_index[indice[key]]].append(cluster_index[indice[value]])
    send_table_dict = send_table
    #pdb.set_trace()

    critical_path_times = graphprocessor.calculate_class_critical_paths(new_G_d, node_table)
    shortest_path_times = graphprocessor.calculate_class_shortest_paths(new_G_d, node_table)
    critical_path_times_dict = critical_path_times
    #print("iteration {}: {}".format(counter,torch.bincount(node_cluster_indices)))
    #print("iteration {}, loss {}".format(counter,loss))
    node_cluster_i=originalcluster
    solutions=originalcluster
    feature_table = {cluster_index[i]: [] for i in cluster_index}
    #pdb.set_trace()
    for i in feature_table:
        for j in cluster_index:
            if(cluster_index[j]==i):
                feature_table[i]=features_pooled[j]
    #pdb.set_trace()
    super_node_features = feature_table #model.aggregate_features_max_pooling(feature_table)
    super_graph = graphprocessor.create_supergraph(node_table, new_G_d.edges)
    topological_order = list(nx.topological_sort(super_graph))
    topo_dict = topological_order

    for node in super_graph.nodes:
        super_graph.nodes[node]['time'] = critical_path_times[node]
        super_graph.nodes[node]['shortest']=shortest_path_times[node]
    all_graphs,level,degree = graphprocessor.create_all_graphs(super_graph)
    #pdb.set_trace()
    graphs_datas = {
                graph_name: graphprocessor.create_torch_geometric_data_from_networkx(graph, super_node_features)
                for graph_name, graph in all_graphs.items()}
    super_graph_dict = super_graph
    all_graphs_dict = all_graphs
    graphs_datas_dict = graphs_datas
    #levels=features
    #pdb.set_trace()
    return cluster_index,super_graph_dict,all_graphs_dict,graphs_datas_dict,critical_path_times_dict,send_table_dict,node_table_dict,topo_dict,level,degree


def recluster(node_cluster_indices,model,data_dict,name,
cluster_index,node_table_dict,send_table_dict,
node_mapping_dict,successor_map_dict,graphprocessor,new_G_d):
    clustersize=torch.bincount(node_cluster_indices)
    largeclusters=(clustersize>300).nonzero().view(-1)
    super_graph_dict={}
    all_graphs_dict={}
    graphs_data_dict={}
    critical_path_time_dict={}
    topo_dict={}
    for t in range(largeclusters.size(0)):
        largecluster=largeclusters[t]
        mask=torch.isin(node_cluster_indices,largecluster)
        originalIndices=(mask==True).nonzero().view(-1)
        #clean the original partition in node_table_dict:
        #for i in range(largecluster.size(0)):
        #pdb.set_trace()
        del node_table_dict[cluster_index[largeclusters[t].item()]]
        del send_table_dict[cluster_index[largeclusters[t].item()]]
        del cluster_index[largeclusters[t].item()]
        #rebuild the graph structure for reclustering
        originaledge=data_dict[name].edge_index
        newgraphx=data_dict[name].x[originalIndices]
        newgraphedgei=[]
        newgraphedgej=[]
        for i in range(originaledge.size(1)):
            if(torch.isin(originaledge[0,i],originalIndices) and torch.isin(originaledge[1,i],originalIndices)):
                newindexi=((originalIndices==originaledge[0,i]).nonzero().view(-1))[0].item()
                newindexj=((originalIndices==originaledge[1,i]).nonzero().view(-1))[0].item()
                newgraphedgei.append(newindexi)
                newgraphedgej.append(newindexj)
        newedgei=torch.tensor(newgraphedgei)
        newedgej=torch.tensor(newgraphedgej)
        newgraphedge=torch.stack((newedgei,newedgej),dim=0).to(torch.long)
        #put the new rebuild graph into GNN clustering 
        #pdb.set_trace()
        adj = to_dense_adj(newgraphedge,max_num_nodes=newgraphx.size(0))[0]
        cluster_index,super_graph_dict,all_graphs_dict,graphs_data_dict,critical_path_time_dict,send_table_dict,node_table_dict,topo_dict,level,degree=reclustering(originalIndices,newgraphx,newgraphedge,adj,model,
        node_table_dict,send_table_dict,node_mapping_dict,cluster_index,
        successor_map_dict,node_cluster_indices,graphprocessor,new_G_d)
        #pdb.set_trace()
    return super_graph_dict,all_graphs_dict,graphs_data_dict,critical_path_time_dict,send_table_dict,node_table_dict,topo_dict,level,degree
    #print("new assignment size is: ", torch.bincount(output))
    #pdb.set_trace()
    #print("oversized cluster is ",largecluster)