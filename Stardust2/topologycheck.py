import constants
from builder import *
from clusterd_graph_processor import *
from dmon import *
from processor import *
from customEnv import *
from ppo import *
import argparse
from Costhead import DifferentiableNetwork
import pdb
from pathlib import Path
import copy
from ilp import *
from utils import *
from collections import deque

class NodeLevelLinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_levels: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_levels)
        #self.drop=nn.Dropout(0.1)

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        """
        node_emb: [num_nodes, input_dim]
        returns:  [num_nodes, num_levels]
        """
        return self.fc(node_emb)


def TrainTopo(isRL,node_limit,hetro,filep,graphpath,loadpath,encoder):
    #update the parameter
    super_graph_dict = {}
    all_graphs_dict = {}
    graphs_datas_dict = {}
    n_clusters = 400

    if hetro:
        node_limit_chip = constants.hetro_limit
    else:
        node_limit_chip = constants.hom_limit
    model = DMoN(256, 256, 64, max_cluster_size=node_limit, n_clusters=n_clusters).to(device)
    model.load_state_dict(torch.load(loadpath)['model_state_dict'])
    model.eval()
    if(encoder==2):
        uses=False
    else:
        uses=True
    graphprocessor = GraphProcessor(64, 64, 1, 8,encoder,output_dim=1,use_sage=uses)
    #graphprocessor.load_state_dict(torch.load(filep)['graphprocessor'])
    graphprocessor.eval()
    dn=DifferentiableNetwork(64,64)
    #prob=NodeLevelLinearProbe(64,32).to(device)
    if(isRL==False):
         dn.load_state_dict(torch.load(filep)['policy_state_dict'])
    dn.eval()
    new_G_d = {} 
    counter = 0
    new_G_d = {}
    data_dict={}
    successors_map_tm_dict={}
    node_mapping_dict={}
    adj_matrixs={}
    send_table_dict = {}
    node_table_dict = {}
    node_feature_dict_ebeddeed = {}
    critical_path_times_dict = {}
    topo_dict = {}
    node_cluster_i={}
    cluster_index={}
    levels={}
    degrees={}
    for graph in graphpath:
        p = Path(graph)
        if p.exists():
            #pdb.set_trace()
            new_G_dt,data_dictt,node_feature_dict_dict,lat_dict_dict,successors_map_tm_dictt,node_mapping_dictt,adj_matrixst=loads(graph)
            new_G_d[list(new_G_dt.keys())[0]]=new_G_dt[list(new_G_dt.keys())[0]]
            data_dict[list(new_G_dt.keys())[0]]=data_dictt[list(new_G_dt.keys())[0]]
            successors_map_tm_dict[list(new_G_dt.keys())[0]]=successors_map_tm_dictt[list(new_G_dt.keys())[0]]
            node_mapping_dict[list(new_G_dt.keys())[0]]=node_mapping_dictt[list(new_G_dt.keys())[0]]
            adj_matrixs[list(new_G_dt.keys())[0]]=adj_matrixst[list(new_G_dt.keys())[0]]
        else:
            #graph_builder.generate_graph()
            #storage(graph_builder,graphpath)
            new_G_d,data_dict,node_feature_dict_dict,lat_dict_dict,successors_map_tm_dict,node_mapping_dict,adj_matrixs=loads(graph)
    solutions={}
    for name in new_G_d.keys():
        name_cleaned = re.sub(r'^\./', '', name)
        data_dict[name_cleaned]=data_dict[name_cleaned].to(device)
        adj_matrixs[name_cleaned]=adj_matrixs[name_cleaned].to(device)
        with torch.no_grad():
            features_pooled, assignments, loss = model(data_dict[name_cleaned].x, data_dict[name_cleaned].edge_index,
                                                       adj_matrixs[name_cleaned])
        node_cluster_indices = torch.argmax(assignments, dim=1)
        node_cluster_i[name]=node_cluster_indices
        solutions[name]=node_cluster_indices
        indice = node_cluster_indices.detach().cpu().numpy()
        unique_indice = np.unique(indice)
        cluster_index_inversed = {i: unique_indice[i] for i in range(len(unique_indice))}
        cluster_index[name] = {v: k for k, v in cluster_index_inversed.items()}
        cluster_indexs=cluster_index[name]
        node_table = {i: [] for i in range(len(unique_indice))}
        send_table = {i: [] for i in range(len(unique_indice))}

        for idx, node_key in enumerate(node_mapping_dict[name_cleaned].keys()):
            node_table[cluster_indexs[indice[idx]]].append(node_key)

        node_table_dict[name_cleaned] = node_table

        for key, values in successors_map_tm_dict[name_cleaned].items():
            if values:
                    for value in values:
                        if cluster_indexs[indice[key]] != cluster_indexs[indice[value]]:
                            send_table[cluster_indexs[indice[key]]].append(cluster_indexs[indice[value]])
        send_table_dict[name_cleaned] = send_table
        critical_path_times = graphprocessor.calculate_class_critical_paths(new_G_d[name], node_table)
        critical_path_times_dict[name_cleaned] = critical_path_times

        critical_path_times = graphprocessor.calculate_class_critical_paths(new_G_d[name], node_table)
        critical_path_times_dict[name_cleaned] = critical_path_times

        feature_table = {i: [] for i in range(len(unique_indice))}
            
        for a in feature_table:
            for b in cluster_indexs:
                if(cluster_indexs[b]==a):
                    feature_table[a]=features_pooled[b]

        super_node_features = model.aggregate_features_max_pooling(feature_table)

        super_graph = graphprocessor.create_supergraph(node_table, new_G_d[name].edges)
        topological_order = list(nx.topological_sort(super_graph))
        topo_dict[name_cleaned] = topological_order
        #pdb.set_trace()
        for node in super_graph.nodes:
            super_graph.nodes[node]['time'] = critical_path_times[node]
        #pdb.set_trace()
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
            with record_function("aggregate"):
                all_graphs,level,degree = graphprocessor.create_all_graphs(super_graph)
        a = prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10)
        #pdb.set_trace()
        aheads = a.find("Self CPU time total")
        back = a[aheads:].find("s")
        judges = a[aheads:aheads + back]
        if (judges[len(judges) - 1] == 'u'):
           time = float(a[aheads + 21: aheads + back - 1]) / 1000
        else:
           time = float(a[aheads + 21: aheads + back - 1])
        graphs_datas = {
                graph_name: graphprocessor.create_torch_geometric_data_from_networkx(graph, super_node_features)
                for graph_name, graph in all_graphs.items()
        }
        #update critical_path_time_dict
        #pdb.set_trace()
        #_,critical_path_times=compute_node_times(graphs_datas['Incomparable Graph'],critical_path_times)
        super_graph_dict[name_cleaned] = super_graph
        all_graphs_dict[name_cleaned] = all_graphs
        graphs_datas_dict[name_cleaned] = graphs_datas
        cluster_index[name_cleaned] = cluster_indexs
        levels[name_cleaned]=level
        degrees[name_cleaned]=degree
    
    for name in new_G_d.keys():
        clustersize=torch.bincount(solutions[name])
        largecluster=(clustersize>300).nonzero().view(-1)
        mask=torch.isin(solutions[name],largecluster)
        originalIndices=(mask==True).nonzero().view(-1)
        #print("Final solution:")
        #for key in node_table_dict[name]:
        #    print("Cluster:{}--size:{}".format(key,len(node_table_dict[name][key])))
        if(originalIndices.size(0)==0):
            print("This partition is OK, skip")
            continue
        super_graph_dict[name],all_graphs_dict[name],graphs_datas_dict[name],critical_path_times_dict[name],send_table_dict[name],node_table_dict[name],topo_dict[name],levels[name],degrees[name] = recluster(solutions[name],model,data_dict,name,
        cluster_index[name],node_table_dict[name],send_table_dict[name],
        node_mapping_dict[name],successors_map_tm_dict[name],
        graphprocessor,new_G_d[name])
        
    names=list(all_graphs_dict.keys())
    graphs_datas=torch.tensor([])
    optimizertopo=torch.optim.Adam(graphprocessor.parameters(), lr=1e-6,weight_decay=1e-8)
    epoch=900
    count=0
    bestmodel=1000
    best=None
    for i in range(len(names)):
        #pdb.set_trace()
        degrees[name]=degrees[name].to(device)
        #with torch.no_grad():
        labels=torch.zeros(degrees[name].size())
        for key in critical_path_times_dict[names[i]]:
            labels[key]=critical_path_times_dict[names[i]][key]
        labels=(labels-min(labels))/(max(labels)-min(labels))
        while(count<epoch):
            graphs_data=graphprocessor(all_graphs_dict[names[i]],graphs_datas_dict[names[i]],levels[names[i]]).to(device)         
            mask = torch.zeros(graphs_data.size(0), dtype=torch.bool)
            mask[:int(graphs_data.size(0)*0.8)] = True
            loss=F.mse_loss(graphs_data[mask],labels[mask])
            #while(count<epoch):
            #    outputs=prob(graphs_data)
                #pdb.set_trace()
            #    degrees[name]=degrees[name].to(torch.long)
            #    loss = F.cross_entropy(outputs, degrees[name])
            loss.backward()
            optimizertopo.step()
        #    pred=torch.argmax(outputs,dim=-1)
        #    accuracy=torch.sum((pred==degrees[name]))/pred.size(0)
            if(loss<bestmodel):
                best=copy.deepcopy(graphprocessor)
                bestmodel=loss
        #    print(accuracy.item())
            print("loss:",loss.item())
            count+=1
    for i in range(len(names)):
        #pdb.set_trace()
        labels=torch.zeros(degrees[name].size())
        for key in critical_path_times_dict[names[i]]:
            labels[key]=critical_path_times_dict[names[i]][key]
        labels=(labels-min(labels))/(max(labels)-min(labels))
        mask = torch.zeros(graphs_data.size(0), dtype=torch.bool)
        mask[int(graphs_data.size(0)*0.8):] = True
        with torch.no_grad():
            graphs_data=best(all_graphs_dict[names[i]],graphs_datas_dict[names[i]],levels[names[i]]).to(device)
        mask = torch.zeros(graphs_data.size(0), dtype=torch.bool)
        mask[int(graphs_data.size(0)*0.8):] = True
        for i in range(labels.size(0)):
            if(labels[i]==0):
                mask[i]=False
        mape=torch.mean(torch.abs(graphs_data[mask]-labels[mask])/labels[mask])
        mse=torch.mean((graphs_data[mask]-labels[mask])**2)
        #pdb.set_trace()
        print("mape:{}%".format(mape*100))
        print("mse:",mse)
    return best


def inferenceTopo(isRL,node_limit,hetro,filep,graphpath,loadpath,encoder,probe):
    #update the parameter
    super_graph_dict = {}
    all_graphs_dict = {}
    graphs_datas_dict = {}
    n_clusters = 400

    if hetro:
        node_limit_chip = constants.hetro_limit
    else:
        node_limit_chip = constants.hom_limit
    model = DMoN(256, 256, 64, max_cluster_size=node_limit, n_clusters=n_clusters).to(device)
    model.load_state_dict(torch.load(loadpath)['model_state_dict'])
    model.eval()
    if(encoder==2):
        uses=False
    else:
        uses=True
    graphprocessor = GraphProcessor(64, 64, 1, 8,encoder,output_dim=1,use_sage=uses)
    graphprocessor.load_state_dict(torch.load(filep)['graphprocessor'])
    graphprocessor.eval()
    dn=DifferentiableNetwork(64,64)
    prob=probe
    if(isRL==False):
         dn.load_state_dict(torch.load(filep)['policy_state_dict'])
    dn.eval()
    new_G_d = {} 
    counter = 0
    new_G_d = {}
    data_dict={}
    successors_map_tm_dict={}
    node_mapping_dict={}
    adj_matrixs={}
    send_table_dict = {}
    node_table_dict = {}
    node_feature_dict_ebeddeed = {}
    critical_path_times_dict = {}
    topo_dict = {}
    node_cluster_i={}
    cluster_index={}
    levels={}
    degrees={}
    for graph in graphpath:
        p = Path(graph)
        if p.exists():
            #pdb.set_trace()
            new_G_dt,data_dictt,node_feature_dict_dict,lat_dict_dict,successors_map_tm_dictt,node_mapping_dictt,adj_matrixst=loads(graph)
            new_G_d[list(new_G_dt.keys())[0]]=new_G_dt[list(new_G_dt.keys())[0]]
            data_dict[list(new_G_dt.keys())[0]]=data_dictt[list(new_G_dt.keys())[0]]
            successors_map_tm_dict[list(new_G_dt.keys())[0]]=successors_map_tm_dictt[list(new_G_dt.keys())[0]]
            node_mapping_dict[list(new_G_dt.keys())[0]]=node_mapping_dictt[list(new_G_dt.keys())[0]]
            adj_matrixs[list(new_G_dt.keys())[0]]=adj_matrixst[list(new_G_dt.keys())[0]]
        else:
            #graph_builder.generate_graph()
            #storage(graph_builder,graphpath)
            new_G_d,data_dict,node_feature_dict_dict,lat_dict_dict,successors_map_tm_dict,node_mapping_dict,adj_matrixs=loads(graph)
    solutions={}
    for name in new_G_d.keys():
        name_cleaned = re.sub(r'^\./', '', name)
        data_dict[name_cleaned]=data_dict[name_cleaned].to(device)
        adj_matrixs[name_cleaned]=adj_matrixs[name_cleaned].to(device)
        with torch.no_grad():
            features_pooled, assignments, loss = model(data_dict[name_cleaned].x, data_dict[name_cleaned].edge_index,
                                                       adj_matrixs[name_cleaned])
        node_cluster_indices = torch.argmax(assignments, dim=1)
        node_cluster_i[name]=node_cluster_indices
        solutions[name]=node_cluster_indices
        indice = node_cluster_indices.detach().cpu().numpy()
        unique_indice = np.unique(indice)
        cluster_index_inversed = {i: unique_indice[i] for i in range(len(unique_indice))}
        cluster_index[name] = {v: k for k, v in cluster_index_inversed.items()}
        cluster_indexs=cluster_index[name]
        node_table = {i: [] for i in range(len(unique_indice))}
        send_table = {i: [] for i in range(len(unique_indice))}

        for idx, node_key in enumerate(node_mapping_dict[name_cleaned].keys()):
            node_table[cluster_indexs[indice[idx]]].append(node_key)

        node_table_dict[name_cleaned] = node_table

        for key, values in successors_map_tm_dict[name_cleaned].items():
            if values:
                    for value in values:
                        if cluster_indexs[indice[key]] != cluster_indexs[indice[value]]:
                            send_table[cluster_indexs[indice[key]]].append(cluster_indexs[indice[value]])
        send_table_dict[name_cleaned] = send_table
        critical_path_times = graphprocessor.calculate_class_critical_paths(new_G_d[name], node_table)
        critical_path_times_dict[name_cleaned] = critical_path_times

        critical_path_times = graphprocessor.calculate_class_critical_paths(new_G_d[name], node_table)
        critical_path_times_dict[name_cleaned] = critical_path_times

        feature_table = {i: [] for i in range(len(unique_indice))}
            
        for a in feature_table:
            for b in cluster_indexs:
                if(cluster_indexs[b]==a):
                    feature_table[a]=features_pooled[b]

        super_node_features = model.aggregate_features_max_pooling(feature_table)

        super_graph = graphprocessor.create_supergraph(node_table, new_G_d[name].edges)
        topological_order = list(nx.topological_sort(super_graph))
        topo_dict[name_cleaned] = topological_order
        #pdb.set_trace()
        for node in super_graph.nodes:
            super_graph.nodes[node]['time'] = critical_path_times[node]
        #pdb.set_trace()
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
            with record_function("aggregate"):
                all_graphs,level,degree = graphprocessor.create_all_graphs(super_graph)
        a = prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10)
        #pdb.set_trace()
        aheads = a.find("Self CPU time total")
        back = a[aheads:].find("s")
        judges = a[aheads:aheads + back]
        if (judges[len(judges) - 1] == 'u'):
           time = float(a[aheads + 21: aheads + back - 1]) / 1000
        else:
           time = float(a[aheads + 21: aheads + back - 1])
        graphs_datas = {
                graph_name: graphprocessor.create_torch_geometric_data_from_networkx(graph, super_node_features)
                for graph_name, graph in all_graphs.items()
        }
        #update critical_path_time_dict
        #pdb.set_trace()
        #_,critical_path_times=compute_node_times(graphs_datas['Incomparable Graph'],critical_path_times)
        super_graph_dict[name_cleaned] = super_graph
        all_graphs_dict[name_cleaned] = all_graphs
        graphs_datas_dict[name_cleaned] = graphs_datas
        cluster_index[name_cleaned] = cluster_indexs
        levels[name_cleaned]=level
        degrees[name_cleaned]=degree
    
    for name in new_G_d.keys():
        clustersize=torch.bincount(solutions[name])
        largecluster=(clustersize>300).nonzero().view(-1)
        mask=torch.isin(solutions[name],largecluster)
        originalIndices=(mask==True).nonzero().view(-1)
        #print("Final solution:")
        #for key in node_table_dict[name]:
        #    print("Cluster:{}--size:{}".format(key,len(node_table_dict[name][key])))
        if(originalIndices.size(0)==0):
            print("This partition is OK, skip")
            continue
        super_graph_dict[name],all_graphs_dict[name],graphs_datas_dict[name],critical_path_times_dict[name],send_table_dict[name],node_table_dict[name],topo_dict[name],levels[name],degrees[name] = recluster(solutions[name],model,data_dict,name,
        cluster_index[name],node_table_dict[name],send_table_dict[name],
        node_mapping_dict[name],successors_map_tm_dict[name],
        graphprocessor,new_G_d[name])
        
    names=list(all_graphs_dict.keys())
    graphs_datas=torch.tensor([])
    optimizertopo=torch.optim.Adam(prob.parameters(), lr=1e-3)
    epoch=250
    count=0
    bestmodel=0
    best=None
    for i in range(len(names)):
        #pdb.set_trace()
        labels=torch.zeros(degrees[name].size())
        for key in critical_path_times_dict[names[i]]:
            labels[key]=critical_path_times_dict[names[i]][key]
        labels=(labels-min(labels))/(max(labels)-min(labels))
        mask = torch.zeros(graphs_data.size(0), dtype=torch.bool)
        mask[int(graphs_data.size(0)*0.8):] = True
        with torch.no_grad():
            graphs_data=graphprocessor(all_graphs_dict[names[i]],graphs_datas_dict[names[i]],levels[names[i]]).to(device)
        mask = torch.zeros(graphs_data.size(0), dtype=torch.bool)
        mask[int(graphs_data.size(0)*0.8):] = True
        mape=torch.mean((graphs_data[mask]-labels[mask])**2/labels[mask])
        print("mape:{}%".format(mape*100))
