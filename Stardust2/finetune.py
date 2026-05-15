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
from topologycheck import *



def finetuning(isRL,node_limit,hetro,filep,graphpath,loadpath,encoder):
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
    graphprocessor = GraphProcessor(64, 64, 1, 8,encoder,use_sage=uses)
    graphprocessor.load_state_dict(torch.load(filep)['graphprocessor'])
    graphprocessor.eval()
    dn=DifferentiableNetwork(64,64)
    if(isRL==False):
         dn.load_state_dict(torch.load(filep)['policy_state_dict'])
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
                all_graphs,level,_ = graphprocessor.create_all_graphs(super_graph)
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
        super_graph_dict[name],all_graphs_dict[name],graphs_datas_dict[name],critical_path_times_dict[name],send_table_dict[name],node_table_dict[name],topo_dict[name],levels[name],_ = recluster(solutions[name],model,data_dict,name,
        cluster_index[name],node_table_dict[name],send_table_dict[name],
        node_mapping_dict[name],successors_map_tm_dict[name],
        graphprocessor,new_G_d[name])
        #print("Final solution:")
        #for key in node_table_dict[name]:
        #    print("Cluster:{}--size:{}".format(key,len(node_table_dict[name][key])))
    env = GraphEnv(
        G_super=super_graph_dict,
        all_graphs=all_graphs_dict,
        graphs_datas=graphs_datas_dict,
        graph_processor=graphprocessor,
        time_map=critical_path_times_dict,
        send_table=send_table_dict,
        node_table=node_table_dict,
        topo_dict=topo_dict,
        num_nodes_limt=node_limit_chip,
        level=levels
    )
        #hs_list = []
    cap = []
    delay = []
        
    names=list(all_graphs_dict.keys())
    graphs_datas=torch.tensor([])
            #pdb.set_trace()
    for i in range(len(names)):
            #levels[names[i]]=levels[names[i]].to(all_graphs_dict[names[i]].device)
        with torch.no_grad():
            graphs_data=graphprocessor(all_graphs_dict[names[i]],graphs_datas_dict[names[i]],levels[names[i]]).to(device)
        graphs_datas=torch.cat((graphs_datas,graphs_data),dim=0)
    hs=graphs_datas
    N = hs.size(0)
    for key in node_limit_chip:
        cap.append(node_limit_chip[key])
    for key in critical_path_times_dict:
        for t in critical_path_times_dict[key]:
            delay.append(critical_path_times_dict[key][t])
    #hs = torch.cat(hs_list, dim=-1).to(device)
    cap = torch.tensor(cap,device=device)
    delays=torch.tensor(delay,device=device,dtype=torch.float32)
    min = delays.min()
    max = delays.max()
    delays = (delays - min) / (max-min)
    delays = torch.clamp(delays, -3.0, 3.0) / 3.0 * 100.0
    N = hs.size(0)
    #pdb.set_trace()
    w=torch.tensor([])
    for name in node_table_dict:
        wt=[]
        for key in node_table_dict[name]:
            wt.append(len(node_table_dict[name][key]))#torch.bincount(node_cluster_i[name])[torch.bincount(node_cluster_i[name]).nonzero().view(-1)]
        wt=torch.tensor(wt)
        w=torch.cat((w,wt),dim=0)
    #with torch.no_grad():
    optimizer=torch.optim.Adam(dn.parameters(),lr=1e-4,weight_decay=1e-6)
    dn.setEnv(env)
    losst=[]
    for i in range(400):
        #pdb.set_trace()
        outputs,caps,g=dn(hs,w,cap,delays)
        loss=caps
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("epoch {}:{}".format(i,loss))
        action=GreedyAdjust(torch.argmin(outputs,dim=-1),node_limit_chip,wt)
        throughput=inferenceDiff(env,action)
        losst.append(throughput.item())
        #print("The throughput is ",throughput)
    torch.save(losst,"OptNet.pt")
    plot_loss("OptNet.pt",False)
    totals=[]
    for _ in range(1):
        observation,time1 , _ = env.reset(False)
        #throughput1=inferenceDiff(env,torch.argmin(outputs,dim=-1))
            
        action=GreedyAdjust(torch.argmin(outputs,dim=-1),node_limit_chip,w)
        print(action)
        throughput=inferenceDiff(env,action)
        totals.append(throughput)
        print(time1+time)
    print("average reward: ", np.mean(totals))


def inferenceDiff(env,actions):
    totals=0
    env.reset(False)
    #pdb.set_trace()
    for i in range(actions.size(0)):
        t=actions.cpu().numpy()
        tu=(i,t[i])
        #pdb.set_trace()
        next_observation,reward,done,_,_=env.step2(tu,False)
        totals+=reward
    print("total throughput: ",totals)
    return totals


def GreedyAdjust(actions, caps, nodesize):
    # caps: dict {chip: cap} or tensor/list length D
    if isinstance(caps, dict):
        chip_ids = sorted(caps.keys())
        cap_t = torch.tensor([caps[k] for k in chip_ids], dtype=torch.float32, device=nodesize.device)
    else:
        chip_ids = list(range(len(caps)))
        cap_t = torch.as_tensor(caps, dtype=torch.float32, device=nodesize.device)

    D = len(chip_ids)
    actions = actions.clone()
    nodesize = nodesize.to(torch.float32)

    # Feasibility check (prevents infinite loops)
    if nodesize.sum() > cap_t.sum() + 1e-6:
        raise ValueError("Infeasible: total nodesize > total capacity, cannot fix overloads.")

    clusters = {c: 0.0 for c in range(D)}
    nodes = {c: [] for c in range(D)}
    chipremaining = cap_t.clone()
    # Build current loads
    for n in range(len(actions)):
        c = int(actions[n].item())
        #pdb.set_trace()
        w = float(nodesize[n].item())
        clusters[c] += w
        nodes[c].append(n)
        chipremaining[c] = max(0.0, float(chipremaining[c]) - w)
    #judge whether there exists overloads on chips
    isload=False
    location=0
    for c in range(D):
        if(clusters[c]>float(cap_t[c])):
            isload=True
            location=c
            break
    if(not isload):
        print("there is no overload")
        return actions
    else:
        print("there is overload happened on ",location)
    # Fix overloads
    for c in range(D):
        while clusters[c] > float(cap_t[c]) + 1e-6:
            if len(nodes[c]) == 0:
                break  # nothing to move (shouldn't happen, but safe)

            # choose a node to move (smallest is ok; largest often converges faster)
            smallest_pos = torch.argmin(nodesize[nodes[c]]).item()
            node_idx = nodes[c][smallest_pos]
            w = float(nodesize[node_idx].item())

            # find a destination chip with enough remaining capacity for this node
            feasible = (chipremaining >= w - 1e-6)
            feasible[c] = False  # don't move to same chip
            if feasible.any():
                dst = int(torch.argmax(chipremaining * feasible).item())
            else:
                # no chip can fit this node right now -> give up to avoid deadlock
                # (or you can pick best-effort dst and accept overflow elsewhere)
                break
            # move
            actions[node_idx] = dst
            clusters[c] -= w
            clusters[dst] += w
            chipremaining[c] += w
            chipremaining[dst] -= w
            # update lists
            nodes[c].pop(smallest_pos)
            nodes[dst].append(node_idx)

    return actions