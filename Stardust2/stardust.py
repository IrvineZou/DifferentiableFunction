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
from finetune import *
from CostheadBaseline import DifferentiableChipMapper,train_chip_mapper

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)



def inferenceRL(env, agent, load_path, node_limit_map, w, num_episodes=10):
        checkpoint = torch.load(load_path, map_location=device)
        agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        agent.policy.eval()
        total_rewards = []
        total_time = []
        for episode in range(num_episodes):
            observation,time1 , _ = env.reset()
            total_time.append(time1)
            observation = torch.tensor(observation).to(device)
            dones = False
            episode_reward = 0
            inst=0
            maps={}
            while not dones:
                with torch.no_grad():
                    #logits_node, logits_position, _ = policy_network(observation.unsqueeze(0))
                    #action_node = torch.argmax(logits_node, dim=1).item()
                    #action_position = torch.argmax(logits_position, dim=1).item()
                    #action = torch.tensor([action_node, action_position])
                    #with torch.no_grad():
                    action, log_prob, value = agent.select_action(observation)
                    action_np = action.cpu().numpy()
                inst+=1
                #print(action)
                #if(action[0] not in maps.keys()):
                maps[action[0].item()]=action[1].item()
                #else:
                if(inst>=50000):
                     break
                next_observation, reward, done, _, _ = env.step2(action_np,False)
                dones=done
                #print(reward)
                episode_reward += reward
                observation = torch.tensor(next_observation).to(device)

            #total_rewards.append(episode_reward)
            #pdb.set_trace()
            print("Policy is:",maps)
            maps = dict(sorted(maps.items()))
            maps_temp=torch.tensor([maps[key] for key in maps])
            maps = GreedyAdjust(maps_temp,node_limit_map,w)
            maps={i:maps[i].item() for i in range(len(maps))}
            env.reset()
            totals=0
            for i in maps:
                tu=(i,maps[i])
                next_observation,reward,done,_,_=env.step2(tu,False)
                totals+=reward
            print("total throughput: ",totals)
            total_rewards.append(totals)

        avg_reward = np.mean(total_rewards)
        avg_time = np.mean(total_time)
        print(f"average_reward: {avg_reward}")
        print(f"average_time: {avg_time}")
        return avg_reward

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

def calculateCommWeight(send_table,env):
    comm={}
    edge=[]
    for key in send_table:
        for i in range(len(send_table[key])):
            path = env.calculate_path(key, send_table[key][i])
            data_size = 32
            if((key,send_table[key][i]) not in comm.keys()):
                keyt=tuple((key,send_table[key][i]))
                comm[keyt]=0
                comm[(key,send_table[key][i])] = env.calculate_routing_time(path[0], path[1],data_size)
                edge.append(keyt)
            else:
                comm[(key,send_table[key][i])] += env.calculate_routing_time(path[0], path[1],data_size)
    return comm,edge



def inferences(isRL,node_limit,hetro,filep,graphpath,loadpath,encoder,args):
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
        cluster_index[name] = {int(v): k for k, v in cluster_index_inversed.items()}
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
        shortest_path_times = graphprocessor.calculate_class_shortest_paths(new_G_d[name], node_table)
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
            super_graph.nodes[node]['shortest']=critical_path_times[node]
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




    if(isRL==0):
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
        if(args.differentible==1):
            with torch.no_grad():
                dn.setEnv(env)
                outputs,caps,g=dn(hs,w,cap,delays)
        else:
            for name in graphs_datas_dict:
                num_nodes=w.size(0)
                #pdb.set_trace()
                for key in graphs_datas_dict[name]:
                    model = DifferentiableChipMapper(
                        num_nodes=num_nodes,
                        num_chips=cap.size(0),
                        node_sizes=w,
                        chip_capacities=cap,
                        edges=graphs_datas_dict[name][key].edge_index.t(),
                    )

                    history, final_out = train_chip_mapper(
                        model,
                        epochs=200,
                        lr=0.1,
                        tau_start=3.0,
                        tau_end=0.3,
                        overflow_weight=20.0,
                        comm_weight=1.0,
                        balance_weight=0.1,
                        infeasible_step_weight=2.0,
                        verbose=True,
                    )
                    outputs=final_out['assignments']
        totals=[]
        for _ in range(1):
            observation,time1 , _ = env.reset(False)
            #print(torch.argmin(outputs,dim=-1))
            throughput1=inferenceDiff(env,torch.argmin(outputs,dim=-1))
            #print(throughput1)
            action=GreedyAdjust(torch.argmin(outputs,dim=-1),node_limit_chip,w)
            print(action)
            throughput=inferenceDiff(env,action)
            totals.append(throughput)
            print(time1+time)
        print("average reward: ", np.mean(totals))
        return np.mean(totals)
    elif(isRL==1):
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

        observation_space = env.observation_space
        action_space = env.action_space

        policy_net = PolicyNetwork(observation_space, action_space)
        policy_net.to(device)

        #optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
        agent = PPOAgent(env,policy_net,None,None)
        w=torch.tensor([])
        for name in node_cluster_i:
            wt=torch.bincount(node_cluster_i[name])[torch.bincount(node_cluster_i[name]).nonzero().view(-1)]
            w=torch.cat((w,wt),dim=0)
        reward=inferenceRL(env,agent,filep,node_limit_chip,w,num_episodes=5)
        return reward
    else:
        env = GraphEnv(
                        G_super=super_graph_dict,
                        all_graphs=all_graphs_dict,
                        graphs_datas=graphs_datas_dict,
                        graph_processor=graphprocessor,
                        time_map=critical_path_times_dict,
                        send_table=send_table_dict,
                        node_table=node_table_dict,
                        topo_dict=topo_dict,
                        num_nodes_limt=node_limit_chip
        )
        #pdb.set_trace()
        comm,edge=calculateCommWeight(send_table,env)
        #pdb.set_trace()
        delay=[]
        #_,critical_path_time_dict2=env.compute_node_times(graphs_datas['Incomparable Graph'],critical_path_times)
        for key in critical_path_times_dict:
            for t in critical_path_times_dict[key]:
                delay.append(critical_path_times_dict[key][t])
        #pdb.set_trace()
        cap=[]
        for key in node_limit_chip:
            cap.append(node_limit_chip[key])
        clustersize=[]
        #for name in node_cluster_i:
        #pdb.set_trace()
        wt=torch.bincount(node_cluster_i[name])[torch.bincount(node_cluster_i[name]).nonzero().view(-1)]
        w=list(wt)
        for i,itemw in enumerate(w):
            itemw=itemw.item()
            w[i]=itemw
        #pdb.set_trace()
        num_tasks=len(w)
        result=solve_workload_mapping_ilp(
            num_tasks,
            64,
            w,
            cap,
            edge,
            delay,
            send_table,
            alpha=1.0,
            beta=1.0,
            time_limit=None,
            msg=False,
        )
        print(result)
        assignment=result['assignment']
        pdb.set_trace()
        throughput=inferenceDiff(env,torch.tensor(assignment))
        #print(throughput)
        return throughput


def GreedyAdjust(actions, caps, nodesize):
    # caps: dict {chip: cap} or tensor/list length D
    layout_cols=8
    if isinstance(caps, dict):
        chip_ids = sorted(caps.keys())
        cap_t = torch.tensor(
            [caps[k] for k in chip_ids],
            dtype=torch.float32,
            device=nodesize.device
        )
    else:
        chip_ids = list(range(len(caps)))
        cap_t = torch.as_tensor(
            caps,
            dtype=torch.float32,
            device=nodesize.device
        )

    D = len(chip_ids)
    actions = actions.clone()
    nodesize = nodesize.to(torch.float32)

    # Feasibility check
    if nodesize.sum() > cap_t.sum() + 1e-6:
        raise ValueError("Infeasible: total nodesize > total capacity, cannot fix overloads.")

    clusters = {c: 0.0 for c in range(D)}
    nodes = {c: [] for c in range(D)}

    # Build current loads
    for n in range(len(actions)):
        c = int(actions[n].item())
        w = float(nodesize[n].item())
        clusters[c] += w
        nodes[c].append(n)

    # Remaining capacity: can be negative for overloaded chips
    chipremaining = cap_t.clone()
    for c in range(D):
        chipremaining[c] = cap_t[c] - clusters[c]

    # Judge whether there exists overload
    isload = False
    location = 0
    for c in range(D):
        if clusters[c] > float(cap_t[c]) + 1e-6:
            isload = True
            location = c
            break

    if not isload:
        print("there is no overload")
        return actions
    else:
        print("there is overload happened on", location)

    # Precompute chip coordinates for 8x8 row-major layout
    chip_indices = torch.arange(D, device=nodesize.device)
    chip_rows = chip_indices // layout_cols
    chip_cols = chip_indices % layout_cols

    # Fix overloads
    for c in range(D):
        while clusters[c] > float(cap_t[c]) + 1e-6:
            if len(nodes[c]) == 0:
                break

            # Move the smallest operator cluster from overloaded chip
            # Smallest is easier to fit into nearby chips
            smallest_pos = torch.argmin(nodesize[nodes[c]]).item()
            node_idx = nodes[c][smallest_pos]
            w = float(nodesize[node_idx].item())

            # Feasible destination chips must have enough remaining capacity
            feasible = chipremaining >= w - 1e-6
            feasible[c] = False

            if feasible.any():
                # Manhattan distance from overloaded chip c to every chip
                src_row = chip_rows[c]
                src_col = chip_cols[c]

                dist = torch.abs(chip_rows - src_row) + torch.abs(chip_cols - src_col)

                # First priority: closest chip
                feasible_dist = dist.to(torch.float32).masked_fill(~feasible, float("inf"))
                min_dist = feasible_dist.min()

                closest_feasible = feasible & (dist == min_dist)

                # Second priority: among closest chips, choose largest remaining capacity
                remaining_score = chipremaining.masked_fill(~closest_feasible, float("-inf"))
                dst = int(torch.argmax(remaining_score).item())

            else:
                # No chip can fit this node right now
                break

            # Move node_idx from c to dst
            actions[node_idx] = dst

            clusters[c] -= w
            clusters[dst] += w

            chipremaining[c] += w
            chipremaining[dst] -= w

            nodes[c].pop(smallest_pos)
            nodes[dst].append(node_idx)

    return actions


def agent(hetro,epoch,node_limit,time_step, save_path, save_every,train,num_episode,load_path,isRL,graphpath,args,encoder):
    super_graph_dict = {}
    all_graphs_dict = {}
    graphs_datas_dict = {}
    new_G_d = {}
    data_dict={}
    successors_map_tm_dict={}
    node_mapping_dict={}
    adj_matrixs={}
    cluster_index={}
    counter = 0
    critical_path_times_dict = {}
    send_table_dict = {}
    node_table_dict = {}
    topo_dict = {}
    node_cluster_i={}
    node_feature_dict_dictt={}
    losst=[]
    time_t=[]
    time_o=[]
    n_clusters = 400
    if hetro:
        node_limit_chip = constants.hetro_limit
    else:
        node_limit_chip = constants.hom_limit
    for graph in graphpath:
        p = Path(graph)
        if p.exists():
            new_G_dt,data_dictt,node_feature_dict_dictt,lat_dict_dict,successors_map_tm_dictt,node_mapping_dictt,adj_matrixst=loads(graph)
            new_G_d[list(new_G_dt.keys())[0]]=new_G_dt[list(new_G_dt.keys())[0]]
            data_dict[list(new_G_dt.keys())[0]]=data_dictt[list(new_G_dt.keys())[0]]
            successors_map_tm_dict[list(new_G_dt.keys())[0]]=successors_map_tm_dictt[list(new_G_dt.keys())[0]]
            node_mapping_dict[list(new_G_dt.keys())[0]]=node_mapping_dictt[list(new_G_dt.keys())[0]]
            adj_matrixs[list(new_G_dt.keys())[0]]=adj_matrixst[list(new_G_dt.keys())[0]]
            #node_feature_dict_dict[list(new_G_dt.keys())[0]]=node_feature_dict_dictt[list(new_G_dt.keys())[0]]
        else:
            dotprocessor = DotProcessor('.')
            #pdb.set_trace()
            print("Reading files")
            dotprocessor.read_dot_files()
            dotprocessor.process_dot_files_0()
            dotprocessor.process_dot_files_5()
            dotprocessor.clear_graph5()
            dotprocessor.clear_graph0()
            graph0s_f = dotprocessor.graph0s_f
            graph5s_pydot = dotprocessor.graph5s_pydot
            graph_builder = GraphBuilder(G_dict={}, new_G_dict={}, graph0s_f=graph0s_f, graph5s_pydot=graph5s_pydot)
            graph_builder.generate_graph()
            graph=storage(graph_builder,graph)
            new_G_d,data_dict,node_feature_dict_dict,lat_dict_dict,successors_map_tm_dict,node_mapping_dict,adj_matrixs=loads(graph)
    model = DMoN(256, 256, 64, max_cluster_size=node_limit, n_clusters=n_clusters).to(device)
    if not train:
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    optimizer1 = torch.optim.Adam(model.parameters(), lr=1e-5)
    if(encoder==2):
        uses=False
    else:
        uses=True
    graphprocessor = GraphProcessor(64, 64, 1, 8,encoder,use_sage=uses)
    optimizerg = torch.optim.Adam(graphprocessor.parameters(),lr=1e-3,weight_decay=1e-5)

    if(args.differentible==1):
        dn=DifferentiableNetwork(64,64)
    #else:
    #   dn=DifferentiableChipMapper()
        dn.to(device)
        optimizer2=torch.optim.Adam(dn.parameters(),lr=1e-3,weight_decay=1e-5)
    
    solutions={}
    levels={}
    if train:
        model.train()
        while counter < 100:
            for name in new_G_d.keys():
                name = re.sub(r'^\./', '', name)
                if(counter==99):
                    with torch.no_grad():
                        features_pooled, assignments, loss = model(data_dict[name].x, data_dict[name].edge_index, adj_matrixs[name])
                else:
                    features_pooled, assignments, loss = model(data_dict[name].x, data_dict[name].edge_index, adj_matrixs[name])
                node_cluster_indices = torch.argmax(assignments, dim=1)
                cluster_index_inversed = {}
                indice = node_cluster_indices.detach().cpu().numpy()
                unique_indice = np.unique(indice)
                for i in range(len(unique_indice)):
                    cluster_index_inversed[i] = unique_indice[i]
                time_table = {}
                cluster_index[name] = {int(v): k for k, v in cluster_index_inversed.items()}
                cluster_indexs=cluster_index[name]
                node_table_item = {i: [] for i in range(len(unique_indice))}
                node_table = {i: [] for i in range(len(unique_indice))}
                send_table = {i: [] for i in range(len(unique_indice))}
                for i, item in enumerate(node_mapping_dict[name].keys()):
                    node_table[cluster_indexs[indice[i]]].append(item)
                node_table_dict[name] = node_table
                for i, item in enumerate(node_mapping_dict[name].keys()):
                    node_table_item[cluster_indexs[indice[i]]].append(node_mapping_dict[name][item])
                
                for key, values in successors_map_tm_dict[name].items():
                    if values != []:
                        for value in values:
                            if cluster_indexs[indice[key]] != cluster_indexs[indice[value]]:
                                send_table[cluster_indexs[indice[key]]].append(cluster_indexs[indice[value]])
                send_table_dict[name] = send_table
                for i in range(n_clusters):
                    time_table[i] = 0
                critical_path_times = graphprocessor.calculate_class_critical_paths(new_G_d[name], node_table)
                critical_path_times_dict[name] = critical_path_times
                #print("iteration {}: {}".format(counter,torch.bincount(node_cluster_indices)))
                print("iteration {}, loss {}".format(counter,loss))
                node_cluster_i[name]=node_cluster_indices
                solutions[name]=node_cluster_indices
                if(counter<99):
                    loss.backward()
                    optimizer1.step()
                    optimizer1.zero_grad()
                feature_table = {i: [] for i in range(len(unique_indice))}
                for i in feature_table:
                    for j in cluster_indexs:
                        if(cluster_indexs[j]==i):
                            feature_table[i]=features_pooled[j]
                super_node_features = feature_table
                super_graph = graphprocessor.create_supergraph(node_table, new_G_d[name].edges)
                topological_order = list(nx.topological_sort(super_graph))
                topo_dict[name] = topological_order
                for node in super_graph.nodes:
                    super_graph.nodes[node]['time'] = critical_path_times[node]
                #pdb.set_trace()
                all_graphs,level,_ = graphprocessor.create_all_graphs(super_graph)
                graphs_datas = {
                    graph_name: graphprocessor.create_torch_geometric_data_from_networkx(graph, super_node_features)
                    for graph_name, graph in all_graphs.items()}
                super_graph_dict[name] = super_graph
                all_graphs_dict[name] = all_graphs
                graphs_datas_dict[name] = graphs_datas
                cluster_index[name]=cluster_indexs
                levels[name]=level
            counter+=1
    else:
        #while(counter<epoch):
        for name in new_G_d.keys():
            name = re.sub(r'^\./', '', name)
            #with torch.no_grad():
            features_pooled, assignments, loss = model(data_dict[name].x, data_dict[name].edge_index, adj_matrixs[name])
            node_cluster_indices = torch.argmax(assignments, dim=1)
            cluster_index_inversed = {}
            indice = node_cluster_indices.detach().cpu().numpy()
            unique_indice = np.unique(indice)
            for i in range(len(unique_indice)):
                cluster_index_inversed[i] = unique_indice[i]
            time_table = {}
            cluster_index[name] = {int(v): k for k, v in cluster_index_inversed.items()}
            cluster_indexs=cluster_index[name]
            node_table_item = {i: [] for i in range(len(unique_indice))}
            node_table = {i: [] for i in range(len(unique_indice))}
            send_table = {i: [] for i in range(len(unique_indice))}
            for i, item in enumerate(node_mapping_dict[name].keys()):
                node_table[cluster_indexs[indice[i]]].append(item)
            node_table_dict[name] = node_table
            for i, item in enumerate(node_mapping_dict[name].keys()):
                node_table_item[cluster_indexs[indice[i]]].append(node_mapping_dict[name][item])
            #pdb.set_trace()
            for key, values in successors_map_tm_dict[name].items():
                if values != []:
                    for value in values:
                        if cluster_indexs[indice[key]] != cluster_indexs[indice[value]]:
                            send_table[cluster_indexs[indice[key]]].append(cluster_indexs[indice[value]])
            #pdb.set_trace()
            send_table_dict[name] = send_table
            for i in range(n_clusters):
                time_table[i] = 0

            critical_path_times = graphprocessor.calculate_class_critical_paths(new_G_d[name], node_table)
            shortest_path_times = graphprocessor.calculate_class_shortest_paths(new_G_d[name], node_table)
            critical_path_times_dict[name] = critical_path_times
            #print("iteration {}: {}".format(counter,torch.bincount(node_cluster_indices)))
            #print("iteration {}, loss {}".format(counter,loss))
            node_cluster_i[name]=node_cluster_indices
            solutions[name]=node_cluster_indices
            feature_table = {i: [] for i in range(len(unique_indice))}
            for i in feature_table:
                for j in cluster_indexs:
                    if(cluster_indexs[j]==i):
                        feature_table[i]=features_pooled[j]
            #pdb.set_trace()
            super_node_features = feature_table#model.aggregate_features_max_pooling(feature_table)
            #pdb.set_trace()
            super_graph = graphprocessor.create_supergraph(node_table, new_G_d[name].edges)
            topological_order = list(nx.topological_sort(super_graph))
            topo_dict[name] = topological_order

            #pdb.set_trace()
            for node in super_graph.nodes:
                super_graph.nodes[node]['time'] = critical_path_times[node]
                super_graph.nodes[node]['shortest']=shortest_path_times[node]
            all_graphs,level,_ = graphprocessor.create_all_graphs(super_graph)
            #pdb.set_trace()
            graphs_datas = {
                graph_name: graphprocessor.create_torch_geometric_data_from_networkx(graph, super_node_features)
                for graph_name, graph in all_graphs.items()}
            super_graph_dict[name] = super_graph
            all_graphs_dict[name] = all_graphs
            graphs_datas_dict[name] = graphs_datas
            cluster_index[name]=cluster_indexs
            levels[name]=level

    for name in new_G_d.keys():
        #print(name)
        clustersize=torch.bincount(solutions[name])
        largecluster=(clustersize>300).nonzero().view(-1)
        mask=torch.isin(solutions[name],largecluster)
        originalIndices=(mask==True).nonzero().view(-1)
        #print("Final solution:")
        #for key in node_table_dict[name]:
        #    print("Cluster:{}--size:{}".format(key,len(node_table_dict[name][key])))
        #pdb.set_trace()
        if(originalIndices.size(0)==0):
            print("This partition is OK, skip")
            continue
        super_graph_dict[name],all_graphs_dict[name],graphs_datas_dict[name],critical_path_times_dict[name],send_table_dict[name],node_table_dict[name],topo_dict[name],levels[name],_ = recluster(solutions[name],model,data_dict,name,
        cluster_index[name],node_table_dict[name],send_table_dict[name],
        node_mapping_dict[name],successors_map_tm_dict[name],
        graphprocessor,new_G_d[name])
        #for key in node_table_dict[name]:
        #    print("Cluster:{}--size:{}".format(key,len(node_table_dict[name][key])))

    for name in new_G_d.keys():
        print(name)
        #pdb.set_trace()
        for key in node_table_dict[name]:
            print("Cluster:{}--size:{}".format(key,len(node_table_dict[name][key])))
    
    #pdb.set_trace()
    #define policy network and agent:
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
    #pdb.set_trace()
    observation_space = env.observation_space
    action_space = env.action_space
    policy_net = PolicyNetwork(observation_space, action_space)
    policy_net.to(device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-8)
    agent = PPOAgent(env, policy_net, optimizer)
    if(args.differentible==0):
        #num of nodes,num_chips,node_sizes,chip_capacities
        n_nodes=0
        num_chips=64
        chip_capacities=torch.zeros(num_chips).to(torch.long)
        wt=[]
        #pdb.set_trace()
        for name in new_G_d.keys():
            for key in graphs_datas_dict[name]:
                n_nodes=graphs_datas_dict[name][key].x.size(0)
            node_sizes=torch.zeros(n_nodes).to(torch.long)
            for key in node_table_dict[name]:
                node_sizes[key]=len(node_table_dict[name][key])
            for key in node_limit_chip:
                chip_capacities[key]=node_limit_chip[key]
            for key in graphs_datas_dict[name]:
                edges=graphs_datas_dict[name][key].edge_index.t()
            for key in node_table_dict[name]:
                wt.append(len(node_table_dict[name][key]))
            wt=torch.tensor(wt,device=device)
            #pdb.set_trace()
            dn=DifferentiableChipMapper(n_nodes,num_chips,node_sizes,chip_capacities,edges=edges)
        _,final=train_chip_mapper(dn)
        action=GreedyAdjust(torch.tensor(final['chosen_chips']),node_limit_chip,wt)
        print(action)
        throughput=inferenceDiff(env,action)
        print(throughput)
        return
        #totals.append(throughput)
        #pdb.set_trace()
    else:
        losst=[]
        bestthroughput=0
        while(counter<epoch):
            throughputt=[]
            #pdb.set_trace()
            for name in new_G_d.keys():
                name = re.sub(r'^\./', '', name)
                #with torch.no_grad():
                features_pooled, assignments, loss = model(data_dict[name].x, data_dict[name].edge_index, adj_matrixs[name])
                node_cluster_indices = torch.argmax(assignments, dim=1)
                cluster_index_inversed = {}
                indice = node_cluster_indices.detach().cpu().numpy()
                unique_indice = np.unique(indice)
                for i in range(len(unique_indice)):
                    cluster_index_inversed[i] = unique_indice[i]
                time_table = {}
                cluster_index[name] = {v: k for k, v in cluster_index_inversed.items()}
                cluster_indexs=cluster_index[name]
                node_table_item = {i: [] for i in range(len(unique_indice))}
                node_table = {i: [] for i in range(len(unique_indice))}
                send_table = {i: [] for i in range(len(unique_indice))}
                for i, item in enumerate(node_mapping_dict[name].keys()):
                    node_table[cluster_indexs[indice[i]]].append(item)
                node_table_dict[name] = node_table
                for i, item in enumerate(node_mapping_dict[name].keys()):
                    node_table_item[cluster_indexs[indice[i]]].append(node_mapping_dict[name][item])
                #pdb.set_trace()
                for key, values in successors_map_tm_dict[name].items():
                    if values != []:
                        for value in values:
                            if cluster_indexs[indice[key]] != cluster_indexs[indice[value]]:
                                send_table[cluster_indexs[indice[key]]].append(cluster_indexs[indice[value]])
                send_table_dict[name] = send_table
                for i in range(n_clusters):
                    time_table[i] = 0

                critical_path_times = graphprocessor.calculate_class_critical_paths(new_G_d[name], node_table)
                shortest_path_times = graphprocessor.calculate_class_shortest_paths(new_G_d[name], node_table)
                critical_path_times_dict[name] = critical_path_times
                #print("iteration {}: {}".format(counter,torch.bincount(node_cluster_indices)))
                print("iteration {}, loss {}".format(counter,loss))
                node_cluster_i[name]=node_cluster_indices
                solutions[name]=node_cluster_indices
                feature_table = {i: [] for i in range(len(unique_indice))}
                for i in feature_table:
                    for j in cluster_indexs:
                        if(cluster_indexs[j]==i):
                            feature_table[i]=features_pooled[j]
                #pdb.set_trace()
                super_node_features = feature_table #model.aggregate_features_max_pooling(feature_table)
                #pdb.set_trace()
                super_graph = graphprocessor.create_supergraph(node_table, new_G_d[name].edges)
                topological_order = list(nx.topological_sort(super_graph))
                topo_dict[name] = topological_order

                #pdb.set_trace()
                for node in super_graph.nodes:
                    super_graph.nodes[node]['time'] = critical_path_times[node]
                    super_graph.nodes[node]['shortest']=shortest_path_times[node]
                all_graphs,level,_ = graphprocessor.create_all_graphs(super_graph)
                #pdb.set_trace()
                graphs_datas = {
                    graph_name: graphprocessor.create_torch_geometric_data_from_networkx(graph, super_node_features)
                    for graph_name, graph in all_graphs.items()}
                super_graph_dict[name] = super_graph
                all_graphs_dict[name] = all_graphs
                graphs_datas_dict[name] = graphs_datas
                cluster_index[name]=cluster_indexs
                levels[name]=level

                if(isRL==0):
                    wt=[]
                    cap = []
                    delay = []
                    #pdb.set_trace()
                    for key in node_limit_chip:
                        cap.append(node_limit_chip[key])
                    for t in critical_path_times_dict[name]:
                        delay.append(critical_path_times_dict[name][t])
                    cap = torch.tensor(cap,device=device)
                    delays=torch.tensor(delay,device=device,dtype=torch.float32)
                    min = delays.min()
                    max = delays.max()
                    delays = (delays - min) / (max-min)
                    delays = torch.clamp(delays, -1.0, 1.0) / 1.0 * 100.0
                    for key in node_table_dict[name]:
                        wt.append(len(node_table_dict[name][key]))
                    wt=torch.tensor(wt,device=device)
                    graphs_datas=graphprocessor(all_graphs_dict[name],graphs_datas_dict[name],levels[name]).to(device)
                    hs=graphs_datas
                    if(args.differentible==1):
                        N = hs.size(0)
                        dn.setEnv(env)
                        outputs,caps,g=dn(hs,wt,cap,delays)
                        cap_pen = caps
                        loss2 = cap_pen
                        print("{}--:{}".format(counter,loss2))
                            #print("Solution is: {}".format(torch.argmin(outputs,dim=-1)))
                        loss2.backward()
                        optimizer2.step()
                        optimizer2.zero_grad()
                        optimizerg.step()
                        optimizerg.zero_grad()
                        #pdb.set_trace()
                        print("Solution is: {}".format(torch.argmin(outputs,dim=-1)))
                        action=GreedyAdjust(torch.argmin(outputs,dim=-1),node_limit_chip,wt)
                        throughput=inferenceDiff(env,action)
                        throughputt.append(throughput)
                    
                    counter+=1
            losst.append(np.mean(throughputt))
            #if(np.mean(throughputt)>bestthroughput):
            bestthroughput=np.mean(throughputt)
            checkpoint = {
                            #'model_state_dict': model.state_dict(),
                            'optimizer_model_state_dict': optimizer1.state_dict(),
                            'policy_state_dict': dn.state_dict(),
                            'optimizer_state_dict': optimizer2.state_dict(),
                            'graphprocessor': graphprocessor.state_dict(),
            }
            torch.save(checkpoint, 'models/dn_epoch_{}_difi.pth'.format(args.dataset))
            checkpoint2 = {
                                'model_state_dict': model.state_dict(),
            }
            torch.save(checkpoint2,load_path)
            print(f"Saved checkpoint at epoch {epoch}")
    if(isRL):
        torch.save(losst,"RL.pt")
        plot_loss("RL.pt",True)
    else:
        losst=torch.tensor(losst)
        torch.save(losst,"OptNet.pt")
        plot_loss("OptNet.pt",False)

    path='epoch_500.pth'
    print("average topoformer execution time is: ", np.mean(time_t))
    print("average operator cluster time:",np.mean(time_o))
    print(f"Training completed. Model saved to {path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--train" ,help ="If train is true, train agent",default=True)
    parser.add_argument("-he","--heterogeneity" ,help ="Use heterogeneous structure ",default=True)
    parser.add_argument("-ti","--time_step" ,help ="Train time steps for agent",default=1000)
    parser.add_argument("-e","--epoch" ,help ="Train epoch for clustering model",default=300)
    parser.add_argument("-s", "--save_path", help="model save path", default=constants.model_save_path)
    parser.add_argument("-l","--load_path", help="model load path", default=constants.model_load_path)
    parser.add_argument("-n","--num_episodes" ,help ="Number of episodes",default=100)
    parser.add_argument("-se","--save_every" ,help ="save_every",default=50)
    parser.add_argument("-nl", "--node_limit", help="node limit number in cluster", default=272)
    parser.add_argument("-dl", "--dataset",type=str)
    parser.add_argument("-encoder","--encoder",type=int)
    parser.add_argument("-check","--check",type=int)
    parser.add_argument("-diff","--differentible",type=int)
    args = parser.parse_args()

    encoder=args.encoder
    checks=args.check
    #pdb.set_trace()
    if(checks==0):
        #graphs=["./graphs/mha.pt","./graphs/mlp.pt","./graphs/lstm.pt"] #,"./graphs/T5.pt"]
        graphs=["./graphs/nmt.pt"] #,"./graphs/mlp.pt","./graphs/lstm.pt","./graphs/mha.pt","./graphs/DeBERTa.pt","./graphs/GPT2.pt"]
        #graphs=["./graphs/GPT_Neo.pt"]
        load_path="models/cluster_path_{}.pth".format(args.dataset)
        agent(True,800,272,500,'.',1,False,100,load_path,False,graphs,args,encoder)
        infergraph=["./graphs/nmt.pt","./graphs/DeBERTa-large.pt","./graphs/mlp-l5.pt","./graphs/T5.pt","./graphs/GPT2_medium.pt","./graphs/GPT_Neo.pt"] #,"./graphs/GPT2.pt","./graphs/GPT2_medium.pt"]
        totalresults=[]
        for i in range(len(infergraph)):
            infergrapht=[infergraph[i]]
            print(infergrapht)
            t=inferences(0,64,True,"models/dn_epoch_{}_difi.pth".format(args.dataset),infergrapht,load_path,encoder,args)
            totalresults.append(t)
        print(totalresults)
    elif(checks==1):
        finetunegraph=["./graphs/lstm.pt"]
        load_path="models/cluster_path_{}.pth".format(args.dataset)
        finetuning(0,64,True,"models/dn_epoch_{}_difi.pth".format(args.dataset),[finetunegraph[0]],load_path,encoder)
    else:
        graphs=["./graphs/GPT2_medium.pt","./graphs/T5.pt","./graphs/mlp-l5.pt","./graphs/DeBERTa-large.pt","./graphs/GPT2.pt","./graphs/GPT2_medium.pt"] #,"./graphs/T5.pt"]
        load_path="models/cluster_path_{}.pth".format(args.dataset)
        model=TrainTopo(0,64,True,"models/dn_epoch_{}_difi.pth".format(args.dataset),[graphs[0]],load_path,encoder)
        #inferenceTopo(0,64,True,"models/dn_epoch_{}_difi.pth".format(args.dataset),[graphs[0]],load_path,encoder,model)


