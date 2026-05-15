from utils import *
import pdb
from torch.profiler import profile, record_function, ProfilerActivity
from collections import deque


class GraphEnv(gym.Env):
    def __init__(self, **kwargs):
        super(GraphEnv, self).__init__()

        # Initialize parameters and data structures from kwargs
        self.G_super = kwargs.get('G_super', None)
        self.all_graphs = kwargs.get('all_graphs', None)
        self.graphs_datas = kwargs.get('graphs_datas', None)
        self.time_map_dict = kwargs.get('time_map', None)
        self.send_table_dict = kwargs.get('send_table', None)
        self.node_table_dict = kwargs.get('node_table', None)
        self.topoorder_dict = kwargs.get('topo_dict', None)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_nodes_limit = kwargs.get('num_nodes_limt', None)
        self.level = kwargs.get('level',None)
        # Initialize environment variables
        self.current_graph_index = 0
        self.feature_dim = 64  # Output dimension of GraphProcessor

        # Initialize the graph processor
        self.graph_processor = kwargs.get('graph_processor', None)


        # Read transfer data sizes from CSV file
        self.transfer_data_sizes = self.read_transfer_data_sizes('data_transfers_summary.csv')

        # Initialize environment state
        self.reset(True)

    def reset(self,isFirst):
        # Reset the environment to the initial state
        #graph_names = list(self.all_graphs.keys())
        #mappedorder={}
        #lengths=0
        #print(self.topoorder_dict)
        #for name in graph_names:
        #    mappedorder[name]={}
        #    for item in self.topoorder_dict[name]:
        #        mappedorder[name][item]=lengths+item
        #    lengths=len(self.topoorder_dict[name])
        #previouslength=0
        #self.num_nodes=0
        #self.topoorder=[]
        #self.node_table={}
        ##self.time_map={}
        #self.send_table={}
        #self.current_graph={}
        #pdb.set_trace()
        time1 =0
        #for name in graph_names:
        #    topo=self.topoorder_dict[name]
        #    for i in range(len(topo)):
        #        self.topoorder.append(topo[i]+previouslength)
        #    node_table = self.node_table_dict[name]
        #    for key in node_table:
        #        #if(key in mappedorder[name].keys()):
        #        self.node_table[mappedorder[name][key]]=node_table[key]
        #    time_map = self.time_map_dict[name]
        #    for key in time_map:
        #        #if(key in mappedorder[name].keys()):
        #        self.time_map[mappedorder[name][key]]=time_map[key]
        #    send_table = self.send_table_dict[name]
        #    for key in send_table:
                #if(key in mappedorder[name].keys()):
        #        self.send_table[mappedorder[name][key]]=send_table[key]
        #    current_graph = self.all_graphs[name]
            #pdb.set_trace()
        #    self.current_graph[name]={}
        #    for key in current_graph:
        #        self.current_graph[name][key]=current_graph[key]
        #    self.previouslength=len(topo)
        #    self.num_nodes+=len(topo)
        #    previouslength=len(topo)
        #pdb.set_trace()
        # Define action and observation spaces dynamically based on the current graph
        #self.grid_mapping = {}
        #self.action_space = gym.spaces.MultiDiscrete([self.num_nodes, self.feature_dim])
        #self.observation_space = gym.spaces.Box(
        #    low=-np.inf, high=np.inf,
        #    shape=(self.num_nodes, self.feature_dim * 2),  # Concatenated along the last dimension
        #    dtype=np.float32
        #)

        # Initialize other environment variables
        #self.i = 0  # Index for topoorder traversal
        #self.done = False
        #self.mapped = []
        #self.node_positions = {}  # Mapping from node to position (node index)

        # Initialize the state_embedding (environment's state for each node)
        #self.state_embedding = torch.full((self.num_nodes, self.feature_dim), -1, dtype=torch.float32).to(self.device)
        # Process the current graph using GraphProcessor

        #for key in self.graphs_datas:
        #    graph_data = self.graphs_datas[key]

        # Also, add the graph and node_ids to graph_data
        #graph_data.graph = current_graph
        #graph_data.node_ids = list(current_graph.nodes())

        #profile the topoformer
        #self.graph_processor_output=torch.tensor([],device=self.device)
        #pdb.set_trace()
        #with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        #    with record_function("aggregate"):
        #pdb.set_trace()
        #        for key in self.current_graph:
            #pdb.set_trace()
        #            graph_processor_o = self.graph_processor(self.current_graph[key],self.graphs_datas[key])  # Shape: (num_nodes, feature_dim)
            #graph_processor_o=graph_processor_o.to(self.device)
        #            self.graph_processor_output = torch.cat((self.graph_processor_output,graph_processor_o))
        #print(self.graph_processor_output)
        #a = prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10)
        #pdb.set_trace()
        #aheads = a.find("Self CPU time total")
        #back = a[aheads:].find("s")
        #judges = a[aheads:aheads + back]
        ##if (judges[len(judges) - 1] == 'u'):
         #   time1 = float(a[aheads + 21: aheads + back - 1]) / 1000
        #else:
        #    time1 = float(a[aheads + 21: aheads + back - 1])

        # Concatenate the state_embedding and graph_processor_output to form the initial state
        #self.state = torch.cat((self.state_embedding, self.graph_processor_output), dim=1)  # Shape: (num_nodes, 128)

        # Update the current graph index for the next reset
        #self.current_graph_index = (self.current_graph_index + 1) % len(graph_names)

        #return self.state, time1,{}
        graph_names = list(self.all_graphs.keys())
        self.current_graph_name = graph_names[self.current_graph_index]
        self.topoorder = self.topoorder_dict[self.current_graph_name]
        self.node_table = self.node_table_dict[self.current_graph_name]
        self.time_map = self.time_map_dict[self.current_graph_name]
        self.send_table = self.send_table_dict[self.current_graph_name]
        self.graph_data = self.graphs_datas[self.current_graph_name]
        current_graph = self.all_graphs[self.current_graph_name]
        self.currentlevel = self.level[self.current_graph_name]
        self.num_nodes = len(self.G_super[self.current_graph_name].nodes)
        self.grid_mapping = {}
        # Define action and observation spaces dynamically based on the current graph
        self.action_space = gym.spaces.MultiDiscrete([self.num_nodes, self.feature_dim])
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_nodes, self.feature_dim * 2),  # Concatenated along the last dimension
            dtype=np.float32
        )

        # Initialize other environment variables
        self.i = 0  # Index for topoorder traversal
        self.done = False
        self.mapped = []
        self.node_positions = {}  # Mapping from node to position (node index)

        # Initialize the state_embedding (environment's state for each node)
        self.state_embedding = torch.full((self.num_nodes, self.feature_dim), -1, dtype=torch.float32).to(self.device)

        # Process the current graph using GraphProcessor
        graph_data = self.graphs_datas[self.current_graph_name]
        #level = self.level[self.current_graph_name]
        # Also, add the graph and node_ids to graph_data
        #graph_data.graph = current_graph
        #graph_data.node_ids = list(current_graph.nodes())
        #pdb.set_trace()
        self.graph_processor.to(self.device)
        self.currentlevel=self.currentlevel.to(self.device)
        withprofile=True
        if(withprofile):
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
                with record_function("aggregate"):
                    self.graph_processor_output = self.graph_processor(current_graph,graph_data,self.currentlevel)  # Shape: (num_nodes, feature_dim)

            a = prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10)
            #pdb.set_trace()
            aheads = a.find("Self CPU time total")
            back = a[aheads:].find("s")
            judges = a[aheads:aheads + back]
            if (judges[len(judges) - 1] == 'u'):
                time1 = float(a[aheads + 21: aheads + back - 1]) / 1000
            else:
                time1 = float(a[aheads + 21: aheads + back - 1])
        else:
            self.graph_processor_output = self.graph_processor(current_graph,graph_data,self.currentlevel)
        # Concatenate the state_embedding and graph_processor_output to form the initial state
        self.state = torch.cat((self.state_embedding, self.graph_processor_output), dim=1)  # Shape: (num_nodes, 128)

        # Update the current graph index for the next reset
        if(isFirst==False):
            self.current_graph_index = (self.current_graph_index + 1) % len(graph_names)
        else:
            self.current_graph_index = self.current_graph_index
        return self.state, time1,{}

    def compute_node_times(self,data, time_dict):
        num_nodes = data.x.size(0)
        edge_index = data.edge_index

        # Build adjacency list and indegree
        adj = [[] for _ in range(num_nodes)]
        indegree = [0] * num_nodes

        for k in range(edge_index.size(1)):
            u = edge_index[0, k].item()
            v = edge_index[1, k].item()
            adj[u].append(v)
            indegree[v] += 1

        # Topological sort
        q = deque([i for i in range(num_nodes) if indegree[i] == 0])
        topo = []

        while q:
            u = q.popleft()
            topo.append(u)
            for v in adj[u]:
                indegree[v] -= 1
                if indegree[v] == 0:
                    q.append(v)

        if len(topo) != num_nodes:
            raise ValueError("Graph is not a DAG")

        # Earliest start/finish times
        start_time_dict = {i: 0 for i in range(num_nodes)}
        finish_time_dict = {i: 0 for i in range(num_nodes)}

        for u in topo:
            finish_time_dict[u] = start_time_dict[u] + time_dict[u]
            for v in adj[u]:
                start_time_dict[v] = max(start_time_dict[v], finish_time_dict[u])

        return start_time_dict, finish_time_dict



    def step(self, action,re):
        # When mapping a node to a position, update grid_mapping
        node, position = action  # Assuming position is a tuple (x, y) for 8x8 grid
        #pdb.set_trace()
        if self.i < len(self.topoorder):
            if node not in self.mapped:
                #if position not in self.node_positions.values():
                    # Map the node to the position
                if(position not in self.node_positions.values()):
                    reward = 0
                else:
                    reward = -1
                self.node_positions[node] = position
                self.grid_mapping[node] = position
                    #if node not in self.mapped:
                self.mapped.append(node)
                        # Update the state_embedding for the node
                self.state_embedding[node] = self.update_node_state(node, position)
                        # Reconstruct the state by concatenating state_embedding and graph_processor_output
                self.state = torch.cat((self.state_embedding, self.graph_processor_output), dim=1)
                #else:
                    # Penalty if the position is already occupied
                #reward = 0
                self.i += 1
            else:
            #    # Penalty for incorrect node mapping or node already mapped
                reward = -0.5
        else:
            reward = -0.5  # Penalty if all nodes have been processed


        if len(self.mapped) == self.num_nodes:
            self.done = True
            self.update_all_transfer_times()
            reward += self.compute_total_reward(re,False)
        else:
            self.done = False

        return self.state, reward, self.done, False, {}

    def step2(self, action,re):
        # When mapping a node to a position, update grid_mapping
        node, position = action  # Assuming position is a tuple (x, y) for 8x8 grid
        #pdb.set_trace()
        if self.i < len(self.topoorder):
            if node not in self.mapped:
                #if position not in self.node_positions.values():
                    # Map the node to the position
                self.node_positions[node] = position
                self.grid_mapping[node] = position
                    #if node not in self.mapped:
                self.mapped.append(node)
                        # Update the state_embedding for the node
                self.state_embedding[node] = self.update_node_state(node, position)
                        # Reconstruct the state by concatenating state_embedding and graph_processor_output
                self.state = torch.cat((self.state_embedding, self.graph_processor_output), dim=1)
                #else:
                    # Penalty if the position is already occupied
                #reward = -0.1
                reward = 0
                self.i += 1
            else:
            #    # Penalty for incorrect node mapping or node already mapped
                reward = 0
        else:
            reward = 0  # Penalty if all nodes have been processed


        if len(self.mapped) == self.num_nodes:
            self.done = True
            self.update_all_transfer_times()
            reward += self.compute_total_reward(re,False)
        else:
            self.done = False

        return self.state, reward, self.done, False, {}

    def steptime(self, action,re):
        # When mapping a node to a position, update grid_mapping
        node, position = action  # Assuming position is a tuple (x, y) for 8x8 grid
        #pdb.set_trace()
        if self.i < len(self.topoorder):
            if node not in self.mapped:
                #if position not in self.node_positions.values():
                    # Map the node to the position
                self.node_positions[node] = position
                self.grid_mapping[node] = position
                    #if node not in self.mapped:
                self.mapped.append(node)
                        # Update the state_embedding for the node
                self.state_embedding[node] = self.update_node_state(node, position)
                        # Reconstruct the state by concatenating state_embedding and graph_processor_output
                self.state = torch.cat((self.state_embedding, self.graph_processor_output), dim=1)
                #else:
                    # Penalty if the position is already occupied
                #reward = -0.1
                reward = 0
                self.i += 1
            else:
            #    # Penalty for incorrect node mapping or node already mapped
                reward = 0
        else:
            reward = 0  # Penalty if all nodes have been processed

        pdb.set_trace()
        if len(self.mapped) == self.num_nodes:
            self.done = True
            self.update_all_transfer_times()
            reward += self.compute_total_reward2(re,True)
        else:
            self.done = False

        return self.state, reward, self.done, False, {}

    def update_node_state(self, node, position):
        position_embedding = torch.zeros(self.feature_dim, dtype=torch.float32).to(self.device)
        position_embedding[position % self.feature_dim] = 1.0

        return position_embedding

    def read_transfer_data_sizes(self, csv_filename):
        transfer_data_sizes = {}
        with open(csv_filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                source = row['Source']
                destination = row['Destination']
                data_size = int(row['Data Transferred (bytes)'])
                transfer_data_sizes[(source, destination)] = data_size
        return transfer_data_sizes

    def update_all_transfer_times(self):
        # Update the total transfer times based on the mapped nodes
        total_time = 0
        transfer_volumes = {}
        #pdb.set_trace()
        for node in self.mapped:
            if node in self.send_table:
                start_position = self.node_positions.get(node)
                for target in self.send_table[node]:
                    end_position = self.node_positions.get(target)
                    if start_position is not None and end_position is not None:
                        # Calculate the path between start and end positions
                        path = self.calculate_path(start_position, end_position)
                        # Get the data size for this transfer from transfer_data_sizes
                        #data_size = self.get_data_size(node, target)
                        data_size = 32
                        transfer_time = self.calculate_routing_time(path[0], path[1],data_size)
                        total_time += transfer_time
        self.all_transfer_times = total_time

    def get_data_size(self, source_node_index, target_node_index):
        # Map node indices to node names
        source_node_name = self.get_node_name(source_node_index)
        target_node_name = self.get_node_name(target_node_index)
        # Retrieve the data size from transfer_data_sizes
        data_size = self.transfer_data_sizes.get((source_node_name, target_node_name), 0)
        return data_size

    def get_node_name(self, node_index):
        # Map node indices to node names
        # You need to implement this method based on your data structures
        node_name = self.node_table.get(node_index, f"Node_{node_index}")
        return node_name

    def calculate_path(self, start_position, end_position):
        path = [min(start_position, end_position), max(start_position, end_position)]
        return path

    def calculate_routing_time(self, start_position, end_position, total_data_bytes):

        x1, y1 = start_position // 8, start_position % 8
        x2, y2 = end_position   // 8, end_position % 8
        total_time = 0

        for x in range(min(x1, x2), max(x1, x2)):
            # Calculate the routing time in cycles for the given data volume
            total_data_bits = total_data_bytes * 8  # Convert bytes to bits
            transfer_time_seconds = total_data_bits / (100e9)  # Transmission speed: 100 Gbps
            transfer_time_ns = transfer_time_seconds * 1e9  # Convert seconds to nanoseconds
            transfer_time_cycles = transfer_time_ns / 1  # Convert nanoseconds to cycles (1 cycle = 1 ns)
            total_time += transfer_time_cycles


        for y in range(min(y1, y2), max(y1, y2)):
            # Calculate the routing time in cycles for the given data volume
            total_data_bits = total_data_bytes * 8  # Convert bytes to bits
            transfer_time_seconds = total_data_bits / (100e9)  # Transmission speed: 100 Gbps
            transfer_time_ns = transfer_time_seconds * 1e9  # Convert seconds to nanoseconds
            transfer_time_cycles = transfer_time_ns / 1  # Convert nanoseconds to cycles (1 cycle = 1 ns)
            total_time += transfer_time_cycles

        return total_time


    def compute_total_reward(self,re,time):
        rewards = []
        penalty = 0

        # Define the node limit per grid position
        num_nodes_limit = self.num_nodes_limit
        # Initialize real node counts per grid position
        self.num_nodes_real = {i: 0 for i in range(64)}

        # Build acc_clu mapping positions to nodes
        self.acc_clu = {}
        for node, position in self.grid_mapping.items():
            if position not in self.acc_clu:
                self.acc_clu[position] = []
            self.acc_clu[position].append(node)
        #pdb.set_trace()
        #_,newmap=self.compute_node_times(self.graph_data['Incomparable Graph'],self.time_map)
        time_p = {position: [] for position in self.acc_clu.keys()}
        for position, nodes in self.acc_clu.items():
            for node in nodes:
                if node in self.time_map:
                    time_p[position].append(self.time_map[node])

        # Collect rewards based on maximum processing time per position
        for position, times in time_p.items():
            if times:
                rewards.append(max(times))

        max_len = 0
        for l in self.node_table.values():
            max_len = max(max_len, len(l))

        # Update real node counts and calculate penalties
        for node, position in self.grid_mapping.items():
            num_of_nodes = len(self.node_table.get(node, []))
            self.num_nodes_real[position] += num_of_nodes
            if self.num_nodes_real[position] > num_nodes_limit[position]:
                penalty += -(self.num_nodes_real[position] - num_nodes_limit[position])


        rewards_normalized = normalize_array(rewards) if rewards else []
        penalty_normalized = normalize_array(penalty) if penalty else []
        #pdb.set_trace()
        #pdb.set_trace()0.1*(penalty / max_len)
        if(rewards):
            if re==True:
                reward = 1/max(rewards_normalized)+0.001*(penalty / max_len)+1/log_scale(self.all_transfer_times)
            #    self.time = max(rewards) + self.all_transfer_times
            else:
                if(time==False):
                    reward = 1 / max(rewards_normalized) + (1 / (log_scale(self.all_transfer_times)))
                else:
                    reward = max(rewards)+self.all_transfer_times
        else:
            reward = (penalty / max_len)+(1 / (log_scale(self.all_transfer_times)))
        return reward

    def compute_total_reward2(self,re,time):
        rewards = []
        penalty = 0

        # Define the node limit per grid position
        num_nodes_limit = self.num_nodes_limit
        # Initialize real node counts per grid position
        self.num_nodes_real = {i: 0 for i in range(64)}

        # Build acc_clu mapping positions to nodes
        self.acc_clu = {}
        for node, position in self.grid_mapping.items():
            if position not in self.acc_clu:
                self.acc_clu[position] = []
            self.acc_clu[position].append(node)
        #pdb.set_trace()
        #_,newmap=self.compute_node_times(self.graph_data['Incomparable Graph'],self.time_map)
        time_p = {position: [] for position in self.acc_clu.keys()}
        for position, nodes in self.acc_clu.items():
            for node in nodes:
                if node in self.time_map:
                    time_p[position].append(self.time_map[node])

        # Collect rewards based on maximum processing time per position
        #pdb.set_trace()
        for position, times in time_p.items():
            if times:
                t=0
                for i in times:
                    t+=i*sigmoid(i)
                rewards.append(int(t))

        max_len = 0
        for l in self.node_table.values():
            max_len = max(max_len, len(l))

        # Update real node counts and calculate penalties
        for node, position in self.grid_mapping.items():
            num_of_nodes = len(self.node_table.get(node, []))
            self.num_nodes_real[position] += num_of_nodes
            if self.num_nodes_real[position] > num_nodes_limit[position]:
                penalty += -(self.num_nodes_real[position] - num_nodes_limit[position])


        rewards_normalized = normalize_array(rewards) if rewards else []
        penalty_normalized = normalize_array(penalty) if penalty else []
        #pdb.set_trace()
        #pdb.set_trace()0.1*(penalty / max_len)
        if(rewards):
            if re==True:
                reward = 1/max(rewards_normalized)+(penalty / max_len)+1/log_scale(self.all_transfer_times)
            #    self.time = max(rewards) + self.all_transfer_times
            else:
                if(time==False):
                    reward = 1 / max(rewards_normalized) + (1 / (log_scale(self.all_transfer_times)))
                else:
                    reward = max(rewards)+self.all_transfer_times
        else:
            reward = (penalty / max_len)+(1 / (log_scale(self.all_transfer_times)))
        return reward


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define any missing functions used in the class
def getKeys(d, value):
    # Return all keys in dictionary `d` that have the given `value`
    return [k for k, v in d.items() if v == value]

def normalize_array(arr):
    # Normalize the array to [1, 2] range
    arr = np.array(arr)
    arr = np.log1p(arr)
    min_val = arr.min()
    max_val = arr.max()
    #pdb.set_trace()
    return (arr - min_val) / (max_val - min_val)+1 if max_val != min_val else [1]

def log_scale(x):
    # Apply logarithmic scaling
    return np.log(x+10)
