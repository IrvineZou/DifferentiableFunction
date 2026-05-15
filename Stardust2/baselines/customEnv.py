from utils import *
import pdb
from torch.profiler import profile, record_function, ProfilerActivity

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
        self.reset()

    def reset(self):
        # Reset the environment to the initial state
        graph_names = list(self.all_graphs.keys())
        self.current_graph_name = graph_names[self.current_graph_index]
        self.topoorder = self.topoorder_dict[self.current_graph_name]
        self.node_table = self.node_table_dict[self.current_graph_name]
        self.time_map = self.time_map_dict[self.current_graph_name]
        self.send_table = self.send_table_dict[self.current_graph_name]
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
        #pdb.set_trace()
        # Process the current graph using GraphProcessor
        graph_data = self.graphs_datas[self.current_graph_name]
        # Also, add the graph and node_ids to graph_data
        #graph_data.graph = current_graph
        #graph_data.node_ids = list(current_graph.nodes())

        #profile the topoformer
        #with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        #    with record_function("aggregate"):
        self.graph_processor.to(self.device)
        self.graph_processor_output = self.graph_processor(current_graph,graph_data,self.currentlevel)  # Shape: (num_nodes, feature_dim)
        #a = prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10)
        #pdb.set_trace()
        #aheads = a.find("Self CPU time total")
        #back = a[aheads:].find("s")
        #judges = a[aheads:aheads + back]
        #if (judges[len(judges) - 1] == 'u'):
        #    time1 = float(a[aheads + 21: aheads + back - 1]) / 1000
        #else:
        #    time1 = float(a[aheads + 21: aheads + back - 1])

        # Concatenate the state_embedding and graph_processor_output to form the initial state
        self.state = torch.cat((self.state_embedding, self.graph_processor_output), dim=1)  # Shape: (num_nodes, 128)

        # Update the current graph index for the next reset
        self.current_graph_index = (self.current_graph_index + 1) % len(graph_names)

        return self.state, time1,{}

    def step(self, action,re):
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

                    reward = 0  # No penalty
                #else:
                    # Penalty if the position is already occupied
                #    reward = 0
                    self.i += 1
            else:
            #    # Penalty for incorrect node mapping or node already mapped
                reward = 0
        else:
            reward = 0  # Penalty if all nodes have been processed


        if len(self.mapped) == self.num_nodes:
            self.done = True
            self.update_all_transfer_times()
            reward += self.compute_total_reward(re)
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

        for x in range(min(y1, y2), max(y1, y2)):
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


    def compute_total_reward(self,re):
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
        #pdb.set_trace()
        for l in self.node_table.values():
            max_len = max(max_len, len(l))

        # Update real node counts and calculate penalties
        for node, position in self.grid_mapping.items():
            num_of_nodes = len(self.node_table.get(node, []))
            self.num_nodes_real[position] += num_of_nodes
            if self.num_nodes_real[position] > num_nodes_limit[position]:
                penalty += -(self.num_nodes_real[position] - num_nodes_limit[position])


        rewards_normalized = normalize_array(rewards) if rewards else []

        #pdb.set_trace()
        if(rewards):
            if re==True:
                reward = 1 / max(rewards_normalized) + (penalty / max_len)+(1 / (log_scale(self.all_transfer_times)))
            #    self.time = max(rewards) + self.all_transfer_times
            else:
                reward = 1 / max(rewards_normalized) + (1 / (log_scale(self.all_transfer_times)))
        else:
            reward = (penalty / max_len)+(1 / (log_scale(self.all_transfer_times)))
        return reward



# Define any missing functions used in the class
def getKeys(d, value):
    # Return all keys in dictionary `d` that have the given `value`
    return [k for k, v in d.items() if v == value]

def normalize_array(arr):
    # Normalize the array to [1, 2] range
    arr = np.array(arr)
    min_val = arr.min()
    max_val = arr.max()
    return (arr - min_val) / (max_val - min_val)+1 if max_val != min_val else arr

def log_scale(x):
    # Apply logarithmic scaling
    return np.log(x+50)
