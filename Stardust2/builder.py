from utils import *
from constants import operation_to_num
import pdb

class GraphBuilder:
    def __init__(self, G_dict, new_G_dict, graph0s_f, graph5s_pydot):
        """
        Initializes the GraphBuilder with provided dictionaries and graphs.

        Args:
            G_dict (dict): Original graph dictionary.
            new_G_dict (dict): Dictionary to store new graphs.
            graph0s_f (dict): Dictionary of processed graph0 files.
            graph5s_pydot (dict): Dictionary of processed graph5 pydot graphs.
        """
        self.max_len = 256
        self.G_dict = G_dict
        self.new_G_dict = new_G_dict
        self.graph0s_f = graph0s_f
        self.graph5s_pydot = graph5s_pydot
        self.G_dict_backup = {}
        self.all_successors_map = {}
        self.cluster_mapping_dict = {}
        self.item_mapping_dict = {}
        self.node_cache_dict = {}
        self.node_features_dict = {}
        self.node_feature_dict_dict = {}
        self.x_dict = {}
        self.lat_dict_dict = {}
        self.node_mapping_dict = {}
        self.edge_index_dict = {}
        self.successors_map_tm_dict = {}
        self.data_dict = {}
        self.adj_matrix_dict = {}

    def find_successor(self):
        """
        Builds successors mapping from graph5s_pydot graphs.
        """
        # Initialize G_dict, G_dict_backup, and new_G_dict
        for name, new_graph in self.graph0s_f.items():
            name = re.sub(r'^\./', '', name)
            self.G_dict[name] = pydot.graph_from_dot_file(new_graph)[0]
            self.G_dict_backup[name] = self.read_dot(new_graph)  # Assuming read_dot is defined elsewhere
            self.new_G_dict[name] = nx.DiGraph()

        # Build all_successors_map from graph5s_pydot
        for name, G5 in self.graph5s_pydot.items():
            name = re.sub(r'^\./', '', name)
            G5 = from_pydot(G5)
            print('Processing', name)
            successors_map = {}
            seen = set()
            for node in tqdm(G5.nodes()):
                # Extract pCore and pTile from node
                pcore_match = re.search(r'pCore = (\d+)', node)
                ptile_match = re.search(r'pTile = (\d+)', node)
                if pcore_match and ptile_match:
                    pcore = int(pcore_match.group(1))
                    ptile = int(ptile_match.group(1))
                    key = f'tile{ptile}_ima_{pcore}'
                    if key not in seen:
                        seen.add(key)
                        successors_map[key] = []
                    # Iterate over successors
                    for successor in G5.successors(node):
                        pcore_match_s = re.search(r'pCore = (\d+)', successor)
                        ptile_match_s = re.search(r'pTile = (\d+)', successor)
                        if pcore_match_s and ptile_match_s:
                            pcore_s = int(pcore_match_s.group(1))
                            ptile_s = int(ptile_match_s.group(1))
                            successor_key = f'tile{ptile_s}_ima_{pcore_s}'
                            successors_map[key].append(successor_key)
            self.all_successors_map[name] = successors_map
            print('Successors mapping',self.all_successors_map)

    def process(self):
        """
        Processes the graphs to build cluster and item mappings.
        """
        # Build node_name_to_node_dict using utils.build_dict
        #pdb.set_trace()
        node_name_to_node_dict = build_dict(self.G_dict)
        for name, node_name_to_node in node_name_to_node_dict.items():
            cluster_mapping = {}
            item_mapping = {}
            print('Processing Graph:', name)
            for node_name, node in tqdm(node_name_to_node.items()):
                # Extract pCore and pTile from node_name
                pattern = r"pCore = (\d+), pTile = (\d+)"
                match = re.search(pattern, node_name)
                # Get operation number
                operation = node_name.split(':')[0].split('\n')[0].strip('"')
                op_num = str(operation_to_num.get(operation.split('_')[0], 0))
                binary_int_op = int(op_num)
                binary_tensor_op = torch.tensor(binary_int_op, dtype=torch.float32).unsqueeze(0)
                if match:
                    pcore = match.group(1)
                    ptile = match.group(2)
                    cluster_key = (pcore, ptile)
                    cluster_name = f'tile{ptile}_ima_{pcore}'
                    if cluster_key not in cluster_mapping:
                        cluster_mapping[cluster_key] = cluster_name
                        self.new_G_dict[name].add_node(cluster_name)
                    if cluster_name not in item_mapping:
                        item_mapping[cluster_name] = []
                    item_mapping[cluster_name].append(binary_tensor_op)
            self.cluster_mapping_dict[name] = cluster_mapping
            self.item_mapping_dict[name] = item_mapping

    def create_feature(self):
        """
        Creates node features from the item mappings.
        """
        self.node_features_dict = {}
        for name, item_mapping in self.item_mapping_dict.items():
            node_features = {}
            #pdb.set_trace()
            print('Processing Graph:', name)
            for node, items in item_mapping.items():
                concat_item = torch.cat(items)
                node_features[node] = concat_item
            self.node_features_dict[name] = node_features

    def build_cache(self):
        """
        Builds a cache mapping from node hex addresses to (pCore, pTile) tuples.
        """
        hex_pattern = re.compile(r'0x[0-9a-fA-F]+')
        pcore_ptile_pattern = re.compile(r'pCore = (\d+), pTile = (\d+)')
        for name, G in self.G_dict.items():
            node_cache = {}
            for node in G.get_nodes():
                node_name = node.get_name()
                matches = hex_pattern.findall(node_name)
                if matches:
                    match_hex = matches[0]
                    match = pcore_ptile_pattern.search(node_name.strip('"'))
                    if match:
                        node_cache[match_hex] = (match.group(1), match.group(2))
                    else:
                        node_cache[match_hex] = (None, None)
            self.node_cache_dict[name] = node_cache

    def get_pcore_ptile_from_node_name(self, node, name):
        """
        Given a node name, returns the (pCore, pTile) tuple.
        """
        hex_pattern = re.compile(r'0x[0-9a-fA-F]+')
        matches = hex_pattern.findall(node)
        if matches:
            match_hex = matches[0]
            return self.node_cache_dict[name].get(match_hex, (None, None))
        return None, None

    def rebuild_edge(self):
        """
        Rebuilds edges in new_G_dict based on cluster mappings.
        """
        for name, G in self.G_dict.items():
            for edge in G.get_edges():
                src = edge.get_source()
                dst = edge.get_destination()
                pcore1, ptile1 = self.get_pcore_ptile_from_node_name(src, name)
                pcore2, ptile2 = self.get_pcore_ptile_from_node_name(dst, name)
                if all([pcore1, ptile1, pcore2, ptile2]):
                    cluster1 = self.cluster_mapping_dict[name][(pcore1, ptile1)]
                    cluster2 = self.cluster_mapping_dict[name][(pcore2, ptile2)]
                    if cluster1 != cluster2 and not self.new_G_dict[name].has_edge(cluster1, cluster2):
                        self.new_G_dict[name].add_edge(cluster1, cluster2)

    def remove_cycle_from_graph(self):
        """
        Removes cycles from graphs in new_G_dict and updates node features.
        """
        self.create_feature()
        for name, new_G in self.new_G_dict.items():
            new_G_no_cycles = remove_cycles(new_G)  # Assuming remove_cycles is defined elsewhere
            for node in new_G_no_cycles.nodes:
                if node in self.node_features_dict[name]:
                    new_G_no_cycles.nodes[node]['features'] = self.node_features_dict[name][node]
            self.new_G_dict[name] = new_G_no_cycles

    def generate_graph(self):
        """
        Generates data and adjacency matrices for the graphs.
        """
        #pdb.set_trace()
        self.find_successor()
        self.build_cache()
        self.process()
        self.rebuild_edge()
        self.remove_cycle_from_graph()

        #df_names = pd.read_csv('cycles_count_summary.csv')['Graph'].to_list()
        analyze_graphs()
        cycles_df = pd.read_csv('cycles_count_summary.csv')
        #pdb.set_trace()
        for name, new_G in self.new_G_dict.items():
            name = re.sub(r'^\./', '', name)
            # Build node features
            node_feature_dict = {
                node: pad_features(new_G.nodes[node]['features'], self.max_len)
                for node in new_G.nodes
            }

            x = torch.stack(list(node_feature_dict.values()))
            self.x_dict[name] = x
            self.node_feature_dict_dict[name] = node_feature_dict
            #cycles_df = df_names[re.sub(r'^\./', '', name)]
            lat_dict = {}
            print('Processing Graph:')
            #pdb.set_trace()
            for i, item in tqdm(enumerate(cycles_df['Tile_IMA'])):
                match = re.match(r"(.*?)(_.+)", item)
                if match:
                    part1 = match.group(1)
                    part2 = match.group(2)[1:]
                    cleanname = name.split("-")[0]
                    if part1 == cleanname:
                        if part2 in node_feature_dict:
                                lat_dict[part2] = cycles_df['Cycle Count'].iloc[i]
                        self.lat_dict_dict[name] = lat_dict
            # Update node 'time' attribute in new_G_dict
            for node in new_G.nodes:
                print(name)
                print(self.lat_dict_dict.keys())
                if node in self.lat_dict_dict[name]:
                    new_G.nodes[node]['time'] = self.lat_dict_dict[name][node]

            # Build node mapping
            node_mapping = {node: i for i, node in enumerate(new_G.nodes)}
            self.node_mapping_dict[name] = node_mapping

            # Build edge index
            edges = list(new_G.edges)
            edge_index = torch.tensor(
                [[node_mapping[u], node_mapping[v]] for u, v in edges],
                dtype=torch.long
            ).t().contiguous()
            self.edge_index_dict[name] = edge_index
            # Build successors map
            successors_map = self.all_successors_map.get(name, {})
            successors_map_tm = {}
            for key, values in successors_map.items():
                if key != 'tile0_ima_0' and key in node_mapping:
                    node_idx = node_mapping[key]
                    successors_map_tm[node_idx] = []
                    for value in values:
                        if value in node_mapping and node_mapping[value] != node_idx:
                            successor_idx = node_mapping[value]
                            if successor_idx not in successors_map_tm[node_idx]:
                                successors_map_tm[node_idx].append(successor_idx)
            self.successors_map_tm_dict[name] = successors_map_tm
            print(f"Successors map for {name}: {successors_map_tm}")

            # Build data object
            data = Data(x=self.x_dict[name], edge_index=self.edge_index_dict[name])
            self.data_dict[name] = data.to(device)

            # Build adjacency matrix
            adj_matrix = to_dense_adj(data.edge_index)
            adj_matrix = adj_matrix.squeeze(0).to(device)
            self.adj_matrix_dict[name] = adj_matrix


    @staticmethod
    def read_dot(file_path):
        """
        Reads a .dot file and returns the pydot graph.

        Args:
            file_path (str): Path to the .dot file.

        Returns:
            pydot.Graph: The graph read from the .dot file.
        """
        return pydot.graph_from_dot_file(file_path)[0]

#class GraphBuilderFromPt:
#    def __init__(self, graphpath):
