from utils import *
import pdb
from GNN.Charlie_GAT import GATConv

class GraphProcessor(nn.Module):
    def __init__(self, num_node_features, hidden_dim, n_layers, n_heads,
                 use_sage=False, output_dim=64, dropout=0.1):
        """
        Initializes the GraphProcessor model.

        Args:
            num_node_features (int): The number of input node features.
            hidden_dim (int): The dimension of hidden layers.
            n_layers (int): Number of Topoformer layers.
            n_heads (int): Number of attention heads.
            use_sage (bool): Flag to use GraphSAGE or Topoformer encoder.
            output_dim (int): The dimension of the output features.
            dropout (float): Dropout rate.
        """
        super(GraphProcessor, self).__init__()
        self.use_sage = use_sage
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.use_sage:
            # Define a GraphSAGE encoder when use_sage is True
            self.encoder = SAGEConv(num_node_features, hidden_dim)
            self.levelembedding=Linear(2,num_node_features).to(self.device)
            self.norml=nn.BatchNorm1d(2).to(self.device)
            self.norm1=nn.LayerNorm(num_node_features).to(self.device)
            self.drop=FixedDropout(dropout)
        else:
            # Initialize the Topoformer encoder when use_sage is False
            self.encoder = self.Topoformer(num_node_features, hidden_dim, n_layers, n_heads, dropout)

        # Fully connected layer for the final output
        self.fc = nn.Linear(hidden_dim * 7, output_dim)

    def forward(self, all_graphs, graph_data_list,level):
        encoded_graphs = []
        for graph, graph_data in zip(all_graphs.values(), graph_data_list.values()):
            #pdb.set_trace()
            if self.use_sage:
                # Use GraphSAGE encoder
                x = graph_data.x  # Node features
                edge_index = graph_data.edge_index.to(self.device)  # Edge indices
                reverse_edge_index = edge_index[[1, 0], :]
                level=level.to(torch.float)
                #pdb.set_trace()
                levelembeddings=self.norml(level.t())
                levelembeddings=self.levelembedding(levelembeddings)
                #node embedding
                #pdb.set_trace()
                x=self.norm1(x)
                x=self.linear1(x)
                levelembeddings = self.linear2(levelembeddings)
                x=x+levelembeddings
                encoded_nodes1 = self.encoderforward(x.to(self.device), edge_index,None)
                encoded_nodes2 = self.encoderbackward(x.to(self.device),reverse_edge_index,None)
                # Apply GraphSAGE convolution
                #encoded_nodes = self.encoder(x.to(self.device), edge_index)
                #encoded_graph = torch.mean(encoded_nodes, dim=0)
                #encoded_graphs.append(encoded_graph)
            else:
                for graph, graph_data in zip(all_graphs.values(), graph_data_list.values()):
                    node_order = list(nx.topological_sort(graph))
                    masks = self.create_masks(graph, node_order)
                    x = graph_data.x
                    encoded_graph = self.encoder(x.unsqueeze(0), [masks])
                    encoded_graphs.append(encoded_graph.squeeze(0))
                #pdb.set_trace()
                concat_encoded = torch.cat(encoded_graphs, dim=-1).squeeze(1)  # [, d_inner * 7]
                output = self.fc(concat_encoded)  # [, 16]
                return output


    class Topoformer(nn.Module):
        def __init__(self, input_dim, hidden_dim, n_layers, n_heads, dropout=0.1):
            super().__init__()
            self.input_linear = nn.Linear(input_dim, hidden_dim)
            # Stack multiple Topoformer layers
            self.layers = nn.ModuleList(
                [GraphProcessor.TopoformerLayer(hidden_dim, n_heads, dropout) for _ in range(n_layers)]
            )
            self.output_linear = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, x, masks):

            x = self.input_linear(x)
            x = x.permute(1, 0, 2)  # Rearrange dimensions to (seq_len, batch_size, hidden_dim)

            # Pass through each Topoformer layer with its corresponding mask
            for layer, mask in zip(self.layers, masks):
                x = layer(x, mask)
            x = self.output_linear(x)
            x = x.permute(1, 0, 2)  # Rearrange back to (batch_size, seq_len, hidden_dim)
            return x

    class TopoformerLayer(nn.Module):

        def __init__(self, hidden_dim, n_heads, dropout=0.1):

            super().__init__()
            # Multi-head attention layer
            self.mha = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout)
            # Position-wise feed-forward layers
            self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
            # Layer normalization
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)

        def forward(self, x, mask):

            # Apply layer normalization
            x2 = self.norm1(x)
            # Multi-head attention
            x2, _ = self.mha(x2, x2, x2, attn_mask=mask)
            # Residual connection
            x = x + x2
            x2 = self.norm2(x)
            # Feed-forward network
            x2 = self.linear2(self.dropout(F.relu(self.linear1(x2))))
            # Another residual connection
            x = x + x2
            return x

    def create_masks(self, graph, node_order):

        n = len(node_order)
        mask = torch.zeros(n, n)
        # Build mask where mask[i, j] = -inf if node j is not a successor of node i
        # This ensures that attention is only paid to successors
        mask = mask.float().fill_(-100000)
        for i, node in enumerate(node_order):
            successors = list(graph.successors(node))
            for succ in successors:
                if succ in node_order:
                    j = node_order.index(succ)
                    mask[i, j] = 0  # Allow attention to successors
        mask = mask.to(self.device)
        return mask

    def create_supergraph(self, node_table, edges):

        super_graph = nx.DiGraph()
        seen = set()

        for edge in edges:
            if edge[0] not in seen:
                seen.add(edge[0])
                clusters_src = getKey(node_table, edge[0])
                clusters_dst = getKey(node_table, edge[1])
                if clusters_src and clusters_dst:
                    for src in clusters_src:
                        for dst in clusters_dst:
                            super_graph.add_edge(src, dst)
                            # Ensure the supergraph remains acyclic
                            if not nx.is_directed_acyclic_graph(super_graph):
                                super_graph.remove_edge(src, dst)
                                # Optionally, log or handle the cycle creation

        return super_graph

    def create_torch_geometric_data_from_networkx(self, G, node_features):

        # Map nodes to indices
        node_mapping = {node: i for i, node in enumerate(G.nodes())}
        # Convert edges to edge_index format
        edge_index = torch.tensor(
            [[node_mapping[u], node_mapping[v]] for u, v in G.edges()],
            dtype=torch.long
        ).t().contiguous()


        if node_features is not None:
            x = torch.stack([node_features[node] for node in G.nodes()])
        else:
            x = torch.ones((len(G), 1))  # Default to ones if no features provided

        data = Data(x=x, edge_index=edge_index)
        data = data.to(self.device)
        return data

    def calculate_critical_path(self, graph, nodes):

        subgraph = graph.subgraph(nodes)
        lengths = {node: 0 for node in subgraph.nodes}

        # Compute the longest path to each node
        for node in nx.topological_sort(subgraph):
            max_predecessor_length = 0
            for predecessor in subgraph.predecessors(node):
                max_predecessor_length = max(max_predecessor_length, lengths[predecessor])
            lengths[node] = max_predecessor_length + graph.nodes[node].get('time', 1)

        # Return the maximum path length found
        return max(lengths.values())

    def calculate_class_critical_paths(self, graph, node_classes):

        class_critical_times = {}
        #pdb.set_trace()
        for key, cls in node_classes.items():
            if cls:
                # Calculate critical path for each class
                class_critical_times[key] = self.calculate_critical_path(graph, cls)
        return class_critical_times

    def get_clusters(self, node_table, node):

        return node_table.get(node, None)

    def create_all_graphs(self,G):
        if(self.use_sage==False):
            OG = G
            #pdb.set_trace()
            TR = nx.transitive_reduction(G)
            # print('TR Done')
            TC = nx.transitive_closure(G)
            # print('TC Done')
            G_without_TR = G.copy()
            G_without_TR.remove_edges_from(TR.edges())
            # print('G_W_TR Done')
            G_without_TC = G.copy()
            G_without_TC.remove_edges_from(TC.edges())
            # print('G_W_TC Done')

            G_reverse = G.reverse()
            TR_reverse = TR.reverse()
            TC_reverse = TC.reverse()

            G_without_TR_reverse = G_without_TR.reverse()
            # print('G_W_TR Done')
            G_without_TC_reverse = G_without_TC.reverse()
            # print('G_W_TC Done')
            #pdb.set_trace()
            incomparable_graph = nx.DiGraph()
            incomparable_graph.add_nodes_from(G.nodes())
        else:
            incomparable_graph = G
        for u, v in itertools.combinations(G.nodes(), 2):
            if not nx.has_path(G, u, v) and not nx.has_path(G, v, u):
                incomparable_graph.add_edge(u, v)
        #pdb.set_trace()
        #shortests=[]
        critical_paths=[]
        for node in G.nodes:
            #shortests.append(G.nodes[node]['shortest'])
            critical_paths.append(G.nodes[node]['time'])
        #pdb.set_trace()
        #shortests=torch.tensor(shortests)
        critical_paths=torch.tensor(critical_paths)
        levels=torch.stack((critical_paths,critical_paths),dim=0)
        ancestor=torch.zeros(G.number_of_nodes())
        for n in G.nodes():
            ancestor[n]=len(nx.ancestors(G,n))
        if(self.use_sage==False):
            return {
                'Original Graph': G,
                'Transitive Reduction': TR,
                'DAG without Transitive Reduction': G_without_TR,
                'DAG without Transitive Closure': G_without_TC,
                'Transitive Reduction Reverse': TR_reverse,
                'DAG without Transitive Closure Reverse': G_without_TC_reverse,
                'Incomparable Graph': incomparable_graph,
            },levels,ancestor
        else:
            return {
                'Incomparable Graph': incomparable_graph,
            },levels,ancestor