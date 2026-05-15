from utils import *
import queue
import pdb
from GNN.Charlie_GAT import GATConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import Sequential,GINConv
from DAGformer import dag_nodeformer

class FixedDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x

        if self.mask is None or self.mask.shape != x.shape:
            self.mask = (torch.rand_like(x) > self.p).float() / (1.0 - self.p)

        return x * self.mask





class GraphProcessor(nn.Module):
    def __init__(self, num_node_features, hidden_dim, n_layers, n_heads,encoder,
                 use_sage=False, output_dim=64, dropout=0):
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
        self.layers=n_layers
        self.encoders=encoder
        #pdb.set_trace()
        if self.use_sage:
            # Define a GraphSAGE encoder when use_sage is True
            #nna=torch.nn.Linear(num_node_features,hidden_dim)
            if(self.encoders==3):
            #DAGFormer
                self.encoder=dag_nodeformer(num_node_features,hidden_dim,hidden_dim).to(self.device)
                self.norm1=nn.BatchNorm1d(num_node_features).to(self.device)
                self.drop=FixedDropout(dropout)
            else:
                if(self.encoders==1):
                    self.levelembedding=Linear(2,num_node_features).to(self.device)
                    self.norml=nn.BatchNorm1d(2).to(self.device)
                self.linear1 = nn.Linear(num_node_features, hidden_dim).to(self.device)
                self.linear2 = nn.Linear(num_node_features, hidden_dim).to(self.device)
                self.output_linear = nn.Linear(hidden_dim*n_heads, hidden_dim*n_heads).to(self.device)
                self.encoderforward = Sequential('x, edge_index, edge_attr', [
                    (GATConv(hidden_dim, hidden_dim, heads=n_heads),
                    'x, edge_index, edge_attr -> x'),
                    FixedDropout(dropout),
                ]).to(self.device)

                self.encoderbackward = Sequential('x, edge_index, edge_attr', [
                    (GATConv(hidden_dim, hidden_dim, heads=n_heads),
                    'x, edge_index, edge_attr -> x'),
                    FixedDropout(dropout),
                ]).to(self.device)
            
            self.norm1=nn.LayerNorm(num_node_features).to(self.device)
            self.drop=FixedDropout(dropout)
            #self.norm2=nn.LayerNorm(hidden_dim).to(self.device)
        else:
            # Initialize the Topoformer encoder when use_sage is False
            self.encoder = self.Topoformer(num_node_features, hidden_dim, n_layers, n_heads, dropout)
            self.norm1=nn.BatchNorm1d(num_node_features).to(self.device)
            self.norm2=nn.LayerNorm(hidden_dim).to(self.device)
            self.drop=FixedDropout(dropout)
            self.encoder.to(self.device)
        # Fully connected layer for the final output
        if(self.use_sage==False):
            self.fc = nn.Linear(hidden_dim*7, output_dim)
        else:
            if(self.encoders==2 or self.encoders==1):
                self.fc = nn.Linear(hidden_dim*n_heads, output_dim)
            else:
                self.fc=nn.Linear(hidden_dim,output_dim)
        self.fc.to(self.device)
        self.reset_parameters()

    def reset_parameters(self):
        if(self.use_sage==True):
            if(self.encoders==3):
                self.encoder.reset_parameters()
            else:
                if hasattr(self.encoderforward, 'reset_parameters'):
                    self.encoderforward.reset_parameters()

                if hasattr(self.encoderbackward, 'reset_parameters'):
                    self.encoderbackward.reset_parameters()
                
                if(self.encoders==1):
                    nn.init.xavier_uniform_(self.levelembedding.weight)
                    nn.init.xavier_uniform_(self.linear1.weight)
                    nn.init.xavier_uniform_(self.linear2.weight)
                    nn.init.xavier_uniform_(self.output_linear.weight)
                #if hasattr(self.encoderforwardl,'reset_parameters'):
                #    self.encoderforwardl.reset_parameters()

                #if hasattr(self.encoderbackwardl,'reset_parameters'):
                #    self.encoderbackwardl.reset_parameters()
        else:
            if hasattr(self.encoder,'reset_parameters'):
                self.encoder.reset_parameters()

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        #nn.init.xavier_uniform(self.linear1)

    def stablize(self,level):
        #to make the positional encoding more stablized
        return torch.sigmoid(level)

    def forward(self, all_graphs, graph_data_list,level):
        encoded_graphs = []
        #pdb.set_trace()
        # Iterate over each graph and its corresponding data
        if self.use_sage:
            for graph, graph_data in zip(all_graphs.values(), graph_data_list.values()):
                # Use GraphSAGE encoder
                #pdb.set_trace()
                x = graph_data.x #(level+1).unsqueeze(1)*graph_data.x  # Node features
                x =x.to(self.device)
                edge_index = graph_data.edge_index.to(self.device)  # Edge indices
                if(self.encoders==3):
                    encoded_nodesfinall=self.encoder(x,edge_index)
                    encoded_nodesfinall = self.drop(encoded_nodesfinall)
                    encoded_nodesfinall = torch.relu(encoded_nodesfinall)
                    encoded_graphs.append(encoded_nodesfinall)
                    continue
                level = level.to(self.device)
                if edge_index.numel() == 0:
                    edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)

                # Apply GAT Bidirectional Message Passing convolution
                if(self.encoders==0):
                    reverse_edge_index = edge_index[[1, 0], :]
                    x=self.linear1(x)
                    encoded_nodes1 = self.encoderforward(x.to(self.device), edge_index,level)
                    encoded_nodes2 = self.encoderbackward(x.to(self.device),reverse_edge_index,level)
                
                #Apply GAT Bidirectional Message Passing with two input: level, node-embedding
                else:
                    reverse_edge_index = edge_index[[1, 0], :]
                    #stablize the level information, this is beneficial for making
                    # GNN transferable to unseen graph
                    #level=self.stablize(level)
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
                    #encoded_levels1 = self.encoderforwardl(levelembeddings.to(self.device),edge_index,None)
                    #encoded_levels2 = self.encoderbackwardl(levelembeddings.to(self.device),reverse_edge_index,None)
                    #encoded_nodes1=torch.cat((encoded_nodes1,encoded_levels1),dim=-1)
                    #encoded_nodes2=torch.cat((encoded_nodes2,encoded_levels2),dim=-1)


                encoded_nodesfinall=(encoded_nodes1+encoded_nodes2)/2
                #x=x+encoded_nodesfinall
                #encoded_nodesfinall,_=self.mha(encoded_nodesfinall,encoded_nodesfinall,encoded_nodesfinall)
                encoded_nodesfinall=self.output_linear(encoded_nodesfinall)
                #encoded_nodesfinall=self.norm2(encoded_nodesfinall)
                encoded_nodesfinall=self.drop(encoded_nodesfinall)
                encoded_nodesfinall = torch.relu(encoded_nodesfinall)

                encoded_graphs.append(encoded_nodesfinall)
            #pdb.set_trace()
            concat_encoded = torch.cat(encoded_graphs, dim=-1).squeeze(1)  # [, d_inner * 7]
            output = self.fc(concat_encoded)  # [, 16]
            return output
        else:
            for graph, graph_data in zip(all_graphs.values(), graph_data_list.values()):
                node_order = list(nx.topological_sort(graph))
                masks = self.create_masks(graph, node_order)
                #pdb.set_trace()
                x = graph_data.x
                #self.encoder.to(self.device)
                x = self.norm1(x)
                encoded_graph = self.encoder(x.unsqueeze(0), [masks])
                encoded_graph = encoded_graph.squeeze(0)
                #encoded_graph = self.norm2(encoded_graph)
                encoded_graph = self.drop(encoded_graph)
                encoded_graph = torch.relu(encoded_graph)
                encoded_graphs.append(encoded_graph)
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
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.input_linear.weight)
            nn.init.zeros_(self.input_linear.bias)

            nn.init.xavier_uniform_(self.output_linear.weight)
            nn.init.zeros_(self.output_linear.bias)

            for layer in self.layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()


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
            self.linear1 = nn.Linear(hidden_dim, hidden_dim * 8)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(hidden_dim * 8, hidden_dim)
            # Layer normalization
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)

        def forward(self, x, mask):

            # Apply layer normalization
            #pdb.set_trace()
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
        for node in node_table.keys():
            super_graph.add_node(node)
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

        #pdb.set_trace()
        
        if node_features is not None:
            x = torch.stack([node_features[node] for node in G.nodes()])
        else:
            x = torch.ones((len(G), 1))  # Default to ones if no features provided

        data = Data(x=x, edge_index=edge_index)
        data = data.to(self.device)
        return data

    def calculate_critical_path(self, graph, nodes):
        #pdb.set_trace()
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

    def calculate_shortest_path(self, graph, nodes):
        #pdb.set_trace()
        subgraph = graph.subgraph(nodes)
        lengths = {node: 0 for node in subgraph.nodes}

        # Compute the longest path to each node
        for node in nx.topological_sort(subgraph):
            min_predecessor_length = 1e12
            for predecessor in subgraph.predecessors(node):
                min_predecessor_length = min(min_predecessor_length, lengths[predecessor])
            if(min_predecessor_length==1e12):
                min_predecessor_length=0
            lengths[node] = min_predecessor_length + graph.nodes[node].get('time', 1)

        # Return the maximum path length found
        return min(lengths.values())




    def calculate_class_critical_paths(self, graph, node_classes):

        class_critical_times = {}
        for key, cls in node_classes.items():
            if cls:
                # Calculate critical path for each class
                #pdb.set_trace()
                class_critical_times[key] = self.calculate_critical_path(graph, cls)
            else:
                class_critical_times[key] = 0
        return class_critical_times

    def calculate_class_shortest_paths(self,graph,node_classes):
        class_shortest_times = {}
        for key, cls in node_classes.items():
            if cls:
                # Calculate critical path for each class
                #pdb.set_trace()
                class_shortest_times[key] = self.calculate_shortest_path(graph, cls)
            else:
                class_shortest_times[key] = 0
        return class_shortest_times


    def get_clusters(self, node_table, node):

        return node_table.get(node, None)


    #todo: create hop-extended graph(edge based) with hop indicating how many hops it required for aggregation
    def create_all_graphs(self,G):
        #pdb.set_trace()
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
    
    #modify the original graph with node level information
    #run bfs first to get 
    def bfs(self,G,features):
        q=queue.Queue()
        level=queue.Queue()
        isvisited=torch.zeros(len(features)).to(torch.bool)
        nodes=list(G.nodes)
        #pdb.set_trace()
        tfeaturenode=torch.zeros(len(features))
        indegree=dict(G.in_degree)
        for i in indegree:
            if(indegree[i]==0):
                q.put(nodes[i])
                level.put(0)
                isvisited[i]=True
        #pdb.set_trace()
        adj_dict = {node: dict(neighbors) for node, neighbors in G.adj.items()}
        while(q.empty()==False):
            #print(q.queue)
            tnode=q.get()
            tlevel=level.get()
            tfeaturenode[tnode]=torch.tensor([tlevel])
            for i in adj_dict[tnode]:
                if(isvisited[i]==False):
                    isvisited[i]=True
                    q.put(i)
                    level.put(tlevel+1)
        #pdb.set_trace()
        for i in range(isvisited.size(0)):
            if(isvisited[i]==False):
                tfeaturenode[i]=torch.tensor([0])
        #pdb.set_trace()
        #tfeaturenode = (tfeaturenode - torch.min(tfeaturenode)) / (torch.max(tfeaturenode) - torch.min(tfeaturenode))
        #for i in features:
        #    features[i]=torch.cat((features[i],torch.tensor(tfeaturenode[i].item())))
        #for i in features:
        #    features[i] = torch.cat((features[i], tfeaturenode[i].view(1)), dim=0)
        return tfeaturenode

    def create_all_graphs2(self,G,features):
        #pdb.set_trace()
        features2=self.bfs(G,features)
        return {'Original Graph':G},features2
        
