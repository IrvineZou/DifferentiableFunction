from utils import *
import pdb

#run at least 3 random seed number to get averaged result
#another potential method is to use GAT instead of 
# GCN to see whether the assignment is more balanced
#Intuition is that 

class DMoN(nn.Module):
    """
    This class finalizes DMoN with added logic to split oversized clusters.
    """

    def __init__(self, in_features, hidden_features, out_features, n_clusters,
                 activation=F.selu, use_skip_connection=True, collapse_regularization=0.1,
                 dropout_rate=0.0, max_cluster_size=None, oversize_regularization=1.0):
        """
        Initializes the combined GCN and DMoN model with cluster size constraints.

        Args:
            in_features (int): Size of each input sample (number of input features per node).
            hidden_features (int): Number of features in the hidden layer.
            out_features (int): Size of each output sample (number of output features per node).
            n_clusters (int): Number of clusters for DMoN pooling.
            activation (callable, optional): Activation function to use. Defaults to F.selu.
            use_skip_connection (bool, optional): Whether to use skip connections in GCN. Defaults to True.
            collapse_regularization (float, optional): Regularization coefficient for cluster collapse. Defaults to 0.1.
            dropout_rate (float, optional): Dropout rate for DMoN layer. Defaults to 0.0.
            max_cluster_size (int, optional): Maximum allowed cluster size. Defaults to None.
            oversize_regularization (float, optional): Regularization coefficient for oversized clusters. Defaults to 1.0.
        """
        super(DMoN, self).__init__()

        # GCN layers
        #self.embedding = nn.Embedding(256, 256)  # Adjust num_embeddings and embedding_dim as needed
        self.conv1 = GCNConv(in_features, hidden_features).to(device)
        self.conv2 = GCNConv(hidden_features,hidden_features).to(device)
        self.conv3 = GCNConv(hidden_features, out_features).to(device)
        self.activation = activation
        self.use_skip_connection = use_skip_connection
        if use_skip_connection:
            self.skip_proj = nn.Linear(in_features, out_features)

        # DMoN layer
        self.n_clusters = n_clusters
        self.collapse_regularization = collapse_regularization
        self.dropout_rate = dropout_rate
        #??Why is this?
        self.transform = nn.Sequential(
            nn.Linear(out_features, n_clusters),
            nn.Dropout(dropout_rate)
        ).to(device)

        # Cluster size constraint parameters
        self.max_cluster_size = max_cluster_size
        self.oversize_regularization = oversize_regularization
        self.reset()

    def reset(self):
        # Embedding
        #nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        #if self.embedding.padding_idx is not None:
        #    with torch.no_grad():
        #        self.embedding.weight[self.embedding.padding_idx].fill_(0)

        # GCNConv layers (PyG provides reset_parameters)
        if hasattr(self.conv1, "reset_parameters"):
            self.conv1.reset_parameters()
        if hasattr(self.conv2, "reset_parameters"):
            self.conv2.reset_parameters()
        if hasattr(self.conv3, "reset_parameters"):
            self.conv3.reset_parameters()

        # Skip connection projection
        if self.use_skip_connection and hasattr(self, "skip_proj"):
            nn.init.xavier_uniform_(self.skip_proj.weight)
            if self.skip_proj.bias is not None:
                nn.init.zeros_(self.skip_proj.bias)

        # Transform (Linear + Dropout): only Linear has params
        lin = self.transform[0]
        nn.init.xavier_uniform_(lin.weight)
        if lin.bias is not None:
            nn.init.zeros_(lin.bias)

    def aggregate_features_max_pooling(self, feature_table):
        super_node_features = {}
        for i, item in enumerate(feature_table.values()):
            if item != []:
                # Stack the tensors in 'item' and apply adaptive max pooling
                super_node_features[i] =item
        return super_node_features

    def cluster_max_pool(self,x, assignments, n_clusters):
        """
        x: [num_nodes, feat_dim]
        assignments: [num_nodes, n_clusters]   (soft assignment probabilities)
        n_clusters: int

        returns:
            features_pooled: [n_clusters, feat_dim]
            cluster_ids: [num_nodes]
        """
        cluster_ids = torch.argmax(assignments, dim=1)  # [num_nodes]

        num_nodes, feat_dim = x.shape
        features_pooled = torch.full(
            (n_clusters, feat_dim),
            float("-inf"),
            device=x.device,
            dtype=x.dtype
        )

        for c in range(n_clusters):
            mask = (cluster_ids == c)
            if mask.any():
                features_pooled[c] = x[mask].max(dim=0).values
            else:
                # handle empty cluster
                features_pooled[c] = 0.0

        return features_pooled, cluster_ids


    def forward(self, x, edge_index, adjacency):
        """
        Forward pass of the combined GCN and DMoN model with cluster size constraints.

        Args:
            x (Tensor): Input tensor containing node features or indices (shape: [num_nodes]).
            edge_index (LongTensor): Graph connectivity in COO format (shape: [2, num_edges]).
            adjacency (Tensor): Adjacency matrix of the graph (shape: [num_nodes, num_nodes]).

        Returns:
            features_pooled (Tensor): Pooled node features after DMoN layer.
            assignments (Tensor): Soft cluster assignments for each node.
            loss (Tensor): Combined loss from spectral, collapse, and oversize terms.
        """
        # GCN part
        x=x.to(device)
        edge_index=edge_index.to(device)
        adjacency=adjacency.to(device)
        identity = x  # Save the original input for skip connection
        #x = self.embedding(x.long())  # Convert indices to embeddings
        #x = x.view(x.shape[0], -1)  # Flatten embeddings
        identity = identity.view(identity.shape[0], -1)  # Adjust identity shape
        #pdb.set_trace()
        x = self.conv1(x.float(), edge_index)  # First GCN layer
        x = self.activation(x)
        x = self.conv2(x, edge_index)  # Second GCN layer
        x = self.activation(x)
        x = self.conv3(x,edge_index)

        if self.use_skip_connection:
            identity = self.skip_proj(identity.float())
            x = x + identity  # Apply skip connection

        x = self.activation(x)  # Activation after skip connection

        # DMoN pooling part
        #pdb.set_trace()
        features = self.transform(x)  # Transform features for clustering
        assignments = F.softmax(features, dim=1)  # Soft assignments to clusters 
        
        cluster_sizes = assignments.sum(dim=0)  # Sum of assignments per cluster
        assignments_pooling = assignments / cluster_sizes  # Normalize assignments
        #pdb.set_trace()

        # Ensure adjacency is dense
        if adjacency.is_sparse:
            adjacency_dense = adjacency.to_dense()
        else:
            adjacency_dense = adjacency

        degrees = adjacency_dense.sum(dim=0).view(-1, 1)  # Degree of each node
        number_of_nodes = adjacency_dense.size(0)
        number_of_edges = degrees.sum() / 2  # Each edge is counted twice

        # Compute pooled graph
        graph_pooled = torch.mm(assignments.t(), torch.mm(adjacency_dense, assignments))

        # Compute normalizer
        normalizer_left = torch.mm(assignments.t(), degrees)
        normalizer_right = torch.mm(degrees.t(), assignments)
        normalizer = torch.mm(normalizer_left, normalizer_right) / (2.0 * number_of_edges)

        # Spectral loss
        spectral_loss = -torch.trace(graph_pooled - normalizer) / (2.0 * number_of_edges)

        # Collapse loss
        #compare two cluster definition:
        #1. summation of all assignments
        #2. maximum assignment times number of assignments per
        collapse_loss1 = (
            torch.norm(cluster_sizes) / number_of_nodes * torch.sqrt(torch.tensor(self.n_clusters).float()) - 1
        )

        collapse_loss2 = (
            torch.max(cluster_sizes) / number_of_nodes * torch.sqrt( torch.tensor(self.n_clusters).float()) - 1
        )

        # Oversized cluster loss
        if self.max_cluster_size is not None:
            oversized_clusters = cluster_sizes - self.max_cluster_size
            oversized_clusters = torch.clamp(oversized_clusters, min=0)
            oversized_loss = torch.sum(oversized_clusters ** 2)
        else:
            oversized_loss = 0.0

        # Total loss
        loss = (
            spectral_loss
            + self.collapse_regularization * collapse_loss2
            + self.oversize_regularization * oversized_loss
        )

        # Pooled features
        features_pooled, cluster_ids = self.cluster_max_pool(x, assignments, self.n_clusters)
        features_pooled = F.selu(features_pooled)

        return features_pooled, assignments, loss



# ****************************************************************************************
# Directly split a cluster into 2 part if it is too large
# ****************************************************************************************


# class DMoN(nn.Module):
#     """
#     This class finalizes DMoN with logic to forcibly split oversized clusters.
#     """
#
#     def __init__(self, in_features, hidden_features, out_features, n_clusters,
#                  activation=F.selu, use_skip_connection=True, collapse_regularization=0.1,
#                  dropout_rate=0.0, max_cluster_size=None):
#         """
#         Initializes the combined GCN and DMoN model with cluster splitting logic.
#
#         Args:
#             in_features (int): Size of each input sample (number of input features per node).
#             hidden_features (int): Number of features in the hidden layer.
#             out_features (int): Size of each output sample (number of output features per node).
#             n_clusters (int): Initial number of clusters for DMoN pooling.
#             activation (callable, optional): Activation function to use. Defaults to F.selu.
#             use_skip_connection (bool, optional): Whether to use skip connections in GCN. Defaults to True.
#             collapse_regularization (float, optional): Regularization coefficient for cluster collapse. Defaults to 0.1.
#             dropout_rate (float, optional): Dropout rate for DMoN layer. Defaults to 0.0.
#             max_cluster_size (int, optional): Maximum allowed cluster size before splitting. Defaults to None.
#         """
#         super(DMoN, self).__init__()
#
#         # GCN layers
#         self.embedding = nn.Embedding(32, 36)  # Adjust num_embeddings and embedding_dim as needed
#         self.conv1 = GCNConv(in_features, hidden_features)
#         self.conv2 = GCNConv(hidden_features, out_features)
#         self.activation = activation
#         self.use_skip_connection = use_skip_connection
#         if use_skip_connection:
#             self.skip_proj = nn.Linear(in_features, out_features)
#
#         # DMoN layer
#         self.n_clusters = n_clusters
#         self.collapse_regularization = collapse_regularization
#         self.dropout_rate = dropout_rate
#         self.transform = nn.Sequential(
#             nn.Linear(out_features, n_clusters),
#             nn.Dropout(dropout_rate)
#         )
#
#         # Cluster size constraint
#         self.max_cluster_size = max_cluster_size
#
#     def aggregate_features_max_pooling(self, feature_table):
#         super_node_features = {}
#         for i, item in enumerate(feature_table.values()):
#             if item != []:
#                 # Stack the tensors in 'item' and apply adaptive max pooling
#                 super_node_features[i] = F.adaptive_max_pool1d(
#                     torch.stack(item).unsqueeze(0).permute(0, 2, 1),
#                     output_size=1
#                 ).squeeze(2).squeeze(0)
#         return super_node_features
#
#     def forward(self, x, edge_index, adjacency):
#         """
#         Forward pass of the combined GCN and DMoN model with cluster splitting logic.
#
#         Args:
#             x (Tensor): Input tensor containing node features or indices (shape: [num_nodes]).
#             edge_index (LongTensor): Graph connectivity in COO format (shape: [2, num_edges]).
#             adjacency (Tensor): Adjacency matrix of the graph (shape: [num_nodes, num_nodes]).
#
#         Returns:
#             features_pooled (Tensor): Pooled node features after DMoN layer.
#             assignments (Tensor): Soft cluster assignments for each node.
#             loss (Tensor): Combined loss from spectral and collapse terms.
#         """
#         # GCN part
#         identity = x  # Save the original input for skip connection
#         x = self.embedding(x.long())  # Convert indices to embeddings
#         x = x.view(x.shape[0], -1)  # Flatten embeddings
#         identity = identity.view(identity.shape[0], -1)  # Adjust identity shape
#
#         x = self.conv1(x.float(), edge_index)  # First GCN layer
#         x = self.activation(x)
#         x = self.conv2(x, edge_index)  # Second GCN layer
#
#         if self.use_skip_connection:
#             identity = self.skip_proj(identity.float())
#             x = x + identity  # Apply skip connection
#
#         x = self.activation(x)  # Activation after skip connection
#
#         # DMoN pooling part
#         features = self.transform(x)  # Transform features for clustering
#         assignments = F.softmax(features, dim=1)  # Soft assignments to clusters
#
#         # Identify oversized clusters and split them
#         if self.max_cluster_size is not None:
#             with torch.no_grad():
#                 # Convert soft assignments to hard assignments for cluster size calculation
#                 hard_assignments = torch.argmax(assignments, dim=1)
#                 cluster_sizes = torch.bincount(hard_assignments, minlength=self.n_clusters)
#
#                 # List to keep track of new clusters
#                 new_assignments_list = []
#
#                 # Initialize total number of clusters
#                 total_clusters = self.n_clusters
#                 updated_assignments = assignments.clone()
#
#                 for cluster_idx in range(self.n_clusters):
#                     size = cluster_sizes[cluster_idx].item()
#                     if size > self.max_cluster_size:
#                         # Get the nodes assigned to this cluster
#                         nodes_in_cluster = (hard_assignments == cluster_idx).nonzero(as_tuple=True)[0]
#
#                         # Features of nodes in the oversized cluster
#                         x_cluster = x[nodes_in_cluster]
#
#                         # Apply bisecting KMeans to split the cluster into two
#
#
#                         # Perform KMeans with 2 clusters
#                         kmeans = KMeans(n_clusters=2, n_init=10)
#                         x_cluster_np = x_cluster.cpu().numpy()
#                         kmeans.fit(x_cluster_np)
#                         subcluster_labels = kmeans.labels_
#
#                         # Update assignments
#                         for i, node_idx in enumerate(nodes_in_cluster):
#                             if subcluster_labels[i] == 0:
#                                 # Keep the node in the original cluster
#                                 pass  # No action needed
#                             else:
#                                 # Assign the node to a new cluster
#                                 new_cluster_idx = total_clusters
#                                 updated_assignments[node_idx] = 0  # Zero out current assignment
#                                 updated_assignments[node_idx, new_cluster_idx] = 1  # Assign to new cluster
#
#                         # Increment total_clusters
#                         total_clusters += 1
#
#                 # Update the assignments and number of clusters
#                 assignments = updated_assignments
#                 self.n_clusters = total_clusters
#                 # Update transform layer to match new number of clusters if necessary
#                 # Note: This may not be straightforward and might require re-initialization
#         else:
#             cluster_sizes = assignments.sum(dim=0)  # Sum of assignments per cluster
#
#         assignments_pooling = assignments / cluster_sizes  # Normalize assignments
#
#         # Ensure adjacency is dense
#         if adjacency.is_sparse:
#             adjacency_dense = adjacency.to_dense()
#         else:
#             adjacency_dense = adjacency
#
#         degrees = adjacency_dense.sum(dim=0).view(-1, 1)  # Degree of each node
#         number_of_nodes = adjacency_dense.size(0)
#         number_of_edges = degrees.sum() / 2  # Each edge is counted twice
#
#         # Compute pooled graph
#         graph_pooled = torch.mm(assignments.t(), torch.mm(adjacency_dense, assignments))
#
#         # Compute normalizer
#         normalizer_left = torch.mm(assignments.t(), degrees)
#         normalizer_right = torch.mm(degrees.t(), assignments)
#         normalizer = torch.mm(normalizer_left, normalizer_right) / (2.0 * number_of_edges)
#
#         # Spectral loss
#         spectral_loss = -torch.trace(graph_pooled - normalizer) / (2.0 * number_of_edges)
#
#         # Collapse loss
#         collapse_loss = (
#             torch.norm(cluster_sizes.float()) / number_of_nodes * torch.sqrt(torch.tensor(self.n_clusters).float()) - 1
#         )
#
#         # Total loss
#         loss = spectral_loss + self.collapse_regularization * collapse_loss
#
#         # Pooled features
#         features_pooled = torch.mm(assignments_pooling.t(), x)
#         features_pooled = F.selu(features_pooled)
#
#         return features_pooled, assignments, loss
