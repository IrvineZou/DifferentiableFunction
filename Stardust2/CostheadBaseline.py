import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import pdb

def topological_order(num_nodes, edges):
    """
    Kahn topological sort.
    edges: list of (u, v), meaning u -> v
    """
    indeg = [0] * num_nodes
    succs = [[] for _ in range(num_nodes)]

    for u, v in edges:
        succs[u].append(v)
        indeg[v] += 1

    q = deque([i for i in range(num_nodes) if indeg[i] == 0])
    order = []

    while q:
        u = q.popleft()
        order.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(order) != num_nodes:
        raise ValueError("The graph must be a DAG.")

    return order


class DifferentiableChipMapper(nn.Module):
    """
    Similar paradigm to the first scheduling version:

        M^i <- []
        for node t in topological order:
            GS_t <- GumbelSoftmax(W_t)
            GSE_t <- ApplyConstraints(GS_t, current loads / capacities)
            M_t^i <- ExtractOneHot(GSE_t)
            append M_t^i
        L <- L(M^i)
        grad <- dL/dW
        W <- update(W)

    Here:
      - each node chooses exactly one chip
      - W_n is the learnable logits over chips for node n
      - capacities are applied incrementally by masking invalid chips
    """

    def __init__(
        self,
        num_nodes,
        num_chips,
        node_sizes,
        chip_capacities,
        edges=None,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_chips = num_chips
        if(edges is not None):
            self.edges = edges.tolist()
        else:
            self.edges=[]

        self.register_buffer(
            "node_sizes",
            torch.as_tensor(node_sizes, dtype=torch.float32)
        )
        self.register_buffer(
            "chip_capacities",
            torch.as_tensor(chip_capacities, dtype=torch.float32)
        )

        if self.node_sizes.numel() != num_nodes:
            raise ValueError("node_sizes must have length num_nodes")
        if self.chip_capacities.numel() != num_chips:
            raise ValueError("chip_capacities must have length num_chips")

        #if transfer_cost is None:
        #    transfer_cost = torch.zeros(num_chips, num_chips, dtype=torch.float32)
        #else:
        #    transfer_cost = torch.as_tensor(transfer_cost, dtype=torch.float32)

        #if transfer_cost.shape != (num_chips, num_chips):
        #    raise ValueError("transfer_cost must have shape [num_chips, num_chips]")

        #self.register_buffer("transfer_cost", transfer_cost)

        if self.edges:
            self.topo = topological_order(num_nodes, self.edges)
        else:
            self.topo = list(range(num_nodes))
        #pdb.set_trace()
        # W in the paper/pseudocode:
        # one learnable logit vector for each node, over all chips
        # shape: [num_nodes, num_chips]
        self.logits = nn.Parameter(torch.zeros(num_nodes, num_chips))

    def _masked_st_gumbel_softmax(self, logits, valid_mask, tau):
        """
        Straight-through Gumbel-Softmax with masking.
        logits: [D]
        valid_mask: [D] bool
        """
        masked_logits = logits.masked_fill(~valid_mask, -1e9)
        y = F.gumbel_softmax(masked_logits, tau=tau, hard=True, dim=-1)
        return y

    def calculate_routing_time(self, start_position, end_position, total_data_bytes=32):

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


    def forward(self, tau=1.0):
        """
        Returns:
          assignments: [N, D] one-hot in forward, soft in backward
          chosen_chips: python list of length N
          chip_loads: [D]
          overflow: [D]
          communication_cost: scalar
          balance_loss: scalar
          infeasible_steps: scalar
          masks: [N, D]
          topo_order: list
        """
        device = self.logits.device
        dtype = self.logits.dtype

        # running loads are used only for incremental masking
        running_loads = torch.zeros(self.num_chips, device=device, dtype=dtype)

        assignment_by_node = [None] * self.num_nodes
        mask_by_node = [None] * self.num_nodes
        chosen_chips = [None] * self.num_nodes

        infeasible_steps = torch.tensor(0.0, device=device)

        for node in self.topo:
            #pdb.set_trace()
            node_size = self.node_sizes[node]

            # ApplyConstraints(...)
            # chip d is valid if current running load + node size <= capacity
            projected_loads = running_loads + node_size
            valid_mask = projected_loads <= self.chip_capacities

            # If no chip is currently feasible, allow all chips so the model
            # can still produce an assignment, and penalize later.
            if not bool(valid_mask.any()):
                valid_mask = torch.ones(
                    self.num_chips, dtype=torch.bool, device=device
                )
                infeasible_steps = infeasible_steps + 1.0

            # GS_t <- GumbelSoftmax(W_t)
            # M_t^i <- ExtractOneHot(...)
            #pdb.set_trace()
            y = self._masked_st_gumbel_softmax(
                logits=self.logits[node],
                valid_mask=valid_mask,
                tau=tau,
            )

            chip_idx = int(y.argmax(dim=-1).item())

            # hard running update, similar in spirit to the first scheduling version
            running_loads[chip_idx] = running_loads[chip_idx] + node_size

            chosen_chips[node] = chip_idx
            assignment_by_node[node] = y
            mask_by_node[node] = valid_mask.float()

        assignments = torch.stack(assignment_by_node, dim=0)  # [N, D]
        masks = torch.stack(mask_by_node, dim=0)              # [N, D]

        # final chip loads from the assignments
        chip_loads = (assignments * self.node_sizes.unsqueeze(1)).sum(dim=0)

        # capacity overflow
        overflow = F.relu(chip_loads - self.chip_capacities)

        # communication cost over edges
        # cost(u, v) = y_u^T * transfer_cost * y_v
        communication_cost = torch.tensor(0.0, device=device, dtype=dtype)
        for u, v in self.edges:
            start=assignments[u].nonzero().view(-1)[0].item()
            end=assignments[v].nonzero().view(-1)[0].item()
            communication_cost = communication_cost + (
                self.calculate_routing_time(start,end)
            )

        # simple balance penalty
        balance_loss = chip_loads.var(unbiased=False)

        return {
            "assignments": assignments,
            "chosen_chips": chosen_chips,
            "chip_loads": chip_loads,
            "overflow": overflow,
            "communication_cost": communication_cost,
            "balance_loss": balance_loss,
            "infeasible_steps": infeasible_steps,
            "masks": masks,
            "topo_order": self.topo,
        }


def chip_mapping_loss(
    out,
    overflow_weight=20.0,
    comm_weight=1.0,
    balance_weight=0.1,
    infeasible_step_weight=2.0,
):
    """
    Overall loss for node-to-chip mapping.
    """
    #pdb.set_trace()
    cap_loss = (out["overflow"] ** 2).sum()
    comm_loss = out["communication_cost"]
    balance_loss = out["balance_loss"]
    infeasible_loss = out["infeasible_steps"]

    loss = (
        overflow_weight * cap_loss
        + comm_weight * comm_loss
        + balance_weight * balance_loss
        + infeasible_step_weight * infeasible_loss
    )
    return loss


def train_chip_mapper(
    model,
    epochs=4000,
    lr=0.1,
    tau_start=3.0,
    tau_end=0.3,
    overflow_weight=20.0,
    comm_weight=1.0,
    balance_weight=0.1,
    infeasible_step_weight=2.0,
    verbose=True,
):
    """
    Same style as the first version:
      for epoch in range(num_epochs):
          out = model(tau)
          loss = chip_mapping_loss(out)
          loss.backward()
          optimizer.step()

    Returns:
      history
      final_out
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(epochs):
        tau = tau_start + (tau_end - tau_start) * epoch / max(epochs - 1, 1)

        optimizer.zero_grad()

        out = model(tau=tau)
        #pdb.set_trace()
        loss = chip_mapping_loss(
            out,
            overflow_weight=overflow_weight,
            comm_weight=comm_weight,
            balance_weight=balance_weight,
            infeasible_step_weight=infeasible_step_weight,
        )

        loss.backward()
        optimizer.step()

        record = {
            "epoch": epoch + 1,
            "tau": tau,
            "loss": float(loss.item()),
            "chosen_chips": list(out["chosen_chips"]),
            "chip_loads": out["chip_loads"].detach().cpu().tolist(),
            "overflow": out["overflow"].detach().cpu().tolist(),
            "communication_cost": float(out["communication_cost"].item()),
        }
        history.append(record)

        if verbose and ((epoch + 1) % 25 == 0 or epoch == 0 or epoch == epochs - 1):
            print(
                f"Epoch {epoch+1:4d} | "
                f"tau={tau:.3f} | "
                f"loss={record['loss']:.4f} | "
                f"mapping={record['chosen_chips']} | "
                f"loads={record['chip_loads']} | "
                f"overflow={record['overflow']} | "
                f"comm={record['communication_cost']:.4f}"
            )

    final_out = model(tau=tau_end)
    return history, final_out


def decode_mapping(assignments):
    """
    assignments: [N, D]
    returns list of chosen chip ids
    """
    return assignments.argmax(dim=-1).tolist()