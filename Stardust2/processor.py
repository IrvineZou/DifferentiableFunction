from utils import *
import pdb
from concurrent.futures import ProcessPoolExecutor
import os


class DotProcessor:
    def __init__(self, base_dir):
        """
        Initializes the GraphProcessor with the given base directory.
        """
        self.base_dir = base_dir
        self.graph0s = []
        self.graph5s = []
        self.graph0s_pydot = {}
        self.graph5s_pydot = {}
        self.all_graph5_vcore_vtile_nodes = {}
        self.all_successors_map = {}
        self.G5s = {}
        self.graph0s_f = {}

    def read_dot_files(self):
        """
        Reads all .dot files in base_dir and populates graph0s and graph5s lists.
        """
        # Find all .dot files in base_dir
        #pdb.set_trace()
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".dot"):
                    filepath = os.path.join(root, file)
                    if "graph0.dot" in file:
                        self.graph0s.append(filepath)
                    elif "graph5-register-allocation.dot" in file:
                        self.graph5s.append(filepath)


    def process_dot_files_0(self):
        """
        Processes graph0.dot files and loads them into self.graph0s_pydot dictionary.
        """
        for item in self.graph0s:
            key = item.replace('-graph0.dot', '')  # Remove '-graph0.dot' suffix from the filename
            graphs = pydot.graph_from_dot_file(item)
            self.graph0s_pydot[key] = graphs[0]

    def process_dot_files_5(self):
        """
        Processes graph5-register-allocation.dot files and loads them into self.graph5s_pydot dictionary.
        """
        for item in self.graph5s:
            print('Processing', item)
            key = item.replace('-graph5-register-allocation.dot', '')  # Remove suffix from filename
            graphs = pydot.graph_from_dot_file(item)
            self.graph5s_pydot[key] = graphs[0]

    def find_tile(self):
        """
        Processes tile directories, counts cycles from ima_trace files, and outputs results to CSV files.
        """
        results_all = {}
        directories = [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]
        for directory in directories:
            dir_path = os.path.join(self.base_dir, directory)
            results = []
            tile_dirs = [d for d in os.listdir(dir_path) if d.startswith('tile')]
            for tile_dir in tile_dirs:
                tile_num = int(tile_dir.replace('tile', ''))
                tile_path = os.path.join(dir_path, tile_dir)
                for ima_num in range(8):
                    file_path = os.path.join(tile_path, f'ima_trace{ima_num}.txt')
                    if os.path.exists(file_path):
                        cycles = self.count_cycles(file_path)
                        results.append([f'tile{tile_num}_ima_trace{ima_num}', cycles])
            if results:
                results_all[directory] = results
        # Create DataFrames and save to CSV
        for key, items in results_all.items():
            df = pd.DataFrame(items, columns=['Trace', 'Cycle Count'])
            df.to_csv(f'{key}_cycles_count_summary.csv', index=False)

    def clear_graph5(self):
        """
        Processes graph5 pydot graphs, extracts node information, builds successors map,
        and converts graphs to NetworkX format.
        """
        for name, graph5 in self.graph5s_pydot.items():
            print('Processing', name)
            vcore_vtile_nodes = {}
            for node in tqdm(graph5.get_nodes()):
                node_id = node.get_name().strip('"')
                pattern = r'0x[0-9a-fA-F]+'
                matches = re.findall(pattern, node_id)
                if matches:
                    vcore_vtile_nodes[node_id] = matches[0]
            self.all_graph5_vcore_vtile_nodes[name] = vcore_vtile_nodes

        # Convert pydot graphs to NetworkX graphs
        for name, graph5 in self.graph5s_pydot.items():
            self.G5s[name] = nx.nx_pydot.from_pydot(graph5)

        # Build successors map
        for name, G5 in self.G5s.items():
            print('Processing', name)
            successors_map = {}
            for node in tqdm(G5.nodes()):
                pcore_match = re.search(r'pCore = (\d+)', node)
                ptile_match = re.search(r'pTile = (\d+)', node)
                if pcore_match and ptile_match:
                    pcore = int(pcore_match.group(1))
                    ptile = int(ptile_match.group(1))
                    node_key = f'tile{ptile}_ima_trace{pcore}'
                    successors = list(G5.successors(node))
                    successors_map[node_key] = []
                    for successor in successors:
                        pcore_s_match = re.search(r'pCore = (\d+)', successor)
                        ptile_s_match = re.search(r'pTile = (\d+)', successor)
                        if pcore_s_match and ptile_s_match:
                            pcore_s = int(pcore_s_match.group(1))
                            ptile_s = int(ptile_s_match.group(1))
                            successors_map[node_key].append(f'tile{ptile_s}_ima_trace{pcore_s}')
            self.all_successors_map[name] = successors_map

    def clear_graph0(self):
        """
        Processes graph0 pydot graphs, updates node names based on all_graph5_vcore_vtile_nodes.
        """
        hex_pattern = re.compile(r'0x[0-9a-fA-F]+')
        #pdb.set_trace()
        for name, graph0 in self.graph0s_pydot.items():
            print("Processing", name)

            vcore_vtile_nodes = self.all_graph5_vcore_vtile_nodes.get(name, {})
            reverse_map = {str(v): k for k, v in vcore_vtile_nodes.items()}

            node_list = graph0.get_nodes()

            for node in tqdm(node_list, desc=f"{name}"):
                node_name = node.get_name().strip('"')

                # Step 1: extract hex address if present
                match = hex_pattern.search(node_name)
                if match:
                    node_name = match.group(0)

                # Step 2: replace with key from reverse map if available
                new_name = reverse_map.get(node_name, node_name)

                # Only update when needed
                if new_name != node.get_name():
                    node.set_name(new_name)

            self.graph0s_pydot[name] = graph0

            base_name = name.replace("-graph0.dot", "")
            new_graph_path = base_name + "new_graph_with_labels_by_id.dot"
            self.graph0s_f[base_name] = new_graph_path
            graph0.write_raw(new_graph_path)

    @staticmethod
    def count_cycles(file_path):
        """
        Counts cycles from an ima_trace file.
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if 'IMA halted at' in line:
                    cycles = int(line.split()[-2])  # Extract the number of cycles
                    return cycles


    @staticmethod
    def find_keys_by_value(d, val):
        """
        Finds keys in a dictionary given a value.
        """
        return [k for k, v in d.items() if v == val]





