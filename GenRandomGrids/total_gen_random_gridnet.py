import subprocess, os
import shutil
import xml.etree.ElementTree as ET
import random
import math
from typing import Set, Dict, List, Tuple



NETCONVERT = os.environ.get("NETCONVERT", shutil.which("netconvert") or "netconvert")
NETGENERATE = os.environ.get("NETGENERATE", shutil.which("netgenerate") or "netgenerate")



class OrderedSet:
    def __init__(self, iterable=None):
        self._dict = {}
        if iterable:
            for item in iterable:
                self.add(item)

    def add(self, item):
        self._dict[item] = None

    def remove(self, item):
        del self._dict[item]

    def discard(self, item):
        self._dict.pop(item, None)

    def __contains__(self, item):
        return item in self._dict

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return f"OrderedSet({list(self._dict.keys())})"

    def __or__(self, other):
        """支持 | 操作符（并集）"""
        result = OrderedSet(self)
        if isinstance(other, OrderedSet):
            for item in other:
                result.add(item)
        elif isinstance(other, (set, list, tuple)):
            for item in other:
                result.add(item)
        else:
            raise TypeError(f"unsupported operand type(s) for |: 'OrderedSet' and '{type(other).__name__}'")
        return result

    def __sub__(self, other):
        """支持 - 操作符（差集）"""
        result = OrderedSet()
        for item in self:
            if isinstance(other, OrderedSet):
                if item not in other:
                    result.add(item)
            elif isinstance(other, (set, list, tuple)):
                if item not in other:
                    result.add(item)
            else:
                raise TypeError(f"unsupported operand type(s) for -: 'OrderedSet' and '{type(other).__name__}'")
        return result
class SUMOGridModifier:
    def __init__(self, nod_file: str, edg_file: str,
                 x_nodes: int, y_nodes: int,
                 x_len: float, y_len: float,
                 attach_len: float):

        self.nod_file = nod_file
        self.edg_file = edg_file
        self.x_nodes = x_nodes
        self.y_nodes = y_nodes
        self.x_len = x_len
        self.y_len = y_len
        self.attach_len = attach_len


        self.nod_tree = ET.parse(nod_file)
        self.edg_tree = ET.parse(edg_file)
        self.nod_root = self.nod_tree.getroot()
        self.edg_root = self.edg_tree.getroot()


        self.nodes = {}
        self.node_coords = {}
        self.fringe_nodes = OrderedSet()
        self.inner_nodes = OrderedSet()

        self._parse_nodes()

    def _parse_nodes(self):
        for node in self.nod_root.findall('node'):
            node_id = node.get('id')
            x = float(node.get('x'))
            y = float(node.get('y'))
            fringe = node.get('fringe')

            self.nodes[node_id] = node
            self.node_coords[node_id] = (x, y)

            if fringe == 'outer':
                self.fringe_nodes.add(node_id)
            else:
                self.inner_nodes.add(node_id)

    def _build_inner_adjacency_graph(self, excluded_nodes: Set[str] = None) -> Dict[str, Set[str]]:

        if excluded_nodes is None:
            excluded_nodes = OrderedSet()


        remaining_inner = self.inner_nodes - excluded_nodes
        graph = {node_id: OrderedSet() for node_id in remaining_inner}

        for edge in self.edg_root.findall('edge'):
            from_node = edge.get('from')
            to_node = edge.get('to')


            if (from_node in remaining_inner and to_node in remaining_inner):
                graph[from_node].add(to_node)
                graph[to_node].add(from_node)

        return graph

    def _is_inner_connected(self, excluded_nodes: Set[str] = None) -> bool:

        graph = self._build_inner_adjacency_graph(excluded_nodes)

        if not graph:
            return True


        start_node = next(iter(graph.keys()))
        visited = OrderedSet()
        queue = [start_node]

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            for neighbor in graph.get(node, OrderedSet()):
                if neighbor not in visited:
                    queue.append(neighbor)


        return len(visited) == len(graph)

    def _calculate_distance(self, node1_id: str, node2_id: str) -> float:
        x1, y1 = self.node_coords[node1_id]
        x2, y2 = self.node_coords[node2_id]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def remove_random_nodes(self, num_nodes: int = None, strategy: str = 'dispersed', random_seed: int = 0) -> Set[str]:

        num_nodes = min(num_nodes, len(self.inner_nodes))

        if strategy == 'dispersed':
            return self._dispersed_removal(num_nodes, random_seed)
        elif strategy == 'random':
            return self._random_removal(num_nodes)
        elif strategy == 'clustered':
            return self._clustered_removal(num_nodes)
        else:
            raise ValueError(f"未知的策略: {strategy}")

    def _dispersed_removal(self, target_num: int, random_seed: int = 0) -> Set[str]:


        random.seed(random_seed)

        removed_nodes = OrderedSet()


        if target_num == 0:
            return removed_nodes

        available_nodes = list(self.inner_nodes)

        first_node = random.choice(available_nodes)
        removed_nodes.add(first_node)



        for i in range(1, target_num):

            candidates = []
            for node in self.inner_nodes:
                if node in removed_nodes:
                    continue


                test_removal = removed_nodes | {node}
                if not self._is_inner_connected(test_removal):
                    continue


                min_dist = min(self._calculate_distance(node, removed)
                              for removed in removed_nodes)
                candidates.append((node, min_dist))

            if not candidates:
                print(f"无法继续移除更多节点，已移除 {len(removed_nodes)} 个")
                break


            candidates.sort(key=lambda x: x[1], reverse=True)


            top_candidates = candidates[:min(3, len(candidates))]
            selected_node = random.choice(top_candidates)[0]

            removed_nodes.add(selected_node)
            print(f"移除第 {i + 1} 个节点: {selected_node} (距离最近已移除节点 {candidates[0][1]:.2f})")

        print(f"\n总共移除了 {len(removed_nodes)} 个节点: {removed_nodes}")
        return removed_nodes

    def _random_removal(self, target_num: int, max_attempts: int = 1000) -> Set[str]:


        if target_num == 0:
            return set()



        for attempt in range(max_attempts):
            candidate_removal = set(random.sample(list(self.inner_nodes), target_num))

            if self._is_inner_connected(candidate_removal):
                print(f"随机策略：成功找到方案（尝试 {attempt + 1} 次）")
                print(f"将移除以下 {len(candidate_removal)} 个节点: {candidate_removal}")
                return candidate_removal

        print(f"警告：{max_attempts}次随机尝试后未找到方案，切换到渐进式移除...")
        return self._progressive_removal(target_num)

    def _clustered_removal(self, target_num: int) -> Set[str]:


        if target_num == 0:
            return OrderedSet()



        removed_nodes = OrderedSet()
        available_nodes = list(self.inner_nodes)


        first_node = random.choice(available_nodes)
        removed_nodes.add(first_node)
        print(f"移除第 1 个节点: {first_node}")


        for i in range(1, target_num):
            candidates = []
            for node in self.inner_nodes:
                if node in removed_nodes:
                    continue

                test_removal = removed_nodes | {node}
                if not self._is_inner_connected(test_removal):
                    continue


                min_dist = min(self._calculate_distance(node, removed)
                              for removed in removed_nodes)
                candidates.append((node, min_dist))

            if not candidates:
                print(f"无法继续移除更多节点，已移除 {len(removed_nodes)} 个")
                break


            candidates.sort(key=lambda x: x[1])


            top_candidates = candidates[:min(3, len(candidates))]
            selected_node = random.choice(top_candidates)[0]

            removed_nodes.add(selected_node)
            print(f"移除第 {i + 1} 个节点: {selected_node} (距离最近已移除节点 {candidates[0][1]:.2f})")

        print(f"\n总共移除了 {len(removed_nodes)} 个节点: {removed_nodes}")
        return removed_nodes

    def _progressive_removal(self, target_num: int) -> Set[str]:



        removed_nodes = OrderedSet()
        available_nodes = list(self.inner_nodes)
        random.shuffle(available_nodes)

        for node in available_nodes:
            if len(removed_nodes) >= target_num:
                break

            test_removal = removed_nodes | {node}

            if self._is_inner_connected(test_removal):
                removed_nodes.add(node)
                print(f"渐进式移除第 {len(removed_nodes)} 个节点: {node}")

        if len(removed_nodes) < target_num:
            print(f"  只能安全移除 {len(removed_nodes)} 个节点，少于目标的 {target_num} 个")

        return removed_nodes

    def _calculate_new_edge_endpoint(self, removed_node_id: str,
                                     other_node_id: str) -> Tuple[str, float, float]:

        removed_x, removed_y = self.node_coords[removed_node_id]
        other_x, other_y = self.node_coords[other_node_id]


        dx = removed_x - other_x
        dy = removed_y - other_y


        distance = math.sqrt(dx**2 + dy**2)


        if distance > 0:
            dx /= distance
            dy /= distance


        if abs(dx) > abs(dy):

            shorten_dist = self.x_len / 2
        else:

            shorten_dist = self.y_len / 2


        new_x = other_x + dx * shorten_dist
        new_y = other_y + dy * shorten_dist


        new_node_id = f"stub_{removed_node_id}_{other_node_id}"

        return new_node_id, new_x, new_y

    def modify_network(self, nodes_to_remove: Set[str]):
        """
        修改网络：移除节点并调整相关边

        参数:
            nodes_to_remove: 要移除的节点ID集合
        """

        new_nodes = {}


        edges_to_remove = OrderedSet()
        edges_to_modify = {}


        for edge in self.edg_root.findall('edge'):
            edge_id = edge.get('id')
            from_node = edge.get('from')
            to_node = edge.get('to')


            if from_node in nodes_to_remove or to_node in nodes_to_remove:
                if from_node in nodes_to_remove and to_node in nodes_to_remove:

                    edges_to_remove.add(edge_id)
                elif from_node in nodes_to_remove:

                    if to_node in self.fringe_nodes:

                        edges_to_remove.add(edge_id)
                    else:

                        new_node_id, new_x, new_y = self._calculate_new_edge_endpoint(
                            from_node, to_node)
                        new_nodes[new_node_id] = (new_x, new_y, 'priority')
                        edges_to_modify[edge_id] = ('from', new_node_id)
                elif to_node in nodes_to_remove:

                    if from_node in self.fringe_nodes:

                        edges_to_remove.add(edge_id)
                    else:

                        new_node_id, new_x, new_y = self._calculate_new_edge_endpoint(
                            to_node, from_node)
                        new_nodes[new_node_id] = (new_x, new_y, 'priority')
                        edges_to_modify[edge_id] = ('to', new_node_id)






        for node in list(self.nod_root.findall('node')):
            if node.get('id') in nodes_to_remove:
                self.nod_root.remove(node)

        for node_id, (x, y, node_type) in new_nodes.items():
            new_node = ET.SubElement(self.nod_root, 'node')
            new_node.set('id', node_id)
            new_node.set('x', f"{x:.2f}")
            new_node.set('y', f"{y:.2f}")
            new_node.set('type', node_type)


        for edge in list(self.edg_root.findall('edge')):
            edge_id = edge.get('id')

            if edge_id in edges_to_remove:
                self.edg_root.remove(edge)
            elif edge_id in edges_to_modify:
                endpoint, new_node_id = edges_to_modify[edge_id]
                edge.set(endpoint, new_node_id)

    def save(self, output_nod_file: str, output_edg_file: str):
        """
        保存修改后的文件

        参数:
            output_nod_file: 输出节点文件路径
            output_edg_file: 输出边文件路径
        """

        self._indent(self.nod_root)
        self._indent(self.edg_root)

        self.nod_tree.write(output_nod_file, encoding='UTF-8', xml_declaration=True)
        self.edg_tree.write(output_edg_file, encoding='UTF-8', xml_declaration=True)

        print(f"\n3. 完成, 已保存修改后的文件(随机减少node):")
        print(f"  节点文件: {output_nod_file}")
        print(f"  边文件: {output_edg_file}")

    def _indent(self, elem, level=0):
        i = "\n" + level * "    "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "    "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                self._indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i



def gen_grid(x_nodes, y_nodes, x_len, y_len, attach_len, output_path):
    output = os.path.join(output_path, "origin_grid{}x{}.net.xml".format(x_nodes, y_nodes))

    cmd = [
        NETGENERATE,
        "--grid",
        "--grid.x-number", str(x_nodes),
        "--grid.y-number", str(y_nodes),
        "--grid.x-length", str(x_len),
        "--grid.y-length", str(y_len),
        "--default.lanenumber", "3",
        "--grid.attach-length", str(attach_len),
        "--tls.guess",
        "--output-file", output,
    ]


    subprocess.run(cmd, check=True)
    print("1. 路网已生成: {}".format(output))
    return output

def split_net(sumo_netfile, output_path):
    output = os.path.join(output_path, "base_plain")
    subprocess.run([
        NETCONVERT,
        "--sumo-net-file", sumo_netfile,
        "--plain-output-prefix", output,
        "--no-turnarounds"
    ], check=True)

    print("2. {} 网络已转化:\n{}.nod.xml\n{}.edg.xml\n...".format(sumo_netfile, output, output))
    node_file = "{}.nod.xml".format(output)
    edge_file = "{}.edg.xml".format(output)
    return node_file, edge_file




def modifity_net(nod_file, edg_file, x_nodes, y_nodes, x_len, y_len, attach_len, del_num_nodes, output_path, random_seed=0):

    modifier = SUMOGridModifier(
        nod_file=nod_file,
        edg_file=edg_file,
        x_nodes=x_nodes,
        y_nodes=y_nodes,
        x_len=x_len,
        y_len=y_len,
        attach_len=attach_len
    )






    nodes_to_remove = modifier.remove_random_nodes(num_nodes=del_num_nodes, strategy='dispersed', random_seed=random_seed)








    modifier.modify_network(nodes_to_remove)

    output_nod_file = os.path.join(output_path, "modified_plain.nod.xml")
    output_edg_file = os.path.join(output_path, "modified_plain.edg.xml")


    modifier.save(
        output_nod_file=output_nod_file,
        output_edg_file=output_edg_file
    )


    return output_nod_file, output_edg_file

def merge_net(output_nod_file, output_edg_file, output_path):
    output_file = os.path.join(output_path, "modified_network.net.xml")
    subprocess.run([
        NETCONVERT,
        "--node-files", output_nod_file,
        "--edge-files", output_edg_file,


        "--output-file", output_file,
        "--no-turnarounds"
    ], check=True)

    print("将上述nod和edg文件重新合为net文件: {}".format(output_file))

    return output_file

def indent_xml(elem, level=0):
    i = "\n" + level * "    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

NEW_PHASES = """        <phase duration="10" state="GGGGGGrrrsssrrrrrrGGGGGGrrrsssrrrrrr"/>
        <phase duration="3"  state="yyyyyyrrrsssrrrrrryyyyyyrrrsssrrrrrr"/>
        <phase duration="10" state="sssrrrGGGsssrrrrrrsssrrrGGGsssrrrrrr"/>
        <phase duration="3"  state="sssrrrGGGsssrrrrrrsssrrryyysssrrrrrr"/>
        <phase duration="10" state="GGGGGGGGGsssrrrrrrsssrrrrrrsssrrrrrr"/>
        <phase duration="3"  state="yyyyyyyyysssrrrrrrsssrrrrrrsssrrrrrr"/>
        <phase duration="10" state="sssrrrrrrsssrrrrrrGGGGGGGGGsssrrrrrr"/>
        <phase duration="3"  state="sssrrrrrrsssrrrrrryyyyyyyyysssrrrrrr"/>
        <phase duration="10" state="sssrrrrrrGGGGGGrrrsssrrrrrrGGGGGGrrr"/>
        <phase duration="3"  state="sssrrrrrryyyyyyrrrsssrrrrrryyyyyyrrr"/>
        <phase duration="10" state="sssrrrrrrsssrrrGGGsssrrrrrrsssrrrGGG"/>
        <phase duration="3"  state="sssrrrrrrsssrrryyysssrrrrrrsssrrrGGG"/>
        <phase duration="10" state="sssrrrrrrsssrrrrrrsssrrrrrrGGGGGGGGG"/>
        <phase duration="3"  state="sssrrrrrrsssrrrrrrsssrrrrrryyyyyyyyy"/>
        <phase duration="10" state="sssrrrrrrGGGGGGGGGsssrrrrrrsssrrrrrr"/>
        <phase duration="3"  state="sssrrrrrryyyyyyyyysssrrrrrrsssrrrrrr"/>"""

def find_traffic_light_nodes(nod_file):
    """从.nod.xml文件中找出所有交通灯节点"""
    tree = ET.parse(nod_file)
    root = tree.getroot()
    tl_nodes = []

    for node in root.findall('node'):
        if node.get('type') == 'traffic_light':
            tl_nodes.append(node.get('id'))

    print(f"找到 {len(tl_nodes)} 个交通灯节点: {tl_nodes}")
    return tl_nodes



























def analyze_edge_connections(edg_file, nod_file, tl_nodes):
    """分析每个交通灯节点的进出车道，并按上下左右排序"""

    nod_tree = ET.parse(nod_file)
    nod_root = nod_tree.getroot()

    node_positions = {}
    for node in nod_root.findall('node'):
        node_id = node.get('id')
        x = float(node.get('x'))
        y = float(node.get('y'))
        node_positions[node_id] = {'x': x, 'y': y}


    edg_tree = ET.parse(edg_file)
    edg_root = edg_tree.getroot()

    edges_info = {}
    for edge in edg_root.findall('edge'):
        edge_id = edge.get('id')
        from_node = edge.get('from')
        to_node = edge.get('to')
        edges_info[edge_id] = {'from': from_node, 'to': to_node}


    node_connections = {}

    for tl_node in tl_nodes:
        if tl_node not in node_positions:
            print(f"⚠️ 警告: 交通灯节点 {tl_node} 未找到坐标信息")
            continue

        tl_x = node_positions[tl_node]['x']
        tl_y = node_positions[tl_node]['y']

        incoming_edges = []
        outgoing_edges = []


        for edge_id, info in edges_info.items():
            if info['to'] == tl_node:

                from_node = info['from']
                if from_node in node_positions:
                    from_x = node_positions[from_node]['x']
                    from_y = node_positions[from_node]['y']

                    dx = tl_x - from_x
                    dy = tl_y - from_y

                    incoming_edges.append({
                        'edge_id': edge_id,
                        'dx': dx,
                        'dy': dy,
                        'from_node': from_node
                    })

            if info['from'] == tl_node:

                to_node = info['to']
                if to_node in node_positions:
                    to_x = node_positions[to_node]['x']
                    to_y = node_positions[to_node]['y']

                    dx = to_x - tl_x
                    dy = to_y - tl_y

                    outgoing_edges.append({
                        'edge_id': edge_id,
                        'dx': dx,
                        'dy': dy,
                        'to_node': to_node
                    })



        incoming_sorted = {'right': None, 'down': None, 'up': None, 'left': None}

        for edge in incoming_edges:
            dx, dy = edge['dx'], edge['dy']
            if abs(dx) > abs(dy):
                if dx > 0:
                    incoming_sorted['right'] = edge['edge_id']
                else:
                    incoming_sorted['left'] = edge['edge_id']
            else:
                if dy > 0:
                    incoming_sorted['up'] = edge['edge_id']
                else:
                    incoming_sorted['down'] = edge['edge_id']


        outgoing_sorted = {'up': None, 'left': None, 'right': None, 'down': None}

        for edge in outgoing_edges:
            dx, dy = edge['dx'], edge['dy']
            if abs(dx) > abs(dy):
                if dx > 0:
                    outgoing_sorted['right'] = edge['edge_id']
                else:
                    outgoing_sorted['left'] = edge['edge_id']
            else:
                if dy > 0:
                    outgoing_sorted['up'] = edge['edge_id']
                else:
                    outgoing_sorted['down'] = edge['edge_id']


        node_connections[tl_node] = {







            'incoming': [
                incoming_sorted['left'],
                incoming_sorted['up'],
                incoming_sorted['down'],
                incoming_sorted['right']
            ],

            'outgoing': [
                outgoing_sorted['up'],
                outgoing_sorted['left'],
                outgoing_sorted['right'],
                outgoing_sorted['down']
            ],
            'incoming_dict': incoming_sorted,
            'outgoing_dict': outgoing_sorted
        }








    return node_connections

def generate_36_connections(node_id, incoming_edges, outgoing_edges):

    connections = []



    if len(incoming_edges) >= 1 and len(outgoing_edges) >= 3:
        right_in = incoming_edges[0] if len(incoming_edges) > 0 else "右进车道edge"
        down_in = incoming_edges[1] if len(incoming_edges) > 1 else "下进车道edge"
        up_in = incoming_edges[2] if len(incoming_edges) > 2 else "上进车道edge"
        left_in = incoming_edges[3] if len(incoming_edges) > 3 else "左进车道edge"

        up_out = outgoing_edges[0] if len(outgoing_edges) > 0 else "上出车道edge"
        left_out = outgoing_edges[1] if len(outgoing_edges) > 1 else "左出车道edge"
        right_out = outgoing_edges[2] if len(outgoing_edges) > 2 else "右出车道edge"
        down_out = outgoing_edges[3] if len(outgoing_edges) > 3 else "下出车道edge"
    else:

        right_in = incoming_edges[0] if incoming_edges else "右进车道edge"
        down_in = incoming_edges[1] if len(incoming_edges) > 1 else "下进车道edge"
        up_in = incoming_edges[2] if len(incoming_edges) > 2 else "上进车道edge"
        left_in = incoming_edges[3] if len(incoming_edges) > 3 else "左进车道edge"

        up_out = outgoing_edges[0] if outgoing_edges else "上出车道edge"
        left_out = outgoing_edges[1] if len(outgoing_edges) > 1 else "左出车道edge"
        right_out = outgoing_edges[2] if len(outgoing_edges) > 2 else "右出车道edge"
        down_out = outgoing_edges[3] if len(outgoing_edges) > 3 else "下出车道edge"


    directions = [
        (right_in, [(0, up_out, 'r'), (1, left_out, 's'), (2, down_out, 'l')]),
        (down_in, [(0, right_out, 'r'), (1, up_out, 's'), (2, left_out, 'l')]),
        (up_in, [(0, left_out, 'r'), (1, down_out, 's'), (2, right_out, 'l')]),
        (left_in, [(0, down_out, 'r'), (1, right_out, 's'), (2, up_out, 'l')])
    ]

    for from_edge, turns in directions:
        for from_lane, to_edge, direction in turns:
            for to_lane in range(3):
                conn = f'    <connection from="{from_edge}" to="{to_edge}" fromLane="{from_lane}" toLane="{to_lane}" tl="{node_id}" dir="{direction}"/>'
                connections.append(conn)

    return connections

def modify_net_xml(net_file, output_file, tl_nodes, node_connections):
    """修改.net.xml文件"""
    tree = ET.parse(net_file)
    root = tree.getroot()


    for tlLogic in root.findall('tlLogic'):
        tl_id = tlLogic.get('id')
        if tl_id in tl_nodes:
            print(f"正在修改交通灯 {tl_id} 的相位配置...")

            for phase in tlLogic.findall('phase'):
                tlLogic.remove(phase)


            phase_lines = NEW_PHASES.strip().split('\n')
            for line in phase_lines:
                line = line.strip()
                if line.startswith('<phase'):

                    import re
                    duration = re.search(r'duration="(\d+)"', line).group(1)
                    state = re.search(r'state="([^"]+)"', line).group(1)

                    phase_elem = ET.SubElement(tlLogic, 'phase')
                    phase_elem.set('duration', duration)
                    phase_elem.set('state', state)


    connections_to_remove = []
    for conn in root.findall('connection'):
        if conn.get('tl') in tl_nodes:
            connections_to_remove.append(conn)

    for conn in connections_to_remove:
        root.remove(conn)




    for tl_node in tl_nodes:
        incoming = node_connections[tl_node]['incoming']
        outgoing = node_connections[tl_node]['outgoing']





        new_conns = generate_36_connections(tl_node, incoming, outgoing)



        for conn_str in new_conns:

            import re
            from_edge = re.search(r'from="([^"]+)"', conn_str).group(1)
            to_edge = re.search(r'to="([^"]+)"', conn_str).group(1)
            from_lane = re.search(r'fromLane="([^"]+)"', conn_str).group(1)
            to_lane = re.search(r'toLane="([^"]+)"', conn_str).group(1)
            tl = re.search(r'tl="([^"]+)"', conn_str).group(1)
            direction = re.search(r'dir="([^"]+)"', conn_str).group(1)

            conn_elem = ET.SubElement(root, 'connection')
            conn_elem.set('from', from_edge)
            conn_elem.set('to', to_edge)
            conn_elem.set('fromLane', from_lane)
            conn_elem.set('toLane', to_lane)

            conn_elem.set('tl', tl)
            conn_elem.set('linkIndex', "9")
            conn_elem.set('dir', direction)

            conn_elem.tail = "\n    "




    indent_xml(root)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)

def rebuild_network(input_file, output_file):
    """使用netconvert重建网络"""

    cmd = [
        NETCONVERT,
        "--sumo-net-file", input_file,

        "--tls.guess",

        "-o", output_file,
        "--no-turnarounds"
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"\n修改相位后的net文件已保存到: {output_file}")

    except subprocess.CalledProcessError as e:
        print(f"❌ 重建失败: {e}")
        print(f"错误输出: {e.stderr}")

def redit(input_path, output_path):

    tree = ET.parse(input_path)
    root = tree.getroot()


    modified_count = 0


    for conn in root.findall(".//connection"):
        tl = conn.get("tl")
        via = conn.get("via")


        if tl and via:

            numbers = list(map(int, via.split('_')[-2:]))
            link_index_sum = sum(numbers)


            conn.set("linkIndex", str(link_index_sum))
            modified_count += 1




    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"修改linkIndex后的文件: {output_path}")
    return output_path



def change_phase(NOD_FILE, EDG_FILE, NET_FILE, OUTPUT_PATH):
    OUTPUT_FILE = os.path.join(OUTPUT_PATH, "modified_network_updated.net.xml")
    REBUILT_FILE = os.path.join(OUTPUT_PATH, "rebuilt.net.xml")


    tl_nodes = find_traffic_light_nodes(NOD_FILE)

    if not tl_nodes:
        print("⚠️ 未找到交通灯节点，程序退出")
        return


    node_connections = analyze_edge_connections(EDG_FILE, NOD_FILE, tl_nodes)


    modify_net_xml(NET_FILE, OUTPUT_FILE, tl_nodes, node_connections)


    rebuild_network(OUTPUT_FILE, REBUILT_FILE)


    output_file = os.path.join(OUTPUT_PATH, "rebuilt2.net.xml")
    final_net = redit(input_path=REBUILT_FILE, output_path=output_file)

    return final_net

def gen_random_net(x_nodes, y_nodes, x_len, y_len, attach_len, del_num_nodes, output_path, random_seed=0):
    os.makedirs(output_path, exist_ok=True)


    origin_gridnetwork = gen_grid(x_nodes, y_nodes, x_len, y_len, attach_len, output_path)

    node_file, edg_file = split_net(origin_gridnetwork, output_path)


    output_nod_file, output_edg_file = modifity_net(node_file, edg_file, x_nodes, y_nodes, x_len, y_len, attach_len, del_num_nodes, output_path, random_seed)


    output_net_file = merge_net(output_nod_file, output_edg_file, output_path)


    final_net = change_phase(output_nod_file, output_edg_file, output_net_file, output_path)
    return final_net

def main():
    x_nodes = 2
    y_nodes = 2
    x_len = 300
    y_len = 240
    attach_len = 200
    del_num_nodes = 1
    output_path = "GenRandomGrids/temp"


    os.makedirs(output_path, exist_ok=True)


    origin_gridnetwork = gen_grid(x_nodes, y_nodes, x_len, y_len, attach_len, output_path)


    node_file, edg_file = split_net(origin_gridnetwork, output_path)


    output_nod_file, output_edg_file = modifity_net(node_file, edg_file, x_nodes, y_nodes, x_len, y_len, attach_len, del_num_nodes, output_path)


    output_net_file = merge_net(output_nod_file, output_edg_file, output_path)


    change_phase(output_nod_file, output_edg_file, output_net_file, output_path)


if __name__ == "__main__":
    main()
