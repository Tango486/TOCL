#!/usr/bin/env python3
"""Generate route files for parameterized SUMO traffic contexts."""

import xml.etree.ElementTree as ET
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
import matplotlib.pyplot as plt
def save_bar(d, path):
    plt.figure(); plt.bar(d.keys(), d.values()); plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.savefig(path); plt.close()


class TrafficScenarioParams:
    """Traffic demand parameters for generated SUMO contexts."""

    def __init__(self,
                 flow_scale: float = 1.0,
                 turn_concentration: float = None,
                 spatial_imbalance: float = 0.0,
                 temporal_variation: float = 0.0):
        """
        Args:
            flow_scale: 流量缩放因子 (0.3-2.5), 1.0为基准流量
            turn_concentration: 转向浓度参数 (0.5-5.0), 影响转向分布的集中度
            spatial_imbalance: 空间不均衡度 (0.0-0.8)
                              0.0 = 各入口流量完全均匀
                              0.8 = 高度不均衡（某些入口流量很大，某些很小）
            temporal_variation: 时间变化强度 (0.0-0.8)
                              0.0 = 流量无时间变化
                              0.8 = 强烈的时间变化模式
        """
        self.flow_scale = flow_scale
        self.spatial_imbalance = np.clip(spatial_imbalance, 0.0, 0.8)
        self.temporal_variation = np.clip(temporal_variation, 0.0, 0.8)


        if turn_concentration is None:
            self.turn_ratios = {'left': 0.1, 'straight': 0.6, 'right': 0.3}
        else:


            alpha = np.array([turn_concentration, turn_concentration * 5,
                            turn_concentration * 2.5])
            turn_probs = np.random.dirichlet(alpha)
            turn_ratios = {
                'left': turn_probs[0],
                'straight': turn_probs[1],
                'right': turn_probs[2]
            }

            total = sum(turn_ratios.values())
            self.turn_ratios = {k: v/total for k, v in turn_ratios.items()}

    @classmethod
    def sample_random(cls,
                     flow_range: Tuple[float, float] = (0.3, 2.5),
                     turn_concentration: float = 1.0,
                     imbalance_range: Tuple[float, float] = (0.0, 0.8),
                     temporal_range: Tuple[float, float] = (0.0, 0.8)):
        """Sample a generated traffic demand parameter vector."""

        flow_scale = np.random.uniform(flow_range[0], flow_range[1])


        spatial_imbalance = np.random.uniform(imbalance_range[0], imbalance_range[1])


        temporal_variation = np.random.uniform(temporal_range[0], temporal_range[1])

        return cls(flow_scale, turn_concentration, spatial_imbalance, temporal_variation)

    def to_array(self) -> np.ndarray:
        """转换为numpy数组,用于GMM"""
        return np.array([
            self.flow_scale,
            self.turn_ratios['left'],
            self.turn_ratios['straight'],
            self.turn_ratios['right'],
            self.spatial_imbalance,
            self.temporal_variation
        ])

    def to_dict(self) -> dict:
        """转换为字典,用于GMM"""
        return {
            "flow_scale": self.flow_scale,
            "left": self.turn_ratios['left'],
            "straight": self.turn_ratios['straight'],
            "right": self.turn_ratios['right'],
            "spatial_imbalance": self.spatial_imbalance,
            "temporal_variation": self.temporal_variation
        }


    @classmethod
    def from_array(cls, arr: np.ndarray):
        """从numpy数组恢复参数"""
        return cls(
            flow_scale=float(arr[0]),
            turn_ratios={
                'left': float(arr[1]),
                'straight': float(arr[2]),
                'right': float(arr[3])
            },
            spatial_imbalance=float(arr[4]) if len(arr) > 4 else 0.0,
            temporal_variation=float(arr[5]) if len(arr) > 5 else 0.0
        )


def calculate_entry_flow_distribution(entry_edges: List[str],
                                     spatial_imbalance: float) -> Dict[str, float]:
    """
    根据空间不均衡度计算各入口的流量分配权重

    Args:
        entry_edges: 入口边列表
        spatial_imbalance: 不均衡度 (0.0-1.0)
                         0.0 = 完全均匀
                         1.0 = 极度不均衡

    Returns:
        各入口的流量权重字典，和为1.0
    """
    n = len(entry_edges)
    if n == 0:
        return {}

    if spatial_imbalance <= 0.1:

        return {edge: 1.0 / n for edge in entry_edges}







    alpha = np.exp(-5 * spatial_imbalance) * 10 + 0.01


    alphas = np.ones(n) * alpha
    weights = np.random.dirichlet(alphas)

    spatio_ratio = {edge: float(w) for edge, w in zip(entry_edges, weights)}


    return spatio_ratio


def indent_xml(elem, level=0):
    """为旧版本Python提供XML格式化功能"""
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


def parse_network(net_file):
    """解析.net.xml文件,获取边缘edges、连接信息和几何信息"""
    tree = ET.parse(net_file)
    root = tree.getroot()

    edges = {}
    connections = {}
    fringe_nodes = set()
    edge_shapes = {}


    for edge in root.findall('edge'):
        edge_id = edge.get('id')
        if not edge_id.startswith(':'):
            from_node = edge.get('from')
            to_node = edge.get('to')


            lanes = edge.findall('lane')
            if lanes:
                shape = lanes[0].get('shape')
                if shape:

                    coords = [tuple(map(float, p.split(','))) for p in shape.split()]
                    edge_shapes[edge_id] = coords

            edges[edge_id] = {
                'from': from_node,
                'to': to_node,
                'lanes': len(lanes)
            }


    for connection in root.findall('connection'):
        from_edge = connection.get('from')
        to_edge = connection.get('to')
        if from_edge and to_edge:
            if from_edge not in connections:
                connections[from_edge] = []
            connections[from_edge].append(to_edge)


    for junction in root.findall('junction'):
        junction_id = junction.get('id')



        type = junction.get('type')
        if type == 'dead_end':
            fringe_nodes.add(junction_id)
    return edges, connections, fringe_nodes, edge_shapes


def calculate_angle(p1, p2):
    """计算从p1到p2的角度（弧度）"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.atan2(dy, dx)


def get_edge_direction(edge_shape):
    """
    获取edge的方向向量（使用末端方向）
    返回角度（弧度）
    """
    if not edge_shape or len(edge_shape) < 2:
        return 0


    p1 = edge_shape[-2]
    p2 = edge_shape[-1]
    return calculate_angle(p1, p2)


def classify_turn_direction(from_edge_shape, to_edge_shape):
    """
    基于几何角度分类转向方向

    Args:
        from_edge_shape: 来向edge的坐标列表
        to_edge_shape: 去向edge的坐标列表

    Returns:
        'left', 'straight', 或 'right'
    """
    if not from_edge_shape or not to_edge_shape:
        return 'straight'


    incoming_angle = get_edge_direction(from_edge_shape)


    outgoing_angle = get_edge_direction(to_edge_shape)


    relative_angle = outgoing_angle - incoming_angle


    while relative_angle > math.pi:
        relative_angle -= 2 * math.pi
    while relative_angle < -math.pi:
        relative_angle += 2 * math.pi


    angle_deg = math.degrees(relative_angle)





    if 30 < angle_deg <= 150:
        return 'left'
    elif -30 <= angle_deg <= 30:
        return 'straight'
    elif -150 <= angle_deg < -30:
        return 'right'
    else:

        return 'left'


def identify_rim_edges(edges, connections, fringe_nodes):
    """识别边缘edges(入口和出口)"""
    if fringe_nodes:
        entry_edges = []
        exit_edges = []

        for edge_id, edge_info in edges.items():
            from_node = edge_info['from']
            to_node = edge_info['to']

            if from_node in fringe_nodes and to_node not in fringe_nodes:
                entry_edges.append(edge_id)

            if to_node in fringe_nodes and from_node not in fringe_nodes:
                exit_edges.append(edge_id)

        return entry_edges, exit_edges


    node_in_degree = {}
    node_out_degree = {}

    for edge_id, edge_info in edges.items():
        from_node = edge_info['from']
        to_node = edge_info['to']

        node_out_degree[from_node] = node_out_degree.get(from_node, 0) + 1
        node_in_degree[to_node] = node_in_degree.get(to_node, 0) + 1

    all_nodes = set(node_in_degree.keys()) | set(node_out_degree.keys())
    potential_fringe = set()

    for node in all_nodes:
        in_deg = node_in_degree.get(node, 0)
        out_deg = node_out_degree.get(node, 0)
        if in_deg == 0 or out_deg == 0:
            potential_fringe.add(node)

    entry_edges = []
    exit_edges = []

    for edge_id, edge_info in edges.items():
        from_node = edge_info['from']
        to_node = edge_info['to']

        if from_node in potential_fringe and to_node not in potential_fringe:
            entry_edges.append(edge_id)

        if to_node in potential_fringe and from_node not in potential_fringe:
            exit_edges.append(edge_id)

    return entry_edges, exit_edges


def build_path_with_turns(start_edge, target_exit, connections, edge_shapes,
                          turn_ratios, max_depth=20, random_state=None):
    """
    从起始edge构建到目标出口的路径，在每个路口根据转向概率做决策

    Args:
        start_edge: 起始edge
        target_exit: 目标出口edge
        connections: 连接字典
        edge_shapes: edge几何形状字典
        turn_ratios: 转向概率
        max_depth: 最大路径深度
        random_state: 随机状态（用于转向决策）

    Returns:
        完整路径的edge列表，如果失败返回None
    """
    if random_state is None:
        random_state = random.Random()

    path = [start_edge]
    current_edge = start_edge
    visited = {start_edge}

    for _ in range(max_depth):

        if current_edge == target_exit:
            return path


        if current_edge not in connections:
            return None

        next_edges = connections[current_edge]
        if not next_edges:
            return None


        available_edges = [e for e in next_edges if e not in visited]
        if not available_edges:

            available_edges = next_edges


        turn_options = {'left': [], 'straight': [], 'right': []}

        current_shape = edge_shapes.get(current_edge, [])
        for next_edge in available_edges:
            next_shape = edge_shapes.get(next_edge, [])
            direction = classify_turn_direction(current_shape, next_shape)
            turn_options[direction].append(next_edge)



        available_directions = [d for d, edges in turn_options.items() if edges]

        if not available_directions:
            return None


        direction_probs = []
        for direction in available_directions:
            prob = turn_ratios[direction]
            direction_probs.append(prob)


        total_prob = sum(direction_probs)
        direction_probs = [p / total_prob for p in direction_probs]


        selected_direction = random_state.choices(
            available_directions,
            weights=direction_probs
        )[0]


        next_edge = random_state.choice(turn_options[selected_direction])


        path.append(next_edge)
        visited.add(next_edge)
        current_edge = next_edge


    return None


def generate_route_file(net_file: str,
                       output_file: str,
                       params: TrafficScenarioParams,
                       simulation_time: int,
                       base_mean_flow: float,
                       random_seed: Optional[int]):
    """
    Generate a SUMO route file with geometric turn assignment.

    Args:
        net_file: .net.xml文件路径
        output_file: 输出的.rou.xml文件路径
        params: traffic demand parameters
        simulation_time: 仿真时间(秒)
        base_mean_flow: 基准平均流量(车辆/小时/车道)
        random_seed: 随机种子
    """

    print("\n=== Generated traffic routes (geometric turns) ===")
    print(f"Flow Scale: {params.flow_scale:.3f}")
    print(f"Spatial Imbalance: {params.spatial_imbalance:.3f}")
    print(f"Temporal Variation: {params.temporal_variation:.3f}")
    print(f"Turn Ratios: L={params.turn_ratios['left']:.3f}, "
          f"S={params.turn_ratios['straight']:.3f}, "
          f"R={params.turn_ratios['right']:.3f}")


    edges, connections, fringe_nodes, edge_shapes = parse_network(net_file)
    entry_edges, exit_edges = identify_rim_edges(edges, connections, fringe_nodes)

    print(f"Entry edges: {len(entry_edges)}, Exit edges: {len(exit_edges)}")
    print(f"Edge shapes loaded: {len(edge_shapes)}")


    effective_flow = base_mean_flow * params.flow_scale


    entry_flow_weights = calculate_entry_flow_distribution(
        entry_edges, params.spatial_imbalance
    )

    print(f"\n入口流量权重分布:")
    for edge, weight in sorted(entry_flow_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {edge}: {weight:.3f} ({weight*100:.1f}%)")


    root = ET.Element('routes')


    vtype = ET.SubElement(root, 'vType', {
        'id': 'car',
        'accel': '2.6',
        'decel': '4.5',
        'sigma': '0.5',
        'length': '5',
        'minGap': '2.5',
        'maxSpeed': '15',
        'guiShape': 'passenger'
    })

    route_id = 0
    vehicle_id = 0
    all_vehicles = []

    for entry_edge in entry_edges:
        if entry_edge not in connections:
            continue


        num_lanes = edges[entry_edge]['lanes']
        entry_weight = entry_flow_weights[entry_edge]


        entry_flow = effective_flow * num_lanes * entry_weight



        entry_routes = {}

        for exit_edge in exit_edges:

            routes_for_exit = []
            for attempt in range(5):
                path = build_path_with_turns(
                    entry_edge, exit_edge, connections, edge_shapes,
                    params.turn_ratios, max_depth=30,
                    random_state=random.Random(random_seed + attempt if random_seed else None)
                )

                if path:

                    route_elem = ET.SubElement(root, 'route', {
                        'id': f'route_{route_id}',
                        'edges': ' '.join(path)
                    })
                    routes_for_exit.append(f'route_{route_id}')
                    route_id += 1

            if routes_for_exit:
                entry_routes[exit_edge] = routes_for_exit

        if not entry_routes:
            print(f"警告: {entry_edge} 无法生成有效路由")
            continue


        current_time = 0
        lambda_rate = entry_flow / 3600.0
        while current_time < simulation_time:


            time_factor = 1.0 + params.temporal_variation * math.sin(
                2 * math.pi * current_time / simulation_time + math.pi/2
            )

            time_factor = max(0.2, min(2.0, time_factor))


            adjusted_lambda_rate = lambda_rate * time_factor

            mean_interval = 1.0 / adjusted_lambda_rate if adjusted_lambda_rate > 0 else 10.0
            std_interval = mean_interval * 0.3
            interval = max(0.1, np.random.normal(mean_interval, std_interval))
            current_time += interval

            if current_time >= simulation_time:
                break


            available_exits = list(entry_routes.keys())
            if not available_exits:
                continue












            exit_probs = [1.0 / len(available_exits)] * len(available_exits)


            selected_exit = random.choices(available_exits, weights=exit_probs)[0]


            selected_route = random.choice(entry_routes[selected_exit])

            all_vehicles.append({
                'id': f'veh_{vehicle_id}',
                'type': 'car',
                'route': selected_route,
                'depart': current_time,
                'entry_edge': entry_edge,
                'exit_edge': selected_exit
            })
            vehicle_id += 1

    all_vehicles.sort(key=lambda v: v['depart'])


    for vehicle_info in all_vehicles:
        vehicle = ET.SubElement(root, 'vehicle', {
            'id': vehicle_info['id'],
            'type': vehicle_info['type'],
            'route': vehicle_info['route'],
            'depart': f"{vehicle_info['depart']:.2f}"
        })


    tree = ET.ElementTree(root)
    indent_xml(root)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)

    print(f"\n总计生成 {len(all_vehicles)} 辆车")
    print(f"总计生成 {route_id} 条路由")
    print(f"路由文件已保存到: {output_file}")


    print("\n各入口edge的车辆分布:")
    entry_stats = {}
    for v in all_vehicles:
        entry = v['entry_edge']
        entry_stats[entry] = entry_stats.get(entry, 0) + 1

    for entry, count in sorted(entry_stats.items()):
        percentage = (count / len(all_vehicles)) * 100 if all_vehicles else 0
        print(f"  {entry}: {count} 辆 ({percentage:.1f}%)")

    return entry_edges, exit_edges




if __name__ == '__main__':
    net_file = r"D:\PythonProject\GenRandomGrids\test\20251201_154946_x2_y5_seed3\rebuilt2.net.xml"
    output_file = r'D:\PythonProject\GenRandomGrids\test\20251201_154946_x2_y5_seed3\vehicle.rou.xml'


    params = TrafficScenarioParams.sample_random(
        flow_range=(0.8, 1.2),
        turn_concentration=2.0,
        imbalance_range=(0.0, 0.7)
    )









    generate_route_file(
        net_file=net_file,
        output_file=output_file,
        params=params,
        simulation_time=3600,
        base_mean_flow=3000,
        random_seed=42
    )

    print(f"\n参数向量(用于GMM): {params.to_dict()}")
