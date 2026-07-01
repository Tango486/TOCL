from GenRandomGrids.total_gen_random_gridnet import gen_random_net
from GenRandomGrids.generate_detectors import gen_det
from GenRandomGrids.gen_routes import generate_route_file
import os
from datetime import datetime
import xml.etree.ElementTree as ET
import random
import numpy as np
from typing import Dict, List, Tuple, Optional



class TrafficScenarioParams:
    """Traffic demand parameters used by the generated source contexts."""

    def __init__(self,
                 flow_scale: float = 1.0,
                 turn_left_ratio: float = None,
                 spatial_imbalance: float = 0.0,
                 temporal_variation: float = 0.0):
        """Store demand scale, left turn tendency, and variation parameters."""
        self.flow_scale = flow_scale
        self.spatial_imbalance = spatial_imbalance
        self.temporal_variation = temporal_variation

        if turn_left_ratio is None:
            self.turn_ratios = {'left': 0.1, 'straight': 0.6, 'right': 0.3}
        else:
            self.turn_ratios = {
                'left': turn_left_ratio,
                'straight': 0.8-turn_left_ratio,
                'right': 0.2
            }

    @classmethod
    def sample_random(cls,
                     flow_range: Tuple[float, float] = (0.3, 2.5),
                     turn_left_ratio: float = 1.0,
                     imbalance_range: Tuple[float, float] = (0.0, 0.8),
                     temporal_range: Tuple[float, float] = (0.0, 0.8)):
        """Sample a generated traffic demand parameter vector."""
        flow_scale = np.random.uniform(flow_range[0], flow_range[1])
        spatial_imbalance = np.random.uniform(imbalance_range[0], imbalance_range[1])
        temporal_variation = np.random.uniform(temporal_range[0], temporal_range[1])

        return cls(flow_scale, turn_left_ratio, spatial_imbalance, temporal_variation)


def indent_xml(elem, level=0):
       """Format XML for Python versions without ElementTree.indent."""
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
class GEN_RANDOM_GRID():
    def __init__(self, whole_output_path):
        self.whole_output_path = whole_output_path
        self.cur_output_path = None
        self.net_file = None
        self.e1_file = None
        self.e2_file = None
        self.routes_file = None
        self.cfg_path = None

    def gen_random_sumocfg(self, x_nodes, y_nodes, x_len, y_len, attach_len, del_num_nodes, sim_time, base_mean_flow, flow_scale, turn_left_ratio, spatial_imbalance, temporal_variation, random_seed, rollout_id):
        """Generate a complete SUMO configuration for one source context."""
        self.gen_net(x_nodes, y_nodes, x_len, y_len, attach_len, del_num_nodes, random_seed, rollout_id)
        self.save_parameters(x_nodes, y_nodes, x_len, y_len, attach_len, del_num_nodes, sim_time, base_mean_flow, flow_scale, turn_left_ratio, spatial_imbalance, temporal_variation, random_seed)

        self.gen_detectors()

        self.gen_routes(sim_time, base_mean_flow, flow_scale, turn_left_ratio, spatial_imbalance, temporal_variation, random_seed)

        cur_name = os.path.basename(self.cur_output_path.rstrip("/\\")) + ".sumocfg"
        self.cfg_path = os.path.join(self.cur_output_path, cur_name)
        self.gen_cfgxml(cfg_path=self.cfg_path)
        print("Generated SUMO scenario: {}".format(self.cfg_path))
        return self.cfg_path

    def gen_net(self, x_nodes, y_nodes, x_len, y_len, attach_len, del_num_nodes, random_seed=0, rollout_id=None):
        self.cur_output_path = os.path.join(self.whole_output_path, "{}_rollout{}_x{}_y{}_del{}_seed{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"), rollout_id, x_nodes, y_nodes, del_num_nodes, random_seed))
        self.net_file = gen_random_net(
            x_nodes=x_nodes,
            y_nodes=y_nodes,
            x_len=x_len,
            y_len=y_len,
            attach_len=attach_len,
            del_num_nodes=del_num_nodes,
            random_seed=random_seed,
            output_path=self.cur_output_path
        )

    def save_parameters(self, x_nodes, y_nodes, x_len, y_len, attach_len, del_num_nodes, sim_time, base_mean_flow, flow_scale, turn_left_ratio, spatial_imbalance, temporal_variation, random_seed):
        """Save the generated context parameters."""
        params_file = os.path.join(self.cur_output_path, "environment_parameters.txt")
        with open(params_file, 'w', encoding='utf-8') as f:
            f.write("=== Scenario parameters ===\n")
            f.write(f"generate_time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"x_nodes: {x_nodes}\n")
            f.write(f"y_nodes: {y_nodes}\n")
            f.write(f"x_len: {x_len}m\n")
            f.write(f"y_len: {y_len}m\n")
            f.write(f"attach_len: {attach_len}m\n")
            f.write(f"del_num_nodes: {del_num_nodes}\n")
            f.write(f"sim_time: {sim_time}s\n")
            f.write(f"base_mean_flow: {base_mean_flow} veh/hour/lane\n")
            f.write(f"flow_scale: {flow_scale}\n")
            f.write(f"base_mean_flow*flow_scale: {base_mean_flow*flow_scale}\n")
            f.write(f"turn_left_ratio: {turn_left_ratio}\n")
            f.write(f"spatial_imbalance: {spatial_imbalance}\n")
            f.write(f"temporal_variation: {temporal_variation}\n")
            f.write(f"random_seed: {random_seed}\n")
        print(f"Saved scenario parameters to: {params_file}")

    def gen_detectors(self):


        self.e1_file, self.e2_file = gen_det(net_file=self.net_file, prefix=self.cur_output_path+'/')

    def gen_routes(self, sim_time, base_mean_flow, flow_scale, turn_left_ratio, spatial_imbalance, temporal_variation, random_seed=0):




        self.routes_file = os.path.join(self.cur_output_path, "vehicle.rou.xml")

        params = TrafficScenarioParams(
            flow_scale=flow_scale,
            turn_left_ratio=turn_left_ratio,
            spatial_imbalance=spatial_imbalance,
            temporal_variation=temporal_variation
        )

        generate_route_file(
            net_file=self.net_file,
            output_file=self.routes_file,
            params=params,
            simulation_time=sim_time,
            base_mean_flow=base_mean_flow,
            random_seed=random_seed
        )

    def gen_cfgxml(self, cfg_path, begin=0, end=3600, step_length=1.0):
        """
        根据已有的 net / routes / e1 / e2 文件路径，生成 sumo 的 .sumocfg 文件

        Args:
            cfg_path: 生成的 .sumocfg 文件保存路径
            begin: 仿真开始时间（秒）
            end: 仿真结束时间（秒）
            step_length: 仿真步长
        """
        root = ET.Element("configuration")


        input_el = ET.SubElement(root, "input")
        ET.SubElement(input_el, "net-file", value=self.net_file)
        ET.SubElement(input_el, "route-files", value=self.routes_file)


        additional_list = [f for f in [self.e1_file, self.e2_file] if f is not None]
        if additional_list:
            ET.SubElement(input_el, "additional-files",
                          value=",".join(additional_list))


        time_el = ET.SubElement(root, "time")
        ET.SubElement(time_el, "begin", value=str(begin))
        ET.SubElement(time_el, "end", value=str(end))
        ET.SubElement(time_el, "step-length", value=str(step_length))


        indent_xml(root)

        tree = ET.ElementTree(root)
        tree.write(cfg_path, encoding="utf-8", xml_declaration=True)

def sample_parameters(random_seed=0):
    """Sample a standalone generated source context parameter dictionary."""


    x_nodes = random.randint(2, 7)
    y_nodes = random.randint(2, 7)
    x_len = random.randint(150, 600)
    y_len = random.randint(150, 600)
    attach_len = random.randint(100, 400)
    del_factor = random.uniform(0, 0.4)
    del_num_nodes = int(del_factor * x_nodes * y_nodes)


    sim_time = 3600
    base_mean_flow = random.randint(800, 2500)
    flow_scale = random.uniform(0.3, 2.5)


    turn_left_ratio = random.uniform(0.2, 0.8)


    spatial_imbalance = random.uniform(0, 0.8)

    temporal_variation = random.uniform(0, 0.8)


    random_seed = random.randint(1, 50)

    return {
        "x_nodes": x_nodes,
        "y_nodes": y_nodes,
        "x_len": x_len,
        "y_len": y_len,
        "attach_len": attach_len,
        "del_num_nodes": del_num_nodes,
        "sim_time": sim_time,
        "base_mean_flow": base_mean_flow,
        "flow_scale": flow_scale,
        "turn_left_ratio": turn_left_ratio,
        "spatial_imbalance": spatial_imbalance,
        "temporal_variation": temporal_variation,
        "random_seed": random_seed
    }

if __name__ == "__main__":
    gen_random_grid = GEN_RANDOM_GRID(whole_output_path='GenRandomGrids/test')
    params = {'x_nodes': 3, 'y_nodes': 3, 'x_len': 576, 'y_len': 226, 'attach_len': 117, 'del_num_nodes': 3, 'sim_time': 3600, 'base_mean_flow': 1836, 'flow_scale': 1.364147791087517, 'turn_left_ratio': 0.4, 'spatial_imbalance': 0.01, 'temporal_variation': 0.5798538669826538, 'random_seed': 24, 'rollout_id': 0}

    gen_random_grid.gen_random_sumocfg(**params)
