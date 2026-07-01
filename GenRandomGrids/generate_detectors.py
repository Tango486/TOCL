#!/usr/bin/env python3

"""
SUMO检测器自动生成器
根据.net.xml文件自动生成E1和E2检测器配置文件

功能说明：
- E1检测器（点检测器）：放置在每个lane的末端附近（距离终点0.1米）
- E2检测器（区域检测器）：覆盖长度50米，终点位置与E1检测器相同

使用方法：
    python generate_detectors.py <network.net.xml> [output_prefix]

示例：
    python generate_detectors.py grid4x4.net.xml
    # 生成 e1_add.xml 和 e2_add.xml

    python generate_detectors.py grid4x4.net.xml my_
    # 生成 my_e1_add.xml 和 my_e2_add.xml
"""

import xml.etree.ElementTree as ET
import sys
def indent(elem, level=0):
    """兼容 Python 3.8 及以下版本的缩进函数"""
    i = "\n" + level * "    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i



E1_OFFSET_FROM_END = 0.1
E2_DETECTOR_LENGTH = 50
E1_OUTPUT_FILE = 'e1_out.xml'
E2_OUTPUT_FILE = 'e2_out.xml'
DETECTOR_FREQ = 60


def parse_net_file(net_file):
    """解析SUMO网络文件，提取所有非internal的edge和lane信息"""
    tree = ET.parse(net_file)
    root = tree.getroot()

    lanes_info = []


    for edge in root.findall('edge'):
        edge_id = edge.get('id')


        if edge_id.startswith(':'):
            continue


        edge_function = edge.get('function')
        if edge_function == 'internal':
            continue


        for lane in edge.findall('lane'):
            lane_id = lane.get('id')
            lane_length = float(lane.get('length'))

            lanes_info.append({
                'id': lane_id,
                'length': lane_length,
                'edge_id': edge_id
            })

    return lanes_info


def generate_e1_detectors(lanes_info, output_file='e1_add.xml'):
    """
    生成E1检测器配置文件

    E1检测器是点检测器，用于检测特定位置的交通流量。
    每个lane都会在末端附近（距离终点E1_OFFSET_FROM_END米）放置一个E1检测器。

    参数：
        lanes_info: lane信息列表
        output_file: 输出文件路径
    """
    root = ET.Element('additional')
    root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/additional_file.xsd')

    for lane in lanes_info:
        lane_id = lane['id']
        lane_length = lane['length']


        pos = lane_length - E1_OFFSET_FROM_END


        detector = ET.SubElement(root, 'e1Detector')
        detector.set('file', E1_OUTPUT_FILE)
        detector.set('freq', str(DETECTOR_FREQ))
        detector.set('friendlyPos', 'x')
        detector.set('id', lane_id)
        detector.set('lane', lane_id)
        detector.set('pos', str(round(pos, 1)))


    tree = ET.ElementTree(root)
    indent(root)
    tree.write(output_file, encoding='UTF-8', xml_declaration=True)
    print(f"E1检测器配置文件已生成: {output_file}")


    return output_file

def generate_e2_detectors(lanes_info, output_file='e2_add.xml'):
    """
    生成E2检测器配置文件

    E2检测器是区域检测器（laneAreaDetector），用于检测一段区域的交通流量。
    每个lane都会在末端附近放置一个长度为E2_DETECTOR_LENGTH米的E2检测器。
    E2检测器的终点与E1检测器的位置相同。

    参数：
        lanes_info: lane信息列表
        output_file: 输出文件路径
    """
    root = ET.Element('additional')
    root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/additional_file.xsd')

    for lane in lanes_info:
        lane_id = lane['id']
        lane_length = lane['length']



        pos = lane_length - E1_OFFSET_FROM_END - E2_DETECTOR_LENGTH


        detector = ET.SubElement(root, 'laneAreaDetector')
        detector.set('file', E2_OUTPUT_FILE)
        detector.set('freq', str(DETECTOR_FREQ))
        detector.set('friendlyPos', 'x')
        detector.set('id', lane_id)
        detector.set('lane', lane_id)
        detector.set('length', str(E2_DETECTOR_LENGTH))
        detector.set('pos', str(round(pos, 1)))


    tree = ET.ElementTree(root)
    indent(root)
    tree.write(output_file, encoding='UTF-8', xml_declaration=True)
    print(f"E2检测器配置文件已生成: {output_file}")



    return output_file

def gen_det(net_file, prefix):
    try:
        lanes_info = parse_net_file(net_file)
    except Exception as e:
        print(f"错误: 无法解析网络文件 - {e}")
        sys.exit(1)


    print("\n" + "-" * 60)
    e1_output = f"{prefix}e1.add.xml"
    e1_file = generate_e1_detectors(lanes_info, e1_output)


    print("\n" + "-" * 60)
    e2_output = f"{prefix}e2.add.xml"
    e2_file = generate_e2_detectors(lanes_info, e2_output)

    return e1_file, e2_file

if __name__ == '__main__':
    net_file = r"gen_origin\rebuilt2.net.xml"
    prefix = r'gen_origin/'

    gen_det(net_file, prefix)
