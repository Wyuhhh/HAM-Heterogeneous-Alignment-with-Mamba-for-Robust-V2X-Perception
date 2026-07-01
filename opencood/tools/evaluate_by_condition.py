"""
按场景条件（光照/环境/天气）评估模型性能
复现论文 Table 4 格式的结果
"""

import argparse
import os
import sys
import yaml
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import resource
torch.multiprocessing.set_sharing_strategy('file_system')
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))
from torch.utils.data import DataLoader
from tqdm import tqdm

root_path = Path(__file__).resolve().parents[2]
sys.path.append(str(root_path))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import inference_utils, train_utils
from opencood.utils import eval_utils_airv2x as eval_utils


def classify_scenario(sun, cloud, precip, fog, town):
    """分类场景"""
    # 光照
    if sun > 45:
        lighting = "Day"
    elif sun > 0:
        lighting = "Dusk"
    else:
        lighting = "Night"
    
    # 环境
    town_num = int(town.replace("Town", "").replace("town", ""))
    environment = "Urban" if town_num <= 5 else "Rural"
    
    # 天气
    if precip > 20:
        weather = "Rainy"
    elif fog > 20:
        weather = "Foggy"
    elif cloud > 50:
        weather = "Cloudy"
    else:
        weather = "Clear"
    
    return lighting, environment, weather


def parse_scenario_config(scenario_path):
    """解析场景配置"""
    # 处理符号链接
    real_path = os.path.realpath(scenario_path)
    config_path = os.path.join(real_path, "scenario_config.yaml")
    
    if not os.path.exists(config_path):
        return None
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    weather = config.get('world', {}).get('weather', {})
    town = config.get('world', {}).get('town', 'Town01')
    
    sun = weather.get('sun_altitude_angle', 45)
    precip = weather.get('precipitation', 0)
    fog = weather.get('fog_density', 0)
    cloud = weather.get('cloudiness', 0)
    
    lighting, environment, weather_cond = classify_scenario(sun, cloud, precip, fog, town)
    
    return {
        'lighting': lighting,
        'environment': environment,
        'weather': weather_cond,
        'town': town,
        'sun': sun,
        'precip': precip,
        'fog': fog,
        'cloud': cloud
    }


def get_scenario_infos(data_root):
    """获取所有场景信息"""
    scenarios = {}
    print("\n" + "="*80)
    print("场景分类:")
    print("="*80)
    print(f"{'场景名':<45} | {'帧数':>6} | {'Town':<6} | {'光照':<5} | {'环境':<5} | {'天气':<6}")
    print("-"*80)
    
    for name in sorted(os.listdir(data_root)):
        path = os.path.join(data_root, name)
        if os.path.isdir(path):
            info = parse_scenario_config(path)
            if info:
                real_path = os.path.realpath(path)
                frames = len([d for d in os.listdir(real_path) if d.startswith('timestamp_')])
                scenarios[name] = info
                scenarios[name]['frames'] = frames
                print(f"{name:<45} | {frames:>6} | {info['town']:<6} | {info['lighting']:<5} | {info['environment']:<5} | {info['weather']:<6}")
    
    return scenarios


def init_result_stat():
    return {"tp": [], "fp": [], "gt": 0, "score": []}


def calculate_ap(result_stat):
    """计算AP"""
    aps = {}
    for iou in [0.3, 0.5, 0.7]:
        if iou not in result_stat or result_stat[iou]["gt"] == 0:
            aps[iou] = 0.0
            continue
        
        tp = np.array(result_stat[iou]["tp"])
        fp = np.array(result_stat[iou]["fp"])
        score = np.array(result_stat[iou]["score"])
        gt = result_stat[iou]["gt"]
        
        if len(tp) == 0:
            aps[iou] = 0.0
            continue
        
        idx = np.argsort(-score)
        tp = tp[idx]
        fp = fp[idx]
        
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        
        prec = tp_cum / (tp_cum + fp_cum + 1e-10)
        rec = tp_cum / (gt + 1e-10)
        
        # VOC-style AP
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            p = prec[rec >= t]
            ap += np.max(p) if len(p) > 0 else 0
        aps[iou] = ap / 11 * 100
    
    return aps


def merge_stats(stats_list):
    """合并统计"""
    merged = {iou: init_result_stat() for iou in [0.3, 0.5, 0.7]}
    for stats in stats_list:
        for iou in [0.3, 0.5, 0.7]:
            if iou in stats:
                merged[iou]["tp"].extend(stats[iou]["tp"])
                merged[iou]["fp"].extend(stats[iou]["fp"])
                merged[iou]["score"].extend(stats[iou]["score"])
                merged[iou]["gt"] += stats[iou]["gt"]
    return merged


def print_table4(results, epoch):
    """打印Table 4格式"""
    print("\n" + "="*100)
    print(f"Table 4: 3D Object Detection Results (Epoch {epoch})")
    print("="*100)
    
    # Header
    print(f"\n{'Method':<12}|", end="")
    for cond in ["Day", "Dusk", "Night", "Urban", "Rural", "Rainy", "Foggy", "Cloudy", "Clear"]:
        print(f" {cond:^11}|", end="")
    print()
    
    print(f"{'':12}|", end="")
    for _ in range(9):
        print(f" {'AP30':>5} {'AP50':>5}|", end="")
    print()
    
    print("-"*120)
    
    # Values
    lighting = results.get('lighting', {})
    env = results.get('environment', {})
    weather = results.get('weather', {})
    
    def get_ap(d, key):
        return d.get(key, {}).get(0.3, 0), d.get(key, {}).get(0.5, 0)
    
    day30, day50 = get_ap(lighting, 'Day')
    dusk30, dusk50 = get_ap(lighting, 'Dusk')
    night30, night50 = get_ap(lighting, 'Night')
    urban30, urban50 = get_ap(env, 'Urban')
    rural30, rural50 = get_ap(env, 'Rural')
    rainy30, rainy50 = get_ap(weather, 'Rainy')
    foggy30, foggy50 = get_ap(weather, 'Foggy')
    cloudy30, cloudy50 = get_ap(weather, 'Cloudy')
    clear30, clear50 = get_ap(weather, 'Clear')
    
    print(f"{'HEAL':<12}|", end="")
    print(f" {day30:>5.1f} {day50:>5.1f}|", end="")
    print(f" {dusk30:>5.1f} {dusk50:>5.1f}|", end="")
    print(f" {night30:>5.1f} {night50:>5.1f}|", end="")
    print(f" {urban30:>5.1f} {urban50:>5.1f}|", end="")
    print(f" {rural30:>5.1f} {rural50:>5.1f}|", end="")
    print(f" {rainy30:>5.1f} {rainy50:>5.1f}|", end="")
    print(f" {foggy30:>5.1f} {foggy50:>5.1f}|", end="")
    print(f" {cloudy30:>5.1f} {cloudy50:>5.1f}|", end="")
    print(f" {clear30:>5.1f} {clear50:>5.1f}|")
    
    # Overall
    overall = results.get('overall', {})
    print(f"\nOverall: AP@0.3={overall.get(0.3,0):.1f}%, AP@0.5={overall.get(0.5,0):.1f}%, AP@0.7={overall.get(0.7,0):.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--eval_epoch', type=int, default=20)
    parser.add_argument('--eval_best', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载配置
    print("加载配置...")
    hypes = yaml_utils.load_yaml(args.config_file, None)
    
    if 'test_dir' in hypes:
        hypes['validate_dir'] = hypes['test_dir']
    
    test_root = hypes['validate_dir']
    print(f"测试数据: {test_root}")
    
    # 获取场景信息
    scenario_infos = get_scenario_infos(test_root)
    
    # 构建数据集
    print("\n构建数据集...")
    dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"样本数: {len(dataset)}")
    
    dataloader = DataLoader(
        dataset, batch_size=1, num_workers=4,
        collate_fn=dataset.collate_batch_test,
        shuffle=False, pin_memory=False, drop_last=False
    )
    
    # 加载模型
    print("\n加载模型...")
    model = train_utils.create_model(hypes)
    model.to(device)
    epoch_id, model = train_utils.load_model(
        args.model_dir, model, args.eval_epoch, start_from_best=args.eval_best
    )
    model.eval()
    print(f"Epoch: {epoch_id}")
    
    # 初始化统计
    scenario_stats = {name: {iou: init_result_stat() for iou in [0.3, 0.5, 0.7]} 
                      for name in scenario_infos}
    
    # 推理
    print("\n开始推理...")
    for batch_data in tqdm(dataloader, total=len(dataloader)):
        metadata_path = batch_data['ego']['metadata_path_list'][0]
        
        # 匹配场景
        scenario_name = None
        for name in scenario_infos:
            # 检查原始名称或符号链接目标
            if name in metadata_path:
                scenario_name = name
                break
            # 检查符号链接指向的路径
            real_path = os.path.realpath(os.path.join(test_root, name))
            if real_path in metadata_path:
                scenario_name = name
                break
        
        if scenario_name is None:
            continue
        
        result_stat = scenario_stats[scenario_name]
        
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            pred_box, pred_score, gt_box, _ = inference_utils.inference_intermediate_fusion(
                batch_data, model, dataset
            )
            
            if pred_box is None:
                continue
            
            for th in [0.3, 0.5, 0.7]:
                eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box, result_stat, th)
    
    # 按条件聚合
    print("\n计算指标...")
    condition_stats = {
        'lighting': defaultdict(list),
        'environment': defaultdict(list),
        'weather': defaultdict(list)
    }
    all_stats = []
    
    for name, stats in scenario_stats.items():
        if name not in scenario_infos:
            continue
        info = scenario_infos[name]
        condition_stats['lighting'][info['lighting']].append(stats)
        condition_stats['environment'][info['environment']].append(stats)
        condition_stats['weather'][info['weather']].append(stats)
        all_stats.append(stats)
    
    # 计算AP
    results = {'lighting': {}, 'environment': {}, 'weather': {}, 'overall': {}}
    
    for cat in ['lighting', 'environment', 'weather']:
        for cond, stats_list in condition_stats[cat].items():
            merged = merge_stats(stats_list)
            aps = calculate_ap(merged)
            results[cat][cond] = aps
            print(f"{cat}/{cond}: AP@0.3={aps[0.3]:.1f}, AP@0.5={aps[0.5]:.1f}")
    
    results['overall'] = calculate_ap(merge_stats(all_stats))
    
    # 打印表格
    print_table4(results, epoch_id)
    
    # 保存结果
    result_file = os.path.join(args.model_dir, f"table4_epoch{epoch_id}.txt")
    with open(result_file, 'w') as f:
        f.write(f"Epoch: {epoch_id}\n\n")
        for cat in ['lighting', 'environment', 'weather']:
            f.write(f"=== {cat} ===\n")
            for cond, aps in results[cat].items():
                f.write(f"{cond}: AP@0.3={aps[0.3]:.1f}, AP@0.5={aps[0.5]:.1f}, AP@0.7={aps[0.7]:.1f}\n")
            f.write("\n")
        f.write(f"Overall: AP@0.3={results['overall'][0.3]:.1f}, AP@0.5={results['overall'][0.5]:.1f}, AP@0.7={results['overall'][0.7]:.1f}\n")
    print(f"\n结果保存至: {result_file}")


if __name__ == '__main__':
    main()
