# -*- coding: utf-8 -*-
# Author: Binyu Zhao <byzhao@stu.hit.edu.cn>
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from opencood.data_utils.datasets.airv2x.early_fusion_dataset import (
    EarlyFusionDatasetAirv2x,
)
from opencood.data_utils.datasets.airv2x.intermediate_fusion_dataset import (
    IntermediateFusionDatasetAirv2x,
)
from opencood.data_utils.datasets.airv2x.intermediate_fusion_dataset_bm2cp import (
    IntermediateFusionDatasetAirv2xBM2CP,
)
from opencood.data_utils.datasets.airv2x.intermediate_fusion_dataset_sicp import (
    IntermediateFusionDatasetAirv2xSiCP,
)
from opencood.data_utils.datasets.dair.intermediate_fusion_dataset import (
    IntermediateFusionDatasetDair,
)

__all__ = {
    "EarlyFusionDatasetAirv2x": EarlyFusionDatasetAirv2x,
    "IntermediateFusionDatasetAirv2x": IntermediateFusionDatasetAirv2x,
    "IntermediateFusionDatasetAirv2xBM2CP": IntermediateFusionDatasetAirv2xBM2CP,
    "IntermediateFusionDatasetAirv2xSiCP": IntermediateFusionDatasetAirv2xSiCP,
    "IntermediateFusionDatasetDair": IntermediateFusionDatasetDair,
}

# AirV2X数据集的评估范围（根据实际情况调整）
GT_RANGE_AIRV2X = [-140, -40, -3, 140, 40, 1]

# AirV2X中不同智能体的通信范围（根据数据集特性设置）
VEHICLE_COM_RANGE = 100  # 车辆通信范围
RSU_COM_RANGE = 150      # 路侧单元通信范围
DRONE_COM_RANGE = 180    # 无人机通信范围

# 与历史代码的常量名保持兼容
VEH_COM_RANGE = VEHICLE_COM_RANGE


def build_dataset(dataset_cfg, visualize=False, train=True):
    """
    构建AirV2X数据集实例的工厂函数
    
    参数:
    - dataset_cfg: 数据集配置字典，必须包含"fusion"]["core_method"键
    - visualize: 是否启用可视化模式
    - train: 是否为训练模式
    
    返回:
    - 对应的数据集实例
    """
    # 从配置中获取数据集名称
    dataset_name = dataset_cfg["fusion"]["core_method"]
    
    # 检查数据集名称是否在支持的列表中
    if dataset_name not in __all__:
        error_message = (
            f"{dataset_name} is not found. "
            f"Supported datasets for AirV2X are: {list(__all__.keys())}"
        )
        raise ValueError(error_message)
    
    # 创建数据集实例
    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train
    )
    
    return dataset