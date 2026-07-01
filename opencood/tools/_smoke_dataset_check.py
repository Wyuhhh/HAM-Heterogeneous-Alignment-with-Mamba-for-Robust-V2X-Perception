import os
import yaml
from opencood.data_utils.datasets.airv2x.basedataset import BaseDataset
from opencood.hypes_yaml.yaml_utils import load_yaml

# Quick smoke-test: load config used for training (try common locations)
cands = [
    'opencood/hypes_yaml/airv2x/lidar/det/airv2x_HEAL_collab_lidar.yaml',
    'opencood/hypes_yaml/airv2x/lidar/det/single/airv2x_HEAL_vehicle_lidar.yaml'
]
config_path = None
for p in cands:
    if os.path.exists(p):
        config_path = p
        break
if config_path is None:
    raise SystemExit('No config yaml found in candidate paths; please pass a config.')

print('Using config:', config_path)
params = load_yaml(config_path)
# Some configs expect top-level keys; training script usually passes a params dict.
# Instantiate dataset in train mode to check indexing and ignore_frames.

ds = BaseDataset(params=params, visualize=False, train=True)
print('Total samples (len):', len(ds))

# Report per-scenario lengths
for i, sd in ds.scenario_database.items():
    # pick first cav to inspect timestamps
    cav0 = next(iter(sd.keys()))
    frames = list(sd[cav0].keys())
    print(f'scenario idx {i}, name unknown, frames count: {len(frames)}')

# Print first 10 idx -> timestamp mapping
print('\nSample idx -> timestamp_key mappings (first 10):')
for idx in range(min(10, len(ds))):
    # find scenario index and timestamp index similar to retrieve_base_data
    scenario_index = 0
    for j, ele in enumerate(ds.len_record):
        if idx < ele:
            scenario_index = j
            break
    timestamp_index = idx if scenario_index == 0 else idx - ds.len_record[scenario_index - 1]
    sd = ds.scenario_database[scenario_index]
    ts_key = ds.return_timestamp_key(sd, timestamp_index)
    print(f'idx {idx} -> scenario_index {scenario_index}, timestamp_key {ts_key}')

print('\nignore_frames set size:', len(getattr(ds, 'ignore_frames', set())))
print('Done.')
