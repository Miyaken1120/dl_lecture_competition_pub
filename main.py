import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from pathlib import Path
import os
import time
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any
from enum import Enum, auto
import torchvision.transforms as transforms

class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def compute_epe_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor):
    epe = torch.mean(torch.mean(torch.norm(pred_flow - gt_flow, p=2, dim=1), dim=(1, 2)), dim=0)
    return epe

def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    np.save(f"{file_name}.npy", flow.cpu().numpy())

def save_optical_flow_to_csv(flow: torch.Tensor, file_name: str):
    flow_np = flow.cpu().numpy()
    N, C, H, W = flow_np.shape
    data = {
        'index': [],
        'channel': [],
        'height': [],
        'width': [],
        'value': []
    }
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    data['index'].append(n)
                    data['channel'].append(c)
                    data['height'].append(h)
                    data['width'].append(w)
                    data['value'].append(flow_np[n, c, h, w])
    df = pd.DataFrame(data)
    df.to_csv(f"{file_name}.csv", index=False)

from src.models.evflownet import EVFlowNet
from src.datasets import DatasetProvider, train_collate

class TrainConfig:
    def __init__(self, config):
        self.initial_learning_rate = config.get('initial_learning_rate', 1e-4)
        self.weight_decay = config.get('weight_decay', 1e-4)
        self.epochs = config.get('epochs', 10)
        self.no_batch_norm = config.get('no_batch_norm', False)

class DataLoaderConfig:
    def __init__(self, config):
        self.batch_size = config.get('batch_size', 4)
        self.shuffle = config.get('shuffle', True)

class Args:
    def __init__(self):
        self.dataset_path = "/content/drive/MyDrive/dl_lecture_competition_pub/data"
        self.seed = 42
        self.data_loader = {
            'train': DataLoaderConfig({'batch_size': 4, 'shuffle': True}),
            'test': DataLoaderConfig({'batch_size': 1, 'shuffle': False})
        }
        self.train = TrainConfig({
            'initial_learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'epochs': 10,
            'no_batch_norm': False
        })

args = Args()
set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------
#    Dataloader
# ------------------
loader = DatasetProvider(
    dataset_path=Path(args.dataset_path),
    representation_type=RepresentationType.VOXEL,
    delta_t_ms=100,
    num_bins=4
)
train_set = loader.get_train_dataset()
test_set = loader.get_test_dataset()
collate_fn = train_collate
train_data = DataLoader(train_set,
                        batch_size=args.data_loader['train'].batch_size,
                        shuffle=args.data_loader['train'].shuffle,
                        collate_fn=collate_fn,
                        drop_last=False)
test_data = DataLoader(test_set,
                       batch_size=args.data_loader['test'].batch_size,
                       shuffle=args.data_loader['test'].shuffle,
                       collate_fn=collate_fn,
                       drop_last=False)

'''
train data:
    Type of batch: Dict
    Key: seq_name, Type: list
    Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
    Key: flow_gt, Type: torch.Tensor, Shape: torch.Size([Batch, 2, 480, 640]) => オプティカルフローデータのバッチ
    Key: flow_gt_valid_mask, Type: torch.Tensor, Shape: torch.Size([Batch, 1, 480, 640]) => オプティカルフローデータのvalid. ベースラインでは使わない

test data:
    Type of batch: Dict
    Key: seq_name, Type: list
    Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
'''

# ------------------
#       Model
# ------------------
model = EVFlowNet(args.train).to(device)

# ------------------
#   optimizer
# ------------------
optimizer = torch.optim.Adam(model.parameters(), lr=args.train.initial_learning_rate, weight_decay=args.train.weight_decay)

# ------------------
#   Start training
# ------------------
model.train()
for epoch in range(args.train.epochs):
    total_loss = 0
    print("on epoch: {}".format(epoch + 1))
    for i, batch in enumerate(tqdm(train_data)):
        batch: Dict[str, Any]
        event_image = batch["event_volume"].to(device)  # [B, 4, 480, 640]
        ground_truth_flow = batch["flow_gt"].to(device)  # [B, 2, 480, 640]

        # 中間層の出力を得る
        flow_outputs = model(event_image)  # 例: [flow1, flow2, flow3, flow_final]

        # 複数スケールでのロス計算
        losses = [compute_epe_error(flow, ground_truth_flow) for flow in flow_outputs]
        loss = sum(losses) / len(losses)

        print(f"batch {i} loss: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_data)}')

# Create the directory if it doesn't exist
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

current_time = time.strftime("%Y%m%d%H%M%S")
model_path = f"checkpoints/model_{current_time}.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# ------------------
#   Start predicting
# ------------------
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
flow: torch.Tensor = torch.tensor([]).to(device)
with torch.no_grad():
    print("start test")
    for batch in tqdm(test_data):
        batch: Dict[str, Any]
        event_image = batch["event_volume"].to(device)
        batch_flow = model(event_image)  # [1, 2, 480, 640]

        flow = torch.cat((flow, batch_flow), dim=0)  # [N, 2, 480, 640]
    print("test done")

# ------------------
#  save submission
# ------------------
file_name = "submission_meg_3"
save_optical_flow_to_npy(flow, file_name)

print("Submission file saved.")
