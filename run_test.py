import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from scipy import stats
from data.dataset_spaq import SPAQDataset
from data.dataset_kadid10k import KADID10KDataset
from data.dataset_tid2013 import TID2013Dataset
from data.dataset_csiq import CSIQDataset
from data.dataset_clive import CLIVEDataset
from data.dataset_koniq10k import KONIQ10KDataset
from data.dataset_live import LIVEDataset
from models.attention_se import EnhancedDistortionDetectionModel
from utils.utils import load_config

# ✅ SRCC 및 PLCC 계산
def calculate_srcc_plcc(preds, targets):
    preds, targets = preds.cpu().numpy(), targets.cpu().numpy()
    srocc, _ = stats.spearmanr(preds.flatten(), targets.flatten())
    plcc, _ = stats.pearsonr(preds.flatten(), targets.flatten())
    return srocc, plcc

# ✅ 테스트 루프
def test(model, dataloader, device):
    model.eval()
    srocc_values, plcc_values = [], []

    with torch.no_grad():
        for batch in dataloader:
            img_A = batch["img_A"].to(device)
            targets = batch["mos"].to(device)

            preds = model(img_A)
            srocc, plcc = calculate_srcc_plcc(preds, targets)

            srocc_values.append(srocc)
            plcc_values.append(plcc)

    return np.mean(srocc_values), np.mean(plcc_values)

if __name__ == "__main__":
    # ✅ 설정 파일 로드
    config_path = "E:/ARNIQA - SE - mix/ARNIQA/config.yaml"
    args = load_config(config_path)

    # ✅ GPU 설정
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # ✅ 저장된 모델 불러오기
    model_path = "E:/ARNIQA - SE - mix/ARNIQA/experiments/my_experiment/regressors/koniq/epoch_17_srocc_0.790.pth"
    model = EnhancedDistortionDetectionModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # ✅ 데이터셋 로드
    datasets = {
        "KADID10K": KADID10KDataset(args.data_base_path_kadid, crop_size=224),
        "TID2013": TID2013Dataset(args.data_base_path_tid, crop_size=224),
        "SPAQ": SPAQDataset(args.data_base_path_spaq, crop_size=224),
        "CSIQ": CSIQDataset(args.data_base_path_csiq, crop_size=224),
        "CLIVE": CLIVEDataset(args.data_base_path_clive, crop_size=224),
        "KONIQ10K": KONIQ10KDataset(args.data_base_path_koniq, crop_size=224),
        "LIVE": LIVEDataset(args.data_base_path_live, crop_size=224),
    }

    results = {}

    for dataset_name, dataset in datasets.items():
        dataloader = DataLoader(dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=4)

        print(f"\n🔹 **Testing on {dataset_name} dataset...**")
        srocc, plcc = test(model, dataloader, device)
        results[dataset_name] = {"SROCC": srocc, "PLCC": plcc}

        print(f"✅ {dataset_name}: SROCC: {srocc:.4f}, PLCC: {plcc:.4f}")

    print("\n🔹 **Final Test Results Across Datasets:** 🔹")
    for dataset, metrics in results.items():
        print(f"📌 {dataset}: SROCC: {metrics['SROCC']:.4f}, PLCC: {metrics['PLCC']:.4f}")

""" 
import torch

# GPU가 인식되는지 확인
print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")
print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}")

# 현재 활성화된 GPU 정보 출력
if torch.cuda.is_available():
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"GPU 총 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Tensor 연산이 GPU에서 수행되는지 확인
tensor = torch.randn(1).to("cuda")
print(f"GPU에서 생성된 텐서: {tensor}")
 """