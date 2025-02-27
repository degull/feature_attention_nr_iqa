import random
import torch

class HardNegativeSampler:
    def __init__(self, dataset1, dataset2, num_negatives=16):  # ✅ Negative 개수를 16개로 조정
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.num_negatives = num_negatives

    def sample_negatives(self, index):
        negative_indices = list(range(len(self.dataset2)))
        
        batch_indices = index.tolist() if isinstance(index, torch.Tensor) else index
        
        for idx in batch_indices:
            if idx in negative_indices:
                negative_indices.remove(idx)

        # ✅ Negative 개수를 16개로 맞춤 (가능한 최대 개수만큼)
        sampled_indices = random.sample(negative_indices, min(self.num_negatives, len(negative_indices)))
        negatives = [self.dataset2[i]["img_A"] for i in sampled_indices]

        return torch.stack(negatives, dim=0)  # ✅ Tensor 반환
