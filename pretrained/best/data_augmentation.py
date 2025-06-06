import numpy as np
import torch

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes  # 잘라낼 구멍의 개수
        self.length = length    # 각 구멍의 한 변 길이 (정사각형)

    def __call__(self, img):
        h, w = img.size(1), img.size(2)  # img: Tensor(C, H, W), 여기서 H, W만 가져옴
        mask = np.ones((h, w), np.float32)  # 전체가 1로 채워진 마스크 생성

        for _ in range(self.n_holes):
            y = np.random.randint(h)  # 무작위 y 좌표
            x = np.random.randint(w)  # 무작위 x 좌표

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.  # 해당 영역을 0으로 잘라냄

        mask = torch.from_numpy(mask).to(img.device).expand_as(img)  # (H, W) → (C, H, W)로 확장
        img = img * mask  # 이미지에서 잘린 영역은 0으로 (까맣게) 남음

        return img
