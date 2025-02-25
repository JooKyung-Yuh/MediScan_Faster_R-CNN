# 알약 군집 탐지 및 분류 시스템

## 프로젝트 개요

이 프로젝트는 Faster R-CNN을 활용하여 이미지 내의 알약을 탐지하고 분류하는 딥러닝 시스템입니다. ResNet50과 FPN(Feature Pyramid Network)을 백본으로 사용하여 높은 정확도의 객체 탐지를 수행합니다.

## 주요 기능

- 이미지 내 알약 위치 탐지
- 알약 종류 분류
- 실시간 시각화 및 결과 출력

## 기술 스택

- Python 3.10.13
- PyTorch
- torchvision
- OpenCV
- PIL (Python Imaging Library)
- NumPy
- Matplotlib

## 설치 방법

1. 가상환경 생성 및 활성화

```bash
python -m venv pill_detection_env
source pill_detection_env/bin/activate  # Linux/Mac
# or
.\pill_detection_env\Scripts\activate  # Windows
```

2. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

## 프로젝트 구조

```
pill_detection/
├── data/               # 데이터셋 저장 디렉토리
├── notebooks/          # Jupyter 노트북 파일들
├── results/            # 모델 출력 결과
├── checkpoints/        # 모델 체크포인트
├── main.ipynb          # 메인 실행 파일
├── requirements.txt    # 의존성 패키지 목록
└── .gitignore         # Git 제외 파일 목록
```

## 사용 방법

1. 데이터셋 준비

   - `data/` 디렉토리에 학습용 이미지와 라벨 데이터를 위치시킵니다.
   - 데이터는 이미지와 해당하는 바운딩 박스 좌표가 포함된 어노테이션이 필요합니다.

2. 모델 학습

```python
# 모델 초기화
model = get_model(num_classes=2)  # 배경 + 알약 클래스

# 학습 실행
train_model(model, train_dataloader, optimizer, num_epochs=10)
```

3. 추론 및 시각화

```python
# 테스트 이미지에 대한 예측 결과 시각화
visualize_prediction(model, "test_image.jpg", threshold=0.5)
```

## 주요 컴포넌트 설명

### PillDataset 클래스

- 커스텀 데이터셋 클래스
- 이미지와 어노테이션을 로드하고 전처리
- 참조 코드:

```34:64:main.ipynb
    "## 데이터셋 클래스 정의\n",
    "class PillDataset(torch.utils.data.Dataset):\n",
    "    def __init__(self, imgs_path, annotations, transforms=None):\n",
    "        self.imgs_path = imgs_path\n",
    "        self.annotations = annotations\n",
    "        self.transforms = transforms\n",
    "        \n",
    "    def __getitem__(self, idx):\n",
    "        # 이미지 로드\n",
    "        img_path = self.imgs_path[idx]\n",
    "        img = Image.open(img_path).convert(\"RGB\")\n",
    "        \n",
    "        # 바운딩 박스와 라벨 정보\n",
    "        boxes = self.annotations[idx]['boxes']\n",
    "        labels = self.annotations[idx]['labels']\n",
    "        \n",
    "        # 텐서로 변환\n",
    "        boxes = torch.as_tensor(boxes, dtype=torch.float32)\n",
    "        labels = torch.as_tensor(labels, dtype=torch.int64)\n",
    "        \n",
    "        target = {}\n",
    "        target[\"boxes\"] = boxes\n",
    "        target[\"labels\"] = labels\n",
    "        \n",
    "        if self.transforms is not None:\n",
    "            img, target = self.transforms(img, target)\n",
    "            \n",
    "        return img, target\n",
    "    \n",
    "    def __len__(self):\n",
    "        return len(self.imgs_path)\n"
```

### 모델 아키텍처

- Faster R-CNN with ResNet50 백본
- FPN(Feature Pyramid Network) 사용
- 참조 코드:

```74:78:main.ipynb
    "def get_model(num_classes):\n",
    "    model = fasterrcnn_resnet50_fpn(pretrained=True)\n",
    "    in_features = model.roi_heads.box_predictor.cls_score.in_features\n",
    "    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)\n",
    "    return model\n",
```

## 라이선스

MIT License

## 기여 방법

1. 이 저장소를 포크합니다.
2. 새로운 브랜치를 생성합니다.
3. 변경사항을 커밋합니다.
4. 브랜치에 푸시합니다.
5. Pull Request를 생성합니다.

## 주의사항

- CUDA 지원 GPU 사용을 권장합니다.
- 충분한 RAM (최소 8GB 이상 권장)
- 데이터셋은 별도로 준비해야 합니다.

## 문의사항

이슈를 통해 문의해주시기 바랍니다.
