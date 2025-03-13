import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix
import os
from torch.optim.lr_scheduler import StepLR
import logging
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 알약 클래스 정보
PILL_CLASSES = ['Background']
PILL_COLORS = ['Background', 'Red', 'Orange', 'White', 'Black', 'Dark Red', 'Blue']
PILL_SHAPES = ['None', 'Circular', 'Circular', 'Elliptical', 'Circular', 'Elliptical', 'Elliptical']

def find_pills_contour(image):
    """
    알약 군집의 윤곽선을 찾아 ROI를 반환하는 함수
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # 가장 큰 윤곽선 찾기
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # 여유 공간 추가
    padding = 50
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2*padding)
    h = min(image.shape[0] - y, h + 2*padding)
    
    return image[y:y+h, x:x+w]

def process_image(img_array, IMG_SIZE=150):
    """
    이미지 전처리를 위한 함수
    """
    try:
        # 알약 군집 영역 찾기
        roi = find_pills_contour(img_array)
        
        # 노이즈 제거
        denoised = cv2.fastNlMeansDenoisingColored(roi, None, 10, 10, 7, 21)
        
        # 크기 조정
        new_array = cv2.resize(denoised, (IMG_SIZE, IMG_SIZE))
        
        # HSV 변환
        hsv = cv2.cvtColor(new_array, cv2.COLOR_BGR2HSV)
        
        return hsv
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        return None

def get_transform(train):
    """
    데이터 증강 및 변환을 위한 함수
    Args:
        train: 학습용 데이터인지 여부
    Returns:
        변환 함수들의 리스트
    """
    transforms_list = []
    transforms_list.append(transforms.ToTensor())
    
    if train:
        transforms_list.extend([
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(150, scale=(0.8, 1.0), ratio=(0.9, 1.1))
        ])
    
    return transforms.Compose(transforms_list)

class PillDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_path, annotations, transforms=None):
        self.imgs_path = imgs_path
        self.annotations = annotations
        self.transforms = transforms
        
    def __getitem__(self, idx):
        try:
            # 이미지 로드
            img_path = self.imgs_path[idx]
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
                
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            
            # 이미지 전처리 적용
            img = process_image(img)
            if img is None:
                raise ValueError(f"Failed to process image: {img_path}")
                
            img = Image.fromarray(img)
            
            # 바운딩 박스와 라벨 정보
            boxes = self.annotations[idx]['boxes']
            labels = self.annotations[idx]['labels']
            
            # 박스 좌표 검증
            if len(boxes) == 0:
                raise ValueError(f"No bounding boxes found for image: {img_path}")
            
            # 텐서로 변환
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
            # 박스 좌표 정규화
            h, w = img.height, img.width
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h)
            
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            
            if self.transforms is not None:
                img, target = self.transforms(img, target)
                
            return img, target
        except Exception as e:
            logger.error(f"Error loading item {idx}: {str(e)}")
            return None
        
    def __len__(self):
        return len(self.imgs_path)

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_model(model, train_loader, val_loader, optimizer, num_epochs=10, patience=3):
    """
    개선된 모델 학습 함수
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    # 학습률 스케줄러 추가
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 학습
        model.train()
        total_loss = 0
        
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
        
        train_loss = total_loss / len(train_loader)
        
        # 검증
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
        
        val_loss = val_loss / len(val_loader)
        
        # 학습률 조정
        scheduler.step()
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}')
        logger.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 모델 저장 및 조기 종료 확인
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info('Early stopping triggered')
            break

def evaluate_model(model, test_loader):
    """
    모델 평가 함수
    Args:
        model: Faster R-CNN 모델
        test_loader: 테스트 데이터 로더
    Returns:
        confusion matrix
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = list(image.to(device) for image in images)
            outputs = model(images)
            
            for output, target in zip(outputs, targets):
                pred = output['labels'].cpu().numpy()
                label = target['labels'].cpu().numpy()
                all_preds.extend(pred)
                all_labels.extend(label)
    
    return confusion_matrix(all_labels, all_preds)

def visualize_prediction(model, image_path, threshold=0.5, save_path=None):
    """
    개선된 예측 결과 시각화 함수
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    
    # 이미지 로드 및 전처리
    img = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(img_tensor)
        
    img_np = np.array(img)
    
    # 결과 시각화
    for box, score, label in zip(prediction[0]['boxes'], prediction[0]['scores'], prediction[0]['labels']):
        if score > threshold:
            box = box.cpu().numpy()
            
            # 클래스별 색상 지정
            color = plt.cm.rainbow(label / len(PILL_CLASSES))
            color = tuple(int(255 * c) for c in color[:3])
            
            # 바운딩 박스 그리기
            cv2.rectangle(img_np, 
                        (int(box[0]), int(box[1])), 
                        (int(box[2]), int(box[3])), 
                        color, 2)
            
            # 라벨 정보 표시
            label_text = f"{PILL_CLASSES[label]} ({PILL_COLORS[label]}, {PILL_SHAPES[label]})"
            score_text = f"Score: {score:.2f}"
            
            cv2.putText(img_np, label_text, 
                       (int(box[0]), int(box[1]-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 2)
            cv2.putText(img_np, score_text,
                       (int(box[0]), int(box[1]-30)),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, color, 2)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(img_np)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def verify_data_structure():
    """데이터 구조 검증 함수"""
    required_files = [
        'data/metadata/directory_reference_images.xls',
        'data/metadata/directory_consumer_grade_images.xlsx'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
            
    if len(os.listdir('data/rximage')) == 0:
        raise ValueError("No images found in data/rximage directory")

def main():
    try:
        verify_data_structure()
        DATA_ROOT = 'data'
        METADATA_PATH = os.path.join(DATA_ROOT, 'metadata')
        RXIMAGE_PATH = os.path.join(DATA_ROOT, 'rximage')
        
        # 메타데이터 로드
        try:
            reference_metadata = pd.read_excel(
                os.path.join(METADATA_PATH, 'directory_reference_images.xls'),
                engine='xlrd'
            )
            logger.info("Available columns in reference_metadata:")
            logger.info(reference_metadata.columns.tolist())
            
            # 메타데이터에서 고유한 약품 이름 추출
            unique_pills = reference_metadata['name'].unique()
            PILL_CLASSES = ['Background'] + [f'Pill_{i}' for i in range(len(unique_pills))]
            
        except Exception as e:
            logger.error(f"Error reading xls file: {e}")
            try:
                reference_metadata = pd.read_excel(
                    os.path.join(METADATA_PATH, 'directory_reference_images.xlsx'),
                    engine='openpyxl'
                )
            except Exception as e:
                logger.error(f"Error reading xlsx file: {e}")
                raise

        # 이미지 경로와 어노테이션 준비
        train_imgs_path = []
        train_annotations = []

        for idx, row in reference_metadata.iterrows():
            img_path = os.path.join(RXIMAGE_PATH, row['RXBASE ORIGINAL'])
            
            # 이미지 크기 정보 (예시 값, 실제 이미지에 맞게 조정 필요)
            img_width = 800
            img_height = 600
            
            # 전체 이미지를 바운딩 박스로 사용
            bbox = [0, 0, img_width, img_height]
            
            # 레이블 정보 (약품 이름을 기준으로)
            pill_name = row['name']
            # PILL_CLASSES에서 가장 근접한 클래스 찾기
            label_idx = 1  # 기본값으로 Pill1 사용
            
            train_imgs_path.append(img_path)
            train_annotations.append({
                'boxes': [bbox],
                'labels': [label_idx]
            })
        
        # 데이터셋 분할
        train_paths, val_paths, train_annots, val_annots = train_test_split(
            train_imgs_path, train_annotations, test_size=0.2, random_state=42
        )

        val_paths, test_paths, val_annots, test_annots = train_test_split(
            val_paths, val_annots, test_size=0.5, random_state=42
        )
        
        # 데이터셋 생성
        train_dataset = PillDataset(
            imgs_path=train_paths,
            annotations=train_annots,
            transforms=get_transform(train=True)
        )
        val_dataset = PillDataset(
            imgs_path=val_paths,
            annotations=val_annots,
            transforms=get_transform(train=False)
        )
        test_dataset = PillDataset(
            imgs_path=test_paths,
            annotations=test_annots,
            transforms=get_transform(train=False)
        )
        
        # 데이터로더 설정
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=lambda x: tuple(zip(*x))
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=lambda x: tuple(zip(*x))
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=lambda x: tuple(zip(*x))
        )
        
        # 모델 초기화
        num_classes = len(PILL_CLASSES)
        model = get_model(num_classes)
        
        # 옵티마이저 설정
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        
        # 모델 학습
        train_model(model, train_dataloader, val_dataloader, optimizer)
        
        # 모델 평가
        confusion_mat = evaluate_model(model, test_dataloader)
        logger.info("Confusion Matrix:")
        logger.info(confusion_mat)
        
        # 결과 시각화
        visualize_prediction(model, "test_image.jpg", save_path="results/prediction.png")
        
        # 모델 저장 경로 설정
        model_save_path = os.path.join('models', 'best_model.pth')
        results_save_path = os.path.join('results', 'prediction.png')
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()