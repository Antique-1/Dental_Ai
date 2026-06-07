"""
Dental Panoramic X-Ray 데이터 전처리 코드
프로젝트: Dental Data Analysis
작성자: 김도현
"""

import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A
import yaml


# ─────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────
IMG_SIZE = 640          # YOLO 표준 입력 크기
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

RAW_IMAGE_DIR  = "data/raw/images"      # 원본 이미지 경로
RAW_LABEL_DIR  = "data/raw/labels"      # 원본 라벨 경로 (YOLO .txt 형식)
OUTPUT_DIR     = "data/processed"       # 전처리 결과 저장 경로


# ─────────────────────────────────────────
# 1. 디렉토리 생성
# ─────────────────────────────────────────
def create_directories(base_dir: str):
    """학습/검증/테스트 디렉토리 구조 생성"""
    for split in ["train", "val", "test"]:
        for sub in ["images", "labels"]:
            Path(f"{base_dir}/{split}/{sub}").mkdir(parents=True, exist_ok=True)
    print("✅ 디렉토리 구조 생성 완료")


# ─────────────────────────────────────────
# 2. 이미지 전처리
# ─────────────────────────────────────────
def preprocess_image(img_path: str, size: int = IMG_SIZE) -> np.ndarray:
    """
    단일 이미지 전처리
      - 그레이스케일 → BGR 변환 (YOLO 입력 호환)
      - CLAHE 로 대비 향상 (X-Ray 특성 반영)
      - 크기 통일 (letterbox 방식으로 비율 유지)
      - [0, 1] 정규화
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {img_path}")

    # 그레이스케일 변환 후 CLAHE 적용
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # Letterbox 리사이즈 (비율 유지, 패딩 추가)
    img = letterbox(img, new_shape=(size, size))

    # 정규화 (0~255 → 0.0~1.0)
    img = img.astype(np.float32) / 255.0

    return img


def letterbox(img: np.ndarray, new_shape=(640, 640),
              color=(114, 114, 114)) -> np.ndarray:
    """비율을 유지하며 지정 크기로 리사이즈 후 패딩 추가"""
    h, w = img.shape[:2]
    target_h, target_w = new_shape

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top    = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left   = (target_w - new_w) // 2
    pad_right  = target_w - new_w - pad_left

    img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                             cv2.BORDER_CONSTANT, value=color)
    return img


# ─────────────────────────────────────────
# 3. 데이터 증강
# ─────────────────────────────────────────
def get_augmentation_pipeline() -> A.Compose:
    """
    X-Ray 이미지에 적합한 증강 파이프라인
      - 좌우 반전 (치아 구조 대칭성)
      - 밝기/대비 조정 (촬영 환경 차이 모사)
      - 소량 회전 (촬영 각도 차이)
      - 가우시안 노이즈 (노이즈 내성)
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.5),
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.4),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.3),
    ], bbox_params=A.BboxParams(format="yolo",
                                label_fields=["class_labels"],
                                min_visibility=0.3))


def augment_image(img: np.ndarray, bboxes: list, class_labels: list,
                  pipeline: A.Compose) -> tuple:
    """이미지 + 바운딩박스 함께 증강"""
    # albumentations 는 uint8 입력 필요
    img_uint8 = (img * 255).astype(np.uint8)
    result = pipeline(image=img_uint8, bboxes=bboxes, class_labels=class_labels)
    aug_img = result["image"].astype(np.float32) / 255.0
    return aug_img, result["bboxes"], result["class_labels"]


# ─────────────────────────────────────────
# 4. 라벨 파싱 / 저장 (YOLO 형식)
# ─────────────────────────────────────────
def parse_yolo_label(label_path: str) -> tuple:
    """YOLO .txt 라벨 파일 파싱 → (bboxes, class_labels)"""
    bboxes, class_labels = [], []
    if not os.path.exists(label_path):
        return bboxes, class_labels

    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                cls, x, y, w, h = parts
                class_labels.append(int(cls))
                bboxes.append((float(x), float(y), float(w), float(h)))
    return bboxes, class_labels


def save_yolo_label(label_path: str, bboxes: list, class_labels: list):
    """YOLO 형식으로 라벨 저장"""
    with open(label_path, "w") as f:
        for cls, (x, y, w, h) in zip(class_labels, bboxes):
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


# ─────────────────────────────────────────
# 5. 데이터셋 분할 및 저장
# ─────────────────────────────────────────
def split_and_save(image_dir: str, label_dir: str,
                   output_dir: str, augment: bool = True):
    """
    전체 파이프라인:
      원본 이미지 로드 → 전처리 → 분할 → (학습셋만) 증강 → 저장
    """
    image_paths = sorted(Path(image_dir).glob("*.jpg")) + \
                  sorted(Path(image_dir).glob("*.png"))

    if not image_paths:
        print(f"⚠️  이미지를 찾을 수 없습니다: {image_dir}")
        return

    print(f"📂 총 이미지 수: {len(image_paths)}")

    # 분할
    train_paths, temp_paths = train_test_split(
        image_paths, test_size=(VAL_RATIO + TEST_RATIO), random_state=42)
    val_paths, test_paths = train_test_split(
        temp_paths,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=42)

    splits = {"train": train_paths, "val": val_paths, "test": test_paths}
    aug_pipeline = get_augmentation_pipeline()

    for split, paths in splits.items():
        print(f"\n🔄 {split} 처리 중 ({len(paths)}장)...")
        for img_path in paths:
            stem = img_path.stem
            label_path = Path(label_dir) / f"{stem}.txt"

            # 전처리
            try:
                img = preprocess_image(str(img_path))
            except FileNotFoundError as e:
                print(f"  ⛔ {e}")
                continue

            bboxes, class_labels = parse_yolo_label(str(label_path))

            # 학습셋만 증강 (원본 + 증강본 1장)
            if split == "train" and augment and bboxes:
                aug_img, aug_bboxes, aug_cls = augment_image(
                    img, bboxes, class_labels, aug_pipeline)

                aug_img_save = (aug_img * 255).astype(np.uint8)
                cv2.imwrite(
                    f"{output_dir}/{split}/images/{stem}_aug.jpg", aug_img_save)
                save_yolo_label(
                    f"{output_dir}/{split}/labels/{stem}_aug.txt",
                    aug_bboxes, aug_cls)

            # 원본 저장
            img_save = (img * 255).astype(np.uint8)
            cv2.imwrite(f"{output_dir}/{split}/images/{stem}.jpg", img_save)
            save_yolo_label(
                f"{output_dir}/{split}/labels/{stem}.txt",
                bboxes, class_labels)

        print(f"  ✅ {split} 완료")


# ─────────────────────────────────────────
# 6. YOLO 학습용 yaml 생성
# ─────────────────────────────────────────
def create_yaml(output_dir: str, class_names: list):
    """YOLOv8 학습에 필요한 data.yaml 생성"""
    config = {
        "path"  : os.path.abspath(output_dir),
        "train" : "train/images",
        "val"   : "val/images",
        "test"  : "test/images",
        "nc"    : len(class_names),
        "names" : class_names,
    }
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"\n📄 data.yaml 생성 완료: {yaml_path}")


# ─────────────────────────────────────────
# 7. 데이터 통계 출력
# ─────────────────────────────────────────
def print_dataset_stats(output_dir: str):
    """전처리 후 데이터셋 통계 출력"""
    print("\n📊 데이터셋 통계")
    print("─" * 30)
    total = 0
    for split in ["train", "val", "test"]:
        img_dir = Path(output_dir) / split / "images"
        count = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        total += count
        print(f"  {split:5s}: {count:4d}장")
    print(f"  {'합계':5s}: {total:4d}장")
    print("─" * 30)


# ─────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────
if __name__ == "__main__":

    CLASS_NAMES = ["cavity"]   # 클래스 이름 (충치)

    print("=" * 40)
    print("  Dental X-Ray 데이터 전처리 시작")
    print("=" * 40)

    create_directories(OUTPUT_DIR)
    split_and_save(RAW_IMAGE_DIR, RAW_LABEL_DIR, OUTPUT_DIR, augment=True)
    create_yaml(OUTPUT_DIR, CLASS_NAMES)
    print_dataset_stats(OUTPUT_DIR)

    print("\n🎉 전처리 완료! 다음 단계: YOLO 모델 학습")
