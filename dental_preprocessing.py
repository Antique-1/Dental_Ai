"""
Dental Panoramic X-Ray 데이터 전처리 코드
프로젝트: Dental Data Analysis
작성자: 김도현

[데이터셋] kaggle - thunderpede/panoramic-dental-dataset
  - 라벨 형식: 픽셀 단위 세그멘테이션 마스크 (PNG)
  - 본 코드: 마스크 → YOLO 바운딩박스 (.txt) 자동 변환 포함
  
[폴더 구조 가정]
  data/raw/
    ├── images/   ← 원본 X-Ray (jpg 또는 png)
    └── masks/    ← 충치 영역 마스크 (동일 파일명, png)
"""

import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A
import yaml


# ─────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────
IMG_SIZE      = 640
TRAIN_RATIO   = 0.8
VAL_RATIO     = 0.1
TEST_RATIO    = 0.1
MASK_MIN_AREA = 100        # 노이즈 제거: 이 픽셀 수 미만 컨투어 무시

RAW_IMAGE_DIR = "data/raw/images"
RAW_MASK_DIR  = "data/raw/masks"   # ← 세그멘테이션 마스크 폴더
OUTPUT_DIR    = "data/processed"


# ─────────────────────────────────────────
# 1. 디렉토리 생성
# ─────────────────────────────────────────
def create_directories(base_dir: str):
    for split in ["train", "val", "test"]:
        for sub in ["images", "labels"]:
            Path(f"{base_dir}/{split}/{sub}").mkdir(parents=True, exist_ok=True)
    print("✅ 디렉토리 구조 생성 완료")


# ─────────────────────────────────────────
# 2. 마스크 → YOLO 바운딩박스 변환  ★핵심 추가★
# ─────────────────────────────────────────
def mask_to_yolo_bboxes(mask_path: str,
                         orig_w: int, orig_h: int) -> tuple:
    """
    세그멘테이션 마스크(PNG) → YOLO 형식 바운딩박스 리스트 변환
    
    - 마스크: 흰색(255) = 충치 영역, 검정(0) = 배경
    - 연결된 흰색 영역(컨투어)마다 바운딩박스 1개 생성
    - 반환: (bboxes, class_labels)  bboxes = [(cx,cy,w,h), ...]  0~1 정규화
    """
    if not Path(mask_path).exists():
        return [], []

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return [], []

    # 마스크 크기가 원본과 다를 경우 맞춤
    if mask.shape != (orig_h, orig_w):
        mask = cv2.resize(mask, (orig_w, orig_h),
                          interpolation=cv2.INTER_NEAREST)

    # 이진화 (임계값 127)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 컨투어 탐지
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    bboxes, class_labels = [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MASK_MIN_AREA:          # 노이즈 무시
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # YOLO 형식: 중심점 + 너비/높이, 0~1 정규화
        cx = (x + w / 2) / orig_w
        cy = (y + h / 2) / orig_h
        nw = w / orig_w
        nh = h / orig_h

        # 경계 클리핑
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(0.001, min(1.0, nw))
        nh = max(0.001, min(1.0, nh))

        bboxes.append((cx, cy, nw, nh))
        class_labels.append(0)            # 0 = cavity

    return bboxes, class_labels


# ─────────────────────────────────────────
# 3. 이미지 전처리
# ─────────────────────────────────────────
def letterbox(img: np.ndarray, new_shape=(640, 640),
              color=(114, 114, 114)) -> tuple:
    """
    비율 유지 리사이즈 + 패딩
    반환: (리사이즈된 이미지, scale, (pad_left, pad_top))
    """
    h, w  = img.shape[:2]
    th, tw = new_shape
    scale  = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)

    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_top    = (th - nh) // 2
    pad_bottom = th - nh - pad_top
    pad_left   = (tw - nw) // 2
    pad_right  = tw - nw - pad_left

    img = cv2.copyMakeBorder(img, pad_top, pad_bottom,
                             pad_left, pad_right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, scale, (pad_left, pad_top)


def adjust_bboxes_for_letterbox(bboxes: list, orig_w: int, orig_h: int,
                                  scale: float, pad_left: int,
                                  pad_top: int, new_size: int) -> list:
    """
    letterbox 변환에 맞춰 바운딩박스 좌표 재조정
    (원본 정규화 좌표 → letterbox 정규화 좌표)
    """
    adjusted = []
    for (cx, cy, bw, bh) in bboxes:
        # 픽셀 좌표로 변환
        px_cx = cx * orig_w * scale + pad_left
        px_cy = cy * orig_h * scale + pad_top
        px_bw = bw * orig_w * scale
        px_bh = bh * orig_h * scale

        # 다시 정규화
        new_cx = px_cx / new_size
        new_cy = px_cy / new_size
        new_bw = px_bw / new_size
        new_bh = px_bh / new_size

        new_cx = max(0.0, min(1.0, new_cx))
        new_cy = max(0.0, min(1.0, new_cy))
        new_bw = max(0.001, min(1.0, new_bw))
        new_bh = max(0.001, min(1.0, new_bh))

        adjusted.append((new_cx, new_cy, new_bw, new_bh))
    return adjusted


def preprocess_image(img_path: str, size: int = IMG_SIZE):
    """이미지 전처리 → (처리된 이미지, scale, pad)"""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {img_path}")

    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    img      = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    img, scale, pad = letterbox(img, new_shape=(size, size))
    img = img.astype(np.float32) / 255.0
    return img, scale, pad


# ─────────────────────────────────────────
# 4. 데이터 증강
# ─────────────────────────────────────────
def get_augmentation_pipeline() -> A.Compose:
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


def augment_image(img: np.ndarray, bboxes: list,
                  class_labels: list, pipeline: A.Compose) -> tuple:
    img_uint8 = (img * 255).astype(np.uint8)
    result    = pipeline(image=img_uint8,
                         bboxes=bboxes,
                         class_labels=class_labels)
    aug_img   = result["image"].astype(np.float32) / 255.0
    return aug_img, result["bboxes"], result["class_labels"]


# ─────────────────────────────────────────
# 5. 라벨 저장
# ─────────────────────────────────────────
def save_yolo_label(label_path: str, bboxes: list, class_labels: list):
    with open(label_path, "w") as f:
        for cls, (cx, cy, bw, bh) in zip(class_labels, bboxes):
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


# ─────────────────────────────────────────
# 6. 데이터셋 분할 및 저장 (전체 파이프라인)
# ─────────────────────────────────────────
def split_and_save(image_dir: str, mask_dir: str,
                   output_dir: str, augment: bool = True):
    """
    마스크 → YOLO 변환 포함 전체 파이프라인
    image_dir : 원본 X-Ray 이미지 폴더
    mask_dir  : 세그멘테이션 마스크 폴더
    """
    image_paths = sorted(Path(image_dir).glob("*.jpg")) + \
                  sorted(Path(image_dir).glob("*.png"))

    if not image_paths:
        print(f"⚠️  이미지를 찾을 수 없습니다: {image_dir}")
        return

    print(f"📂 총 이미지 수: {len(image_paths)}")

    # 마스크가 없는 이미지 필터링
    valid_paths = []
    for p in image_paths:
        mask_path = Path(mask_dir) / f"{p.stem}.png"
        if not mask_path.exists():
            # jpg 마스크도 시도
            mask_path = Path(mask_dir) / f"{p.stem}.jpg"
        if mask_path.exists():
            valid_paths.append(p)
        else:
            print(f"  ⚠️  마스크 없음, 건너뜀: {p.name}")

    print(f"📂 마스크 매칭된 이미지: {len(valid_paths)}장")

    if not valid_paths:
        print("⛔ 처리할 이미지가 없습니다. 폴더 경로를 확인하세요.")
        return

    # 분할
    train_p, temp_p = train_test_split(
        valid_paths, test_size=(VAL_RATIO + TEST_RATIO), random_state=42)
    val_p, test_p = train_test_split(
        temp_p,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=42)

    splits      = {"train": train_p, "val": val_p, "test": test_p}
    aug_pipeline = get_augmentation_pipeline()
    skip_count  = 0

    for split, paths in splits.items():
        print(f"\n🔄 {split} 처리 중 ({len(paths)}장)...")
        for img_path in paths:
            stem = img_path.stem

            # 마스크 경로 탐색
            mask_path = Path(mask_dir) / f"{stem}.png"
            if not mask_path.exists():
                mask_path = Path(mask_dir) / f"{stem}.jpg"

            # 원본 이미지 크기 확인 (마스크 변환에 필요)
            raw = cv2.imread(str(img_path))
            if raw is None:
                print(f"  ⛔ 읽기 실패: {img_path.name}")
                skip_count += 1
                continue
            orig_h, orig_w = raw.shape[:2]

            # ★ 마스크 → YOLO 바운딩박스 변환
            bboxes, class_labels = mask_to_yolo_bboxes(
                str(mask_path), orig_w, orig_h)

            if not bboxes:
                print(f"  ℹ️  충치 없음(배경): {img_path.name}")

            # 이미지 전처리 (CLAHE + letterbox)
            try:
                img, scale, (pad_left, pad_top) = preprocess_image(
                    str(img_path))
            except FileNotFoundError as e:
                print(f"  ⛔ {e}")
                skip_count += 1
                continue

            # 바운딩박스 letterbox 보정
            if bboxes:
                bboxes = adjust_bboxes_for_letterbox(
                    bboxes, orig_w, orig_h, scale,
                    pad_left, pad_top, IMG_SIZE)

            # 학습셋 증강 (원본 + 증강본 1장)
            if split == "train" and augment and bboxes:
                try:
                    aug_img, aug_bb, aug_cls = augment_image(
                        img, bboxes, class_labels, aug_pipeline)
                    aug_save = (aug_img * 255).astype(np.uint8)
                    cv2.imwrite(
                        f"{output_dir}/{split}/images/{stem}_aug.jpg",
                        aug_save)
                    save_yolo_label(
                        f"{output_dir}/{split}/labels/{stem}_aug.txt",
                        aug_bb, aug_cls)
                except Exception as e:
                    print(f"  ⚠️  증강 실패 ({stem}): {e}")

            # 원본 저장
            img_save = (img * 255).astype(np.uint8)
            cv2.imwrite(f"{output_dir}/{split}/images/{stem}.jpg", img_save)
            save_yolo_label(
                f"{output_dir}/{split}/labels/{stem}.txt",
                bboxes, class_labels)

        print(f"  ✅ {split} 완료")

    if skip_count:
        print(f"\n⚠️  총 {skip_count}장 건너뜀")


# ─────────────────────────────────────────
# 7. data.yaml 생성
# ─────────────────────────────────────────
def create_yaml(output_dir: str, class_names: list):
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
# 8. 데이터 통계 출력
# ─────────────────────────────────────────
def print_dataset_stats(output_dir: str):
    print("\n📊 데이터셋 통계")
    print("─" * 30)
    total = 0
    for split in ["train", "val", "test"]:
        img_dir = Path(output_dir) / split / "images"
        count   = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        total  += count
        print(f"  {split:5s}: {count:4d}장")
    print(f"  {'합계':5s}: {total:4d}장")
    print("─" * 30)


# ─────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────
if __name__ == "__main__":

    CLASS_NAMES = ["cavity"]

    print("=" * 45)
    print("  Dental X-Ray 데이터 전처리 시작")
    print("  (마스크 → YOLO 변환 포함)")
    print("=" * 45)

    # 폴더 구조 안내
    print(f"\n📁 이미지 폴더: {RAW_IMAGE_DIR}")
    print(f"📁 마스크 폴더: {RAW_MASK_DIR}")
    print("  ※ 마스크 파일명은 이미지 파일명과 동일해야 합니다.")
    print("  ※ 마스크: 흰색(255)=충치, 검정(0)=배경\n")

    create_directories(OUTPUT_DIR)
    split_and_save(RAW_IMAGE_DIR, RAW_MASK_DIR, OUTPUT_DIR, augment=True)
    create_yaml(OUTPUT_DIR, CLASS_NAMES)
    print_dataset_stats(OUTPUT_DIR)

    print("\n🎉 전처리 완료! 다음 단계: python dental_train.py")
