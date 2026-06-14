"""
Dental Panoramic X-Ray 데이터 전처리 코드
프로젝트: Dental Data Analysis
작성자: 김도현

[데이터셋] kaggle - thunderpede/panoramic-dental-dataset
  실측 구조:
    archive/
      ├── images/                  원본 X-Ray (.png, 2943x1435)
      ├── images_cut/              치아 경계로 크롭한 이미지 (.png, 1536x768)  ★사용★
      ├── labels/                  원본 기준 마스크 (.png)
      ├── labels_cut/              크롭 기준 마스크 (.png)
      └── annotations/
            ├── bboxes_caries/     충치 bbox (.txt)  ← 본 코드가 사용
            └── bboxes_teeth/      치아 bbox (.txt)

  ★ 중요: bboxes_caries 좌표계는 원본(images)이 아니라
          images_cut(1536x768) 기준이다.
          (labels_cut 충치픽셀 100%가 bbox 내부에 포함됨을 실측 확인)
          따라서 반드시 images_cut 과 짝지어 사용해야 한다.

  bbox 형식: "x1 y1 x2 y2" (좌상단, 우하단 픽셀 좌표, 공백 구분)
  → YOLO 형식 "cls cx cy w h" (0~1 정규화) 으로 변환
"""

import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A
import yaml


# ─────────────────────────────────────────
# 설정값  ★ 압축 푼 폴더 경로에 맞게 RAW_DIR 만 바꾸면 됩니다 ★
# ─────────────────────────────────────────
RAW_DIR    = "C:/archive"                          # 압축 해제 폴더
RAW_IMAGE_DIR = f"{RAW_DIR}/images_cut"          # ★ 크롭 이미지 사용 (bbox 좌표계와 일치)
RAW_BBOX_DIR  = f"{RAW_DIR}/annotations/bboxes_caries"
OUTPUT_DIR    = "data/processed"

IMG_SIZE     = 640
TRAIN_RATIO  = 0.8
VAL_RATIO    = 0.1
TEST_RATIO   = 0.1
AUG_PER_IMG  = 3            # 데이터가 100장으로 적어 증강본을 늘림


# ─────────────────────────────────────────
# 1. 디렉토리 생성
# ─────────────────────────────────────────
def create_directories(base_dir: str):
    for split in ["train", "val", "test"]:
        for sub in ["images", "labels"]:
            Path(f"{base_dir}/{split}/{sub}").mkdir(parents=True, exist_ok=True)
    print("✅ 디렉토리 구조 생성 완료")


# ─────────────────────────────────────────
# 2. 충치 bbox(.txt) → YOLO 형식 변환  ★핵심★
# ─────────────────────────────────────────
def parse_caries_bbox(bbox_path: str, img_w: int, img_h: int) -> tuple:
    """
    "x1 y1 x2 y2" (코너 픽셀) → YOLO "cx cy w h" (0~1 정규화)
    반환: (bboxes, class_labels)
    """
    bboxes, class_labels = [], []
    if not Path(bbox_path).exists():
        return bboxes, class_labels

    with open(bbox_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            x1, y1, x2, y2 = map(float, parts)

            # 좌표 정렬 (혹시 순서가 뒤바뀐 경우 대비)
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            # YOLO 형식: 중심점 + 너비/높이, 정규화
            cx = ((x_min + x_max) / 2) / img_w
            cy = ((y_min + y_max) / 2) / img_h
            bw = (x_max - x_min) / img_w
            bh = (y_max - y_min) / img_h

            # 경계 클리핑
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            bw = max(0.001, min(1.0, bw))
            bh = max(0.001, min(1.0, bh))

            bboxes.append((cx, cy, bw, bh))
            class_labels.append(0)        # 0 = cavity

    return bboxes, class_labels


# ─────────────────────────────────────────
# 3. 이미지 전처리 (CLAHE + letterbox)
# ─────────────────────────────────────────
def letterbox(img: np.ndarray, new_shape=(640, 640),
              color=(114, 114, 114)) -> tuple:
    h, w   = img.shape[:2]
    th, tw = new_shape
    scale  = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_top    = (th - nh) // 2
    pad_bottom = th - nh - pad_top
    pad_left   = (tw - nw) // 2
    pad_right  = tw - nw - pad_left

    img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, scale, (pad_left, pad_top)


def adjust_bboxes_for_letterbox(bboxes, orig_w, orig_h,
                                 scale, pad_left, pad_top, new_size):
    """원본 정규화 bbox → letterbox 정규화 bbox"""
    adjusted = []
    for (cx, cy, bw, bh) in bboxes:
        px_cx = cx * orig_w * scale + pad_left
        px_cy = cy * orig_h * scale + pad_top
        px_bw = bw * orig_w * scale
        px_bh = bh * orig_h * scale

        ncx = max(0.0, min(1.0, px_cx / new_size))
        ncy = max(0.0, min(1.0, px_cy / new_size))
        nbw = max(0.001, min(1.0, px_bw / new_size))
        nbh = max(0.001, min(1.0, px_bh / new_size))
        adjusted.append((ncx, ncy, nbw, nbh))
    return adjusted


def preprocess_image(img_path: str, size: int = IMG_SIZE):
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
# 4. 데이터 증강 (albumentations 신/구 버전 모두 호환)
# ─────────────────────────────────────────
def get_augmentation_pipeline() -> A.Compose:
    # GaussNoise 파라미터가 버전마다 달라(var_limit→std_range) 안전하게 분기
    try:
        gauss = A.GaussNoise(std_range=(0.05, 0.2), p=0.3)          # 신버전(>=1.4)
    except TypeError:
        gauss = A.GaussNoise(var_limit=(10.0, 50.0), p=0.3)         # 구버전

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.5),
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.4),
        gauss,
        A.CLAHE(clip_limit=2.0, p=0.3),
    ], bbox_params=A.BboxParams(format="yolo",
                                label_fields=["class_labels"],
                                min_visibility=0.3))


def augment_image(img, bboxes, class_labels, pipeline):
    img_uint8 = (img * 255).astype(np.uint8)
    result    = pipeline(image=img_uint8, bboxes=bboxes,
                         class_labels=class_labels)
    aug_img   = result["image"].astype(np.float32) / 255.0
    return aug_img, result["bboxes"], result["class_labels"]


# ─────────────────────────────────────────
# 5. 라벨 저장
# ─────────────────────────────────────────
def save_yolo_label(label_path, bboxes, class_labels):
    with open(label_path, "w") as f:
        for cls, (cx, cy, bw, bh) in zip(class_labels, bboxes):
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


# ─────────────────────────────────────────
# 6. 전체 파이프라인
# ─────────────────────────────────────────
def split_and_save(image_dir, bbox_dir, output_dir, augment=True):
    image_paths = sorted(Path(image_dir).glob("*.png")) + \
                  sorted(Path(image_dir).glob("*.jpg"))

    if not image_paths:
        print(f"⚠️  이미지를 찾을 수 없습니다: {image_dir}")
        return

    print(f"📂 총 이미지 수: {len(image_paths)}")

    train_p, temp_p = train_test_split(
        image_paths, test_size=(VAL_RATIO + TEST_RATIO), random_state=42)
    val_p, test_p = train_test_split(
        temp_p, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=42)

    splits       = {"train": train_p, "val": val_p, "test": test_p}
    aug_pipeline = get_augmentation_pipeline()
    skip_count   = 0

    for split, paths in splits.items():
        print(f"\n🔄 {split} 처리 중 ({len(paths)}장)...")
        for img_path in paths:
            stem = img_path.stem

            raw = cv2.imread(str(img_path))
            if raw is None:
                print(f"  ⛔ 읽기 실패: {img_path.name}")
                skip_count += 1
                continue
            orig_h, orig_w = raw.shape[:2]

            # 충치 bbox → YOLO
            bbox_path = Path(bbox_dir) / f"{stem}.txt"
            bboxes, class_labels = parse_caries_bbox(
                str(bbox_path), orig_w, orig_h)

            # 이미지 전처리
            img, scale, (pl, pt) = preprocess_image(str(img_path))

            # bbox letterbox 보정
            if bboxes:
                bboxes = adjust_bboxes_for_letterbox(
                    bboxes, orig_w, orig_h, scale, pl, pt, IMG_SIZE)

            # 원본 저장
            img_save = (img * 255).astype(np.uint8)
            cv2.imwrite(f"{output_dir}/{split}/images/{stem}.jpg", img_save)
            save_yolo_label(f"{output_dir}/{split}/labels/{stem}.txt",
                            bboxes, class_labels)

            # 학습셋 증강 (여러 장)
            if split == "train" and augment and bboxes:
                for i in range(AUG_PER_IMG):
                    try:
                        a_img, a_bb, a_cls = augment_image(
                            img, bboxes, class_labels, aug_pipeline)
                        if not a_bb:          # 증강 후 bbox 사라지면 skip
                            continue
                        a_save = (a_img * 255).astype(np.uint8)
                        cv2.imwrite(
                            f"{output_dir}/{split}/images/{stem}_aug{i}.jpg",
                            a_save)
                        save_yolo_label(
                            f"{output_dir}/{split}/labels/{stem}_aug{i}.txt",
                            a_bb, a_cls)
                    except Exception as e:
                        print(f"  ⚠️  증강 실패 ({stem}_aug{i}): {e}")

        print(f"  ✅ {split} 완료")

    if skip_count:
        print(f"\n⚠️  총 {skip_count}장 건너뜀")


# ─────────────────────────────────────────
# 7. data.yaml 생성
# ─────────────────────────────────────────
def create_yaml(output_dir, class_names):
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
# 8. 통계 출력
# ─────────────────────────────────────────
def print_dataset_stats(output_dir):
    print("\n📊 데이터셋 통계")
    print("─" * 30)
    total = 0
    for split in ["train", "val", "test"]:
        img_dir = Path(output_dir) / split / "images"
        count   = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        total  += count
        print(f"  {split:5s}: {count:4d}장")
    print(f"  {'합계':5s}: {total:4d}장 (증강 포함)")
    print("─" * 30)


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────
if __name__ == "__main__":
    CLASS_NAMES = ["cavity"]

    print("=" * 45)
    print("  Dental X-Ray 데이터 전처리 시작")
    print("  (bboxes_caries → YOLO 변환)")
    print("=" * 45)
    print(f"\n📁 이미지: {RAW_IMAGE_DIR}")
    print(f"📁 bbox  : {RAW_BBOX_DIR}\n")

    create_directories(OUTPUT_DIR)
    split_and_save(RAW_IMAGE_DIR, RAW_BBOX_DIR, OUTPUT_DIR, augment=True)
    create_yaml(OUTPUT_DIR, CLASS_NAMES)
    print_dataset_stats(OUTPUT_DIR)

    print("\n🎉 전처리 완료! 다음 단계: python dental_train.py")
