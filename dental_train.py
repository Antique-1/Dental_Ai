"""
Dental X-Ray YOLOv8 모델 학습 코드
프로젝트: Dental Data Analysis
작성자: 김도현
"""

import os
import yaml
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from pathlib import Path
from ultralytics import YOLO


# ─────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────
DATA_YAML   = "data/processed/data.yaml"
OUTPUT_DIR  = "runs/train"
IMG_SIZE    = 640
EPOCHS      = 100
BATCH_SIZE  = 16
DEVICE      = "0" if torch.cuda.is_available() else "cpu"

MODELS = {
    "yolov8n": "yolov8n.pt",   # nano  - 경량
    "yolov8m": "yolov8m.pt",   # medium - 정확도 우선
}


# ─────────────────────────────────────────
# 1. 단일 모델 학습
# ─────────────────────────────────────────
def train_model(model_name: str, model_weight: str,
                data_yaml: str, output_dir: str) -> dict:
    """
    YOLOv8 모델 학습
    Returns: 학습 결과 metrics dict
    """
    print(f"\n{'='*45}")
    print(f"  모델 학습 시작: {model_name}")
    print(f"{'='*45}")

    model = YOLO(model_weight)

    results = model.train(
        data       = data_yaml,
        epochs     = EPOCHS,
        imgsz      = IMG_SIZE,
        batch      = BATCH_SIZE,
        device     = DEVICE,
        project    = output_dir,
        name       = model_name,
        patience   = 15,          # Early Stopping
        optimizer  = "AdamW",
        lr0        = 0.001,
        lrf        = 0.01,
        momentum   = 0.937,
        weight_decay = 0.0005,
        warmup_epochs = 3,
        cos_lr     = True,        # Cosine LR Scheduler
        augment    = True,
        val        = True,
        save       = True,
        plots      = True,
        verbose    = True,
    )

    # 최종 성능 지표 추출
    metrics = {
        "model"       : model_name,
        "mAP50"       : float(results.results_dict.get("metrics/mAP50(B)",   0)),
        "mAP50_95"    : float(results.results_dict.get("metrics/mAP50-95(B)",0)),
        "precision"   : float(results.results_dict.get("metrics/precision(B)",0)),
        "recall"      : float(results.results_dict.get("metrics/recall(B)",  0)),
        "best_weights": str(Path(output_dir) / model_name / "weights" / "best.pt"),
    }

    print(f"\n📊 [{model_name}] 학습 완료")
    print(f"   mAP50     : {metrics['mAP50']:.4f}")
    print(f"   mAP50-95  : {metrics['mAP50_95']:.4f}")
    print(f"   Precision : {metrics['precision']:.4f}")
    print(f"   Recall    : {metrics['recall']:.4f}")

    return metrics


# ─────────────────────────────────────────
# 2. 두 모델 비교 학습
# ─────────────────────────────────────────
def train_all_models(data_yaml: str, output_dir: str) -> list:
    """YOLOv8n / YOLOv8m 순차 학습 후 결과 반환"""
    all_metrics = []
    for name, weight in MODELS.items():
        metrics = train_model(name, weight, data_yaml, output_dir)
        all_metrics.append(metrics)
    return all_metrics


# ─────────────────────────────────────────
# 3. 성능 비교 시각화
# ─────────────────────────────────────────
def plot_comparison(all_metrics: list, save_dir: str):
    """두 모델 성능 지표 비교 막대 그래프"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    metric_keys   = ["mAP50", "mAP50_95", "precision", "recall"]
    metric_labels = ["mAP50", "mAP50-95", "Precision", "Recall"]
    model_names   = [m["model"] for m in all_metrics]
    colors        = ["#4C72B0", "#DD8452"]

    x = range(len(metric_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metrics in enumerate(all_metrics):
        vals = [metrics[k] for k in metric_keys]
        bars = ax.bar([p + i * width for p in x], vals, width,
                      label=metrics["model"], color=colors[i], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks([p + width / 2 for p in x])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("YOLOv8n vs YOLOv8m - Performance Comparison", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    save_path = os.path.join(save_dir, "model_comparison.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📈 비교 그래프 저장: {save_path}")


# ─────────────────────────────────────────
# 4. 최적 모델 선정 및 저장
# ─────────────────────────────────────────
def select_best_model(all_metrics: list, save_dir: str) -> dict:
    """mAP50 기준 최적 모델 선정 후 best_model.yaml 저장"""
    best = max(all_metrics, key=lambda m: m["mAP50"])

    result = {
        "best_model"   : best["model"],
        "best_weights" : best["best_weights"],
        "mAP50"        : best["mAP50"],
        "mAP50_95"     : best["mAP50_95"],
        "precision"    : best["precision"],
        "recall"       : best["recall"],
    }

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    yaml_path = os.path.join(save_dir, "best_model.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(result, f, default_flow_style=False)

    print(f"\n🏆 최적 모델: {best['model']}  (mAP50: {best['mAP50']:.4f})")
    print(f"   가중치 경로: {best['best_weights']}")
    print(f"   설정 저장  : {yaml_path}")
    return result


# ─────────────────────────────────────────
# 5. 테스트셋 평가
# ─────────────────────────────────────────
def evaluate_best_model(best_info: dict, data_yaml: str):
    """최적 모델로 테스트셋 최종 평가"""
    print(f"\n🔍 테스트셋 평가: {best_info['best_model']}")
    model = YOLO(best_info["best_weights"])
    metrics = model.val(data=data_yaml, split="test",
                        imgsz=IMG_SIZE, device=DEVICE)
    print(f"   테스트 mAP50   : {metrics.box.map50:.4f}")
    print(f"   테스트 mAP50-95: {metrics.box.map:.4f}")


# ─────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 45)
    print("  Dental X-Ray YOLOv8 모델 학습")
    print("=" * 45)
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {EPOCHS}")
    print(f"  Batch  : {BATCH_SIZE}")

    all_metrics = train_all_models(DATA_YAML, OUTPUT_DIR)
    plot_comparison(all_metrics, OUTPUT_DIR)
    best_info   = select_best_model(all_metrics, OUTPUT_DIR)
    evaluate_best_model(best_info, DATA_YAML)

    print("\n🎉 모든 학습 완료! 다음 단계: streamlit_app.py 실행")
