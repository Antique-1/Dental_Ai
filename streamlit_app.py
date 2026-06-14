"""
Dental X-Ray 충치 탐지 웹 서비스
프로젝트: Dental Data Analysis
작성자: 김도현
실행: streamlit run streamlit_app.py
"""

import os
import yaml
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path
from ultralytics import YOLO


# ─────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Dental AI - 충치 탐지 시스템",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# 상수
# ─────────────────────────────────────────
BEST_MODEL_YAML = "runs/train/best_model.yaml"
FALLBACK_WEIGHT = "runs/train/yolov8n/weights/best.pt"
CONF_THRESHOLD  = 0.25
CLASS_NAMES     = {0: "충치(Cavity)"}

RISK_THRESHOLDS = {
    "high"   : 3,   # 탐지 개수 3개 이상 → 고위험
    "medium" : 1,   # 1~2개 → 중위험
}

CARE_ADVICE = {
    "high": {
        "label" : "🔴 고위험",
        "color" : "#FF4B4B",
        "bg"    : "#FFF0F0",
        "tips"  : [
            "즉시 치과 방문을 권장합니다.",
            "당분 및 산성 음식 섭취를 최소화하세요.",
            "식후 30분 이내 양치질을 반드시 진행하세요.",
            "불소 함유 치약 및 구강 세정제를 사용하세요.",
            "치실 및 치간 칫솔로 치아 사이 관리를 강화하세요.",
        ],
    },
    "medium": {
        "label" : "🟡 중위험",
        "color" : "#FFA500",
        "bg"    : "#FFFBF0",
        "tips"  : [
            "3개월 이내 치과 정기 검진을 받으세요.",
            "당분 섭취를 줄이고 수분 섭취를 늘리세요.",
            "하루 2회 이상 올바른 방법으로 양치질하세요.",
            "불소 치약 사용을 권장합니다.",
        ],
    },
    "low": {
        "label" : "🟢 정상/저위험",
        "color" : "#00C851",
        "bg"    : "#F0FFF4",
        "tips"  : [
            "현재 구강 상태가 양호합니다.",
            "6개월마다 정기 검진을 유지하세요.",
            "하루 2회 양치 및 치실 사용 습관을 유지하세요.",
            "균형 잡힌 식단으로 치아 건강을 유지하세요.",
        ],
    },
}


# ─────────────────────────────────────────
# 모델 로드 (캐싱)
# ─────────────────────────────────────────
@st.cache_resource
def load_model() -> YOLO:
    weight_path = FALLBACK_WEIGHT

    if Path(BEST_MODEL_YAML).exists():
        with open(BEST_MODEL_YAML) as f:
            info = yaml.safe_load(f)
        candidate = info.get("best_weights", "")
        if Path(candidate).exists():
            weight_path = candidate

    if not Path(weight_path).exists():
        st.error(f"모델 가중치를 찾을 수 없습니다: {weight_path}\n"
                 "dental_train.py 를 먼저 실행해 주세요.")
        st.stop()

    return YOLO(weight_path)


# ─────────────────────────────────────────
# 추론 전처리 (★학습 때와 동일하게 CLAHE 적용★)
# ─────────────────────────────────────────
def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """
    학습 데이터와 동일한 전처리를 추론 입력에도 적용한다.
    (전처리 단계에서 grayscale→CLAHE→3채널 변환을 거쳤으므로
     추론 입력도 같은 분포로 맞춰야 탐지 성능이 유지된다.)
    """
    gray     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


# ─────────────────────────────────────────
# 추론
# ─────────────────────────────────────────
def run_inference(model: YOLO, img_array: np.ndarray,
                  conf: float = CONF_THRESHOLD):
    """YOLOv8 추론 → (결과 이미지, 탐지 리스트)"""
    # 학습 때와 동일한 CLAHE 전처리 적용
    processed = apply_clahe(img_array)

    # imgsz=640 으로 명시 (학습 입력 크기와 일치)
    results = model.predict(processed, conf=conf, imgsz=640, verbose=False)
    result  = results[0]

    detections = []
    for box in result.boxes:
        detections.append({
            "class_id"  : int(box.cls),
            "class_name": CLASS_NAMES.get(int(box.cls), f"class_{int(box.cls)}"),
            "confidence": float(box.conf),
            "bbox"      : box.xyxy[0].tolist(),
        })

    annotated = result.plot()   # BGR numpy array
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb, detections


# ─────────────────────────────────────────
# 위험도 판정
# ─────────────────────────────────────────
def assess_risk(detections: list) -> str:
    n = len(detections)
    if n >= RISK_THRESHOLDS["high"]:
        return "high"
    elif n >= RISK_THRESHOLDS["medium"]:
        return "medium"
    return "low"


# ─────────────────────────────────────────
# UI 컴포넌트
# ─────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/emoji/96/tooth-emoji.png", width=80)
        st.title("🦷 Dental AI")
        st.markdown("**치과 X-Ray 충치 탐지 시스템**")
        st.divider()

        st.subheader("⚙️ 탐지 설정")
        conf = st.slider("신뢰도 임계값 (Confidence)",
                         min_value=0.10, max_value=0.90,
                         value=CONF_THRESHOLD, step=0.05,
                         help="낮을수록 더 많은 탐지, 높을수록 정밀 탐지")

        st.divider()
        st.subheader("ℹ️ 사용 안내")
        st.markdown("""
1. 파노라마 치과 X-Ray 이미지 업로드
2. **분석 시작** 버튼 클릭
3. 탐지 결과 및 관리 방법 확인
        """)
        st.divider()
        st.caption("⚠️ 본 시스템은 AI 보조 진단 도구입니다.\n정확한 진단은 치과 전문의와 상담하세요.")

    return conf


def render_detection_results(original_img, result_img,
                              detections: list, risk_level: str):
    advice = CARE_ADVICE[risk_level]

    # 이미지 비교
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 원본 이미지")
        st.image(original_img, use_container_width=True)
    with col2:
        st.subheader("🔍 탐지 결과")
        st.image(result_img, use_container_width=True)

    st.divider()

    # 위험도 배지
    st.markdown(
        f"""
        <div style="background:{advice['bg']};border-left:5px solid {advice['color']};
                    padding:16px;border-radius:8px;margin-bottom:16px;">
            <h3 style="color:{advice['color']};margin:0;">
                {advice['label']} &nbsp;|&nbsp; 탐지된 충치: {len(detections)}개
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 탐지 상세 + 관리 방법
    col_det, col_care = st.columns([1, 1])

    with col_det:
        st.subheader("📋 탐지 상세")
        if detections:
            for i, d in enumerate(detections, 1):
                st.markdown(
                    f"**{i}. {d['class_name']}** — "
                    f"신뢰도: `{d['confidence']*100:.1f}%`"
                )
        else:
            st.success("탐지된 충치가 없습니다.")

    with col_care:
        st.subheader("💊 사후관리 권장사항")
        for tip in advice["tips"]:
            st.markdown(f"- {tip}")

    # 수치 요약 메트릭
    st.divider()
    st.subheader("📊 분석 요약")
    m1, m2, m3 = st.columns(3)
    avg_conf = (sum(d["confidence"] for d in detections) / len(detections)
                if detections else 0)
    m1.metric("탐지된 충치 수",  f"{len(detections)}개")
    m2.metric("평균 신뢰도",     f"{avg_conf*100:.1f}%")
    m3.metric("위험도",          advice["label"])


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────
def main():
    conf_threshold = render_sidebar()

    st.title("🦷 Dental AI - 충치 자동 탐지 시스템")
    st.markdown("파노라마 치과 X-Ray 이미지를 업로드하면 AI가 충치 부위를 자동으로 탐지하고 관리 방법을 안내합니다.")
    st.divider()

    model = load_model()

    uploaded = st.file_uploader(
        "X-Ray 이미지 업로드 (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        help="파노라마 치과 X-Ray 이미지를 업로드하세요.",
    )

    if uploaded:
        pil_img    = Image.open(uploaded).convert("RGB")
        img_array  = np.array(pil_img)

        st.success(f"✅ 이미지 로드 완료 — {pil_img.width} × {pil_img.height} px")

        if st.button("🔍 분석 시작", type="primary", use_container_width=True):
            with st.spinner("AI 분석 중..."):
                result_img, detections = run_inference(
                    model, img_array, conf=conf_threshold)
                risk_level = assess_risk(detections)

            render_detection_results(pil_img, result_img,
                                     detections, risk_level)
    else:
        st.info("👆 위에서 X-Ray 이미지를 업로드해 주세요.")


if __name__ == "__main__":
    main()
