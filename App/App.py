import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Agricultural Pest Classifier",
    page_icon="🐛",
    layout="wide"
)

st.title("🐛 Agricultural Pest Classification System")
st.markdown("### AI-Powered Pest Detection for Agriculture")
st.markdown("Upload an image to identify common agricultural pests")
st.markdown("---")

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "pest_model.h5")  # H5 file, same folder as app.py

        model = keras.models.load_model(MODEL_PATH, compile=False)
        return model

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.error(f"📂 Tried path: {MODEL_PATH}")
        return None

# Load model once
with st.spinner("Loading model..."):
    model = load_model()

# =========================
# Classes & Info
# =========================
CLASS_NAMES = [
    'Ants','Bees','Beetle','Caterpillar','Earthworms','Earwig',
    'Grasshopper','Moth','Slug','Snail','Wasp','Weevil'
]

PEST_INFO = {
    'Ants': {'severity': 'Low', 'color': 'green', 'impact': 'May protect aphids that damage crops'},
    'Bees': {'severity': 'Beneficial', 'color': 'blue', 'impact': 'Important pollinators - should be protected'},
    'Beetle': {'severity': 'Medium-High', 'color': 'orange', 'impact': 'Can cause significant leaf and root damage'},
    'Caterpillar': {'severity': 'High', 'color': 'red', 'impact': 'Major defoliator, can destroy entire crops'},
    'Earthworms': {'severity': 'Beneficial', 'color': 'blue', 'impact': 'Improve soil health and structure'},
    'Earwig': {'severity': 'Low-Medium', 'color': 'orange', 'impact': 'Feeds on plants and other insects'},
    'Grasshopper': {'severity': 'High', 'color': 'red', 'impact': 'Can cause extensive crop damage in swarms'},
    'Moth': {'severity': 'Medium-High', 'color': 'orange', 'impact': 'Larvae can damage fruits and leaves'},
    'Slug': {'severity': 'Medium', 'color': 'orange', 'impact': 'Feeds on leaves, stems, and fruits'},
    'Snail': {'severity': 'Medium', 'color': 'orange', 'impact': 'Causes damage to seedlings and soft tissues'},
    'Wasp': {'severity': 'Low-Beneficial', 'color': 'green', 'impact': 'Natural pest control, pollinators'},
    'Weevil': {'severity': 'High', 'color': 'red', 'impact': 'Damages grains, seeds, and stored products'}
}

# =========================
# Preprocessing
# =========================
def preprocess_image(image, target_size=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# =========================
# Prediction (multi-input fix)
# =========================
def predict(model, image):
    processed_image = preprocess_image(image)
    # Multi-input workaround: send same tensor twice
    predictions = model.predict([processed_image, processed_image], verbose=0)
    return predictions[0]

# =========================
# Plot Predictions
# =========================
def plot_predictions(predictions, top_k=5):
    top_indices = np.argsort(predictions)[::-1][:top_k]
    top_probs = predictions[top_indices]
    top_classes = [CLASS_NAMES[i] for i in top_indices]

    colors = []
    for idx in top_indices:
        severity = PEST_INFO[CLASS_NAMES[idx]]['severity']
        if 'High' in severity:
            colors.append('#ff4444')
        elif 'Medium' in severity:
            colors.append('#ffaa00')
        elif 'Beneficial' in severity:
            colors.append('#4444ff')
        else:
            colors.append('#44aa44')

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(top_classes)), top_probs, color=colors)
    ax.set_yticks(range(len(top_classes)))
    ax.set_yticklabels(top_classes)
    ax.set_xlabel('Confidence Score')
    ax.set_title('Top Pest Predictions')
    ax.set_xlim([0, 1])

    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        ax.text(prob + 0.02, i, f'{prob*100:.1f}%', va='center')

    plt.tight_layout()
    return fig

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload Pest Image", type=['jpg','jpeg','png','bmp'])
    show_probabilities = st.checkbox("Show All Predictions", value=True)
    if show_probabilities:
        top_k = st.slider("Number of Results", 1, len(CLASS_NAMES), 5)
    show_chart = st.checkbox("Show Chart", value=True)
    st.markdown("---")
    with st.expander("ℹ️ About This Model"):
        st.info("""
        **Dataset**: Agricultural Pests Image Dataset
        **Classes**: 12 types of common agricultural pests
        **Purpose**: Helps farmers identify pests for timely intervention
        """)
    st.markdown("---")
    st.success("💡 Upload an image to start classification")

# =========================
# Main content
# =========================
if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📸 Input Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        st.caption(f"Size: {image.size[0]} x {image.size[1]} pixels")
        st.caption(f"Format: {image.format}")

    with col2:
        st.subheader("🎯 Classification Results")
        if model is not None:
            if st.button("🔍 Classify Pest", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    predictions = predict(model, image)
                    top_pred_idx = np.argmax(predictions)
                    top_pred_prob = predictions[top_pred_idx]
                    top_class = CLASS_NAMES[top_pred_idx]

                    st.success("✅ Classification Complete!")
                    result_container = st.container()
                    with result_container:
                        pest_info = PEST_INFO[top_class]
                        severity_color = pest_info['color']
                        st.markdown(f"""
                        <div style='padding: 20px; background-color: #f0f2f6; border-radius: 10px; border-left: 5px solid {severity_color}'>
                            <h2 style='margin: 0; color: {severity_color};'>{top_class}</h2>
                            <p style='font-size: 24px; margin: 10px 0;'><strong>Confidence: {top_pred_prob*100:.2f}%</strong></p>
                            <p style='margin: 5px 0;'><strong>Severity:</strong> {pest_info['severity']}</p>
                            <p style='margin: 5px 0;'>{pest_info['impact']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(float(top_pred_prob))

                    if show_probabilities:
                        st.markdown("---")
                        st.markdown("### All Predictions")
                        sorted_indices = np.argsort(predictions)[::-1][:top_k]
                        for idx in sorted_indices:
                            class_name = CLASS_NAMES[idx]
                            prob = predictions[idx]
                            info = PEST_INFO[class_name]
                            with st.container():
                                col_name, col_prob = st.columns([2,1])
                                with col_name:
                                    st.markdown(f"**{class_name}** ({info['severity']})")
                                with col_prob:
                                    st.metric("Confidence", f"{prob*100:.2f}%")
                                st.progress(float(prob))
                                st.caption(info['impact'])

                    if show_chart:
                        st.markdown("---")
                        st.markdown("### Confidence Distribution")
                        fig = plot_predictions(predictions, top_k=top_k)
                        st.pyplot(fig)
        else:
            st.error("❌ Model not loaded. Please check the model path in the code.")
else:
    st.info("👈 Upload an image from the sidebar to begin pest classification")
    st.markdown("### 🐛 Detectable Pests (12 Classes)")
    col1, col2, col3 = st.columns(3)
    for i, pest in enumerate(CLASS_NAMES):
        info = PEST_INFO[pest]
        with [col1, col2, col3][i % 3]:
            st.markdown(f"""
            <div style='padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin: 5px 0;'>
                <strong>{pest}</strong><br>
                <span style='color: {info['color']}; font-size: 12px;'>{info['severity']}</span>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📋 How to Use")
    st.markdown("""
    1. **Upload Image**: Click the upload button in the sidebar
    2. **Classify**: Click the 'Classify Pest' button
    3. **Review Results**: See the identified pest and confidence score
    4. **Take Action**: Use the information to make pest management decisions

    ### 🎯 Tips for Best Results
    - Use clear, well-lit images
    - Capture the pest from a close distance
    - Ensure the pest is the main focus of the image
    - Avoid blurry or low-quality images
    """)

# =========================
# Meet Our Team & Footer
# =========================
st.markdown("## 👥 Meet Our Team")
st.markdown("""
    - Abdelrahman Saeed
    - Hossam Eldin Mahmod
    - Mohamed Tareq Farouq
    - Youssef Ibrahim
    """, unsafe_allow_html=True)

st.markdown("### Supervised By")
st.markdown("**Eng. Sara Helal**")
st.markdown("---")
