import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Agricultural Pest Classifier",
    page_icon="🐛",
    layout="wide"
)

# Title and description
st.title("🐛 Agricultural Pest Classification System")
st.markdown("### AI-Powered Pest Detection for Agriculture")
st.markdown("Upload an image to identify common agricultural pests")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    """
    Load your trained model here

    """
    try:
        # Update this path to your actual model
        model = keras.models.load_model('/content/pest_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.warning("⚠️ Please update the model path in the code")
        return None

# Pest class names - 12 classes from Agricultural Pests Dataset
CLASS_NAMES = [
    'Ants',
    'Bees',
    'Beetle',
    'Caterpillar',
    'Earthworms',
    'Earwig',
    'Grasshopper',
    'Moth',
    'Slug',
    'Snail',
    'Wasp',
    'Weevil'
]

# Pest descriptions and severity
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

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for model input
    Adjust target_size to match your model's expected input
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize image
    image = image.resize(target_size)

    # Convert to numpy array
    img_array = np.array(image)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize - adjust based on your training preprocessing
    # Common options:
    # Option 1: Scale to [0, 1]
    img_array = img_array / 255.0

    # Option 2: If you used ImageNet preprocessing
    # img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Option 3: Standard normalization
    # img_array = (img_array - 127.5) / 127.5

    return img_array

def predict(model, image):
    """
    Make prediction on the image
    """
    # Preprocess image
    processed_image = preprocess_image(image)

    # Get predictions
    predictions = model.predict(processed_image, verbose=0)

    return predictions[0]

def plot_predictions(predictions, top_k=5):
    """
    Create a bar chart of top predictions
    """
    # Get top k predictions
    top_indices = np.argsort(predictions)[::-1][:top_k]
    top_probs = predictions[top_indices]
    top_classes = [CLASS_NAMES[i] for i in top_indices]

    # Create color mapping based on severity
    colors = []
    for idx in top_indices:
        class_name = CLASS_NAMES[idx]
        severity = PEST_INFO[class_name]['severity']
        if 'High' in severity:
            colors.append('#ff4444')
        elif 'Medium' in severity:
            colors.append('#ffaa00')
        elif 'Beneficial' in severity:
            colors.append('#4444ff')
        else:
            colors.append('#44aa44')

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(top_classes)), top_probs, color=colors)
    ax.set_yticks(range(len(top_classes)))
    ax.set_yticklabels(top_classes)
    ax.set_xlabel('Confidence Score')
    ax.set_title('Top Pest Predictions')
    ax.set_xlim([0, 1])

    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        ax.text(prob + 0.02, i, f'{prob*100:.1f}%', va='center')

    plt.tight_layout()
    return fig

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Pest Image",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Supported formats: JPG, JPEG, PNG, BMP"
    )

    st.markdown("---")

    # Display options
    show_probabilities = st.checkbox("Show All Predictions", value=True)

    if show_probabilities:
        top_k = st.slider("Number of Results", 1, len(CLASS_NAMES), 5)

    show_chart = st.checkbox("Show Chart", value=True)

    st.markdown("---")

    # Model info
    with st.expander("ℹ️ About This Model"):
        st.info("""
        **Dataset**: Agricultural Pests Image Dataset

        **Classes**: 12 types of common agricultural pests

        **Purpose**: Helps farmers identify pests for timely intervention
        """)

    st.markdown("---")
    st.success("💡 Upload an image to start classification")

# Main content
if uploaded_file is not None:
    # Create columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📸 Input Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        # Image info
        st.caption(f"Size: {image.size[0]} x {image.size[1]} pixels")
        st.caption(f"Format: {image.format}")

    with col2:
        st.subheader("🎯 Classification Results")

        # Load model
        model = load_model()

        if model is not None:
            # Predict button
            if st.button("🔍 Classify Pest", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Make prediction
                    predictions = predict(model, image)

                    # Get top prediction
                    top_pred_idx = np.argmax(predictions)
                    top_pred_prob = predictions[top_pred_idx]
                    top_class = CLASS_NAMES[top_pred_idx]

                    # Display main result
                    st.success("✅ Classification Complete!")

                    # Create result container
                    result_container = st.container()

                    with result_container:
                        st.markdown("### Primary Detection")

                        # Display pest name with severity color
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

                        # Progress bar
                        st.progress(float(top_pred_prob))

                    # Show all predictions if enabled
                    if show_probabilities:
                        st.markdown("---")
                        st.markdown("### All Predictions")

                        # Sort predictions
                        sorted_indices = np.argsort(predictions)[::-1][:top_k]

                        for idx in sorted_indices:
                            class_name = CLASS_NAMES[idx]
                            prob = predictions[idx]
                            info = PEST_INFO[class_name]

                            with st.container():
                                col_name, col_prob = st.columns([2, 1])

                                with col_name:
                                    st.markdown(f"**{class_name}** ({info['severity']})")

                                with col_prob:
                                    st.metric("Confidence", f"{prob*100:.2f}%")

                                st.progress(float(prob))
                                st.caption(info['impact'])

                    # Show chart if enabled
                    if show_chart:
                        st.markdown("---")
                        st.markdown("### Confidence Distribution")
                        fig = plot_predictions(predictions, top_k=top_k)
                        st.pyplot(fig)
        else:
            st.error("❌ Model not loaded. Please check the model path in the code.")

else:
    # Welcome screen
    st.info("👈 Upload an image from the sidebar to begin pest classification")

    # Display pest classes
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

    # Instructions
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
# Meet Our Team Section
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



# Footer
st.markdown("---")