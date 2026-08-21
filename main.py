from pathlib import Path
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Maize Disease Recognition | UAS Bangalore",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f7faf7;
    }

    /* Main content */
    .main {
        padding-top: 1rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0b4d2c 0%,
            #063b22 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Sidebar title */
    .sidebar-title {
        text-align: center;
        font-size: 25px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sidebar-subtitle {
        text-align: center;
        font-size: 13px;
        opacity: 0.85;
        margin-bottom: 20px;
    }

    /* Hero */
    .hero {
        background: linear-gradient(
            135deg,
            #0b5d32 0%,
            #178447 50%,
            #2c9c55 100%
        );
        padding: 35px;
        border-radius: 20px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }

    .hero h1 {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero p {
        font-size: 18px;
        line-height: 1.6;
        margin-bottom: 0;
    }

    /* Cards */
    .info-card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e2e8e2;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
        margin-bottom: 18px;
    }

    .info-card h3 {
        color: #0b5d32;
        margin-top: 0;
    }

    /* Disease card */
    .disease-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #168345;
        box-shadow: 0 3px 12px rgba(0,0,0,0.06);
        margin-bottom: 15px;
    }

    .disease-card h3 {
        color: #0b5d32;
        margin-top: 0;
    }

    /* Result */
    .prediction-card {
        background: linear-gradient(145deg, #f4fff7 0%, #ffffff 55%, #eefaf2 100%);
        padding: 30px 24px 26px;
        border-radius: 22px;
        border: 1.5px solid #9ed8af;
        box-shadow: 0 10px 30px rgba(18, 100, 52, 0.12);
        text-align: center;
        margin: 22px 0 18px;
        position: relative;
        overflow: hidden;
    }

    .prediction-card::before {
        content: "";
        display: block;
        width: 72px;
        height: 5px;
        border-radius: 10px;
        background: #168345;
        margin: 0 auto 18px;
    }

    .prediction-label {
        font-size: 14px;
        font-weight: 800;
        letter-spacing: 1.8px;
        color: #557064;
        margin-bottom: 10px;
        text-transform: uppercase;
    }

    .prediction-icon {
        font-size: 42px;
        line-height: 1.1;
        margin: 4px 0 8px;
    }

    .prediction-name {
        font-size: 30px;
        line-height: 1.25;
        font-weight: 800;
        color: #086b2e;
        margin: 0 auto 13px;
    }

    .prediction-status {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        background: #e2f6e8;
        color: #176b38;
        font-size: 13px;
        font-weight: 650;
    }

    /* Section title */
    .section-title {
        color: #0b5d32;
        font-weight: 750;
        font-size: 28px;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 25px 10px;
        margin-top: 40px;
        border-top: 1px solid #dce6de;
        color: #64756a;
        font-size: 13px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL SETTINGS
# ============================================================

# Resolve the model relative to this main.py file so the app works
# correctly on Streamlit Cloud/GitHub regardless of the working directory.
BASE_DIR = Path(__file__).resolve().parent

# The deployment will automatically locate a Keras model in the
# same repository as this application. Explicit preferred names are
# checked first, followed by any other .keras files.
PREFERRED_MODEL_NAMES = [
    "best_model_selected.keras",
    "best_model.keras",
]

def find_model_file():
    # 1. Check preferred filenames first.
    for name in PREFERRED_MODEL_NAMES:
        path = BASE_DIR / name
        if path.is_file():
            return path

    # 2. Search the repository recursively for .keras files.
    candidates = sorted(
        [p for p in BASE_DIR.rglob("*.keras") if p.is_file()],
        key=lambda p: (p.name.lower(), str(p).lower())
    )

    if not candidates:
        return None

    # If there is exactly one Keras model, use it automatically.
    if len(candidates) == 1:
        return candidates[0]

    # Otherwise prefer names containing "best" or "selected".
    preferred = [
        p for p in candidates
        if "best" in p.name.lower() or "selected" in p.name.lower()
    ]

    if preferred:
        return preferred[0]

    # Final fallback: first discovered .keras model.
    return candidates[0]

IMAGE_SIZE = (224, 224)

# IMPORTANT:
# This order MUST match the class-index order used during training.
CLASS_NAMES = [
    "Common Rust",
    "Fusarium Stalk Rot",
    "Healthy Maize",
    "Maize Downy Mildew",
    "Southern Corn Blight",
    "Turcicum Leaf Blight"
]


# ============================================================
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    "Common Rust": {
        "emoji": "🟤",
        "description":
            "A foliar disease characterized by rust-coloured pustules "
            "on maize leaves. Severe infections can reduce photosynthetic "
            "activity and plant productivity.",
        "management": [
            "Use resistant or tolerant maize hybrids where available.",
            "Monitor fields regularly for early symptoms.",
            "Maintain appropriate plant density and crop aeration.",
            "Manage infected crop residues appropriately.",
            "Use recommended fungicides when disease pressure warrants chemical control."
        ]
    },

    "Fusarium Stalk Rot": {
        "emoji": "🟠",
        "description":
            "A stalk disease that can weaken maize plants and may lead "
            "to stalk lodging, particularly when plants are under stress.",
        "management": [
            "Use resistant or tolerant hybrids where available.",
            "Maintain balanced plant nutrition.",
            "Avoid excessive nitrogen application.",
            "Minimize drought and other environmental stresses.",
            "Manage insect damage that can provide infection sites.",
            "Remove or properly manage infected crop residues."
        ]
    },

    "Healthy Maize": {
        "emoji": "🌱",
        "description":
            "The image does not show prominent visual symptoms corresponding "
            "to the disease classes included in the present classification system.",
        "management": [
            "Use quality and disease-free seed.",
            "Maintain balanced crop nutrition.",
            "Ensure proper irrigation and drainage.",
            "Monitor the crop regularly for disease symptoms.",
            "Maintain good field sanitation.",
            "Follow recommended integrated crop management practices."
        ]
    },

    "Maize Downy Mildew": {
        "emoji": "🟡",
        "description":
            "A disease affecting maize foliage that can cause characteristic "
            "leaf symptoms and may reduce plant growth and productivity.",
        "management": [
            "Use certified, disease-free seed.",
            "Prefer resistant or tolerant hybrids.",
            "Avoid continuous maize cultivation where practical.",
            "Maintain appropriate field drainage.",
            "Remove severely infected plants where recommended.",
            "Follow locally recommended seed-treatment practices."
        ]
    },

    "Southern Corn Blight": {
        "emoji": "🔴",
        "description":
            "A foliar disease capable of producing characteristic leaf "
            "lesions and reducing effective photosynthetic area.",
        "management": [
            "Use resistant or tolerant maize hybrids.",
            "Practice crop rotation with suitable non-host crops.",
            "Manage infected crop residues.",
            "Maintain balanced plant nutrition.",
            "Avoid excessive crop density where possible.",
            "Use recommended fungicides when economically justified."
        ]
    },

    "Turcicum Leaf Blight": {
        "emoji": "🟣",
        "description":
            "A foliar blight of maize associated with elongated leaf lesions. "
            "Severe disease can reduce green leaf area and crop productivity.",
        "management": [
            "Use resistant or tolerant hybrids.",
            "Practice crop rotation.",
            "Manage infected crop residues.",
            "Maintain balanced fertilization.",
            "Regularly inspect lower leaves for early symptoms.",
            "Apply recommended fungicides when disease pressure is high."
        ]
    }
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    # Always resolve the model relative to this main.py file.
    # This avoids Streamlit Cloud working-directory/path issues.
    model_path = BASE_DIR / "best_model.keras"

    if not model_path.is_file():
        # Also check the preferred model name used by the original app.
        preferred_path = BASE_DIR / "best_model_selected.keras"

        if preferred_path.is_file():
            model_path = preferred_path
        else:
            available_models = sorted(
                str(p.relative_to(BASE_DIR))
                for p in BASE_DIR.rglob("*.keras")
                if p.is_file()
            )

            raise FileNotFoundError(
                f"Model file was not found in the application directory: "
                f"{BASE_DIR}. Expected: {BASE_DIR / 'best_model.keras'}. "
                f"Keras files found: {available_models}"
            )

    st.info(f"Loaded model: {model_path.name}")

    # A valid .keras model is a ZIP-based Keras archive.
    if model_path.stat().st_size < 1024:
        raise ValueError(
            f"The model file '{model_path.name}' is only "
            f"{model_path.stat().st_size} bytes and appears incomplete. "
            "Upload the complete .keras model file to GitHub."
        )

    import zipfile

    if not zipfile.is_zipfile(model_path):
        raise ValueError(
            f"'{model_path.name}' is not a valid .keras model archive. "
            "Make sure GitHub contains the complete model file and not "
            "a Git LFS pointer or incomplete upload."
        )

    try:
        # Prediction-only deployment does not need the training optimizer.
        model = tf.keras.models.load_model(
            str(model_path),
            compile=False
        )
    except Exception as exc:
        raise RuntimeError(
            f"The model file was found at '{model_path}', but Keras could "
            f"not load it. File size: "
            f"{model_path.stat().st_size / (1024**2):.2f} MB. "
            f"Original error: {exc}"
        ) from exc

    return model


# ============================================================
# GRAD-CAM EXPLAINABILITY
# ============================================================

def _find_last_conv_layer(model):
    """Find the last 4-D convolution-like layer for Grad-CAM."""
    # Prefer convolution layers in the current model.
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                               tf.keras.layers.SeparableConv2D,
                               tf.keras.layers.DepthwiseConv2D)):
            return layer

        # Handle nested Functional/Sequential models.
        if isinstance(layer, (tf.keras.Model, tf.keras.Sequential)):
            try:
                nested_layer = _find_last_conv_layer(layer)
                if nested_layer is not None:
                    return nested_layer
            except Exception:
                pass

    # Fallback: inspect layers by output shape.
    for layer in reversed(model.layers):
        try:
            output_shape = layer.output_shape
            if isinstance(output_shape, tuple) and len(output_shape) == 4:
                return layer
        except Exception:
            pass

    return None


def generate_gradcam(test_image, predicted_index):
    """
    Generate a Grad-CAM heatmap for the predicted class and overlay it
    on the uploaded image. The existing prediction pipeline is unchanged.
    """
    model = load_model()

    # Locate the last convolutional feature layer.
    last_conv_layer = _find_last_conv_layer(model)
    if last_conv_layer is None:
        raise ValueError(
            "Grad-CAM could not find a suitable convolutional feature layer "
            "in the loaded model."
        )

    # Prepare the same image representation used for prediction.
    original_image = Image.open(test_image).convert("RGB")
    display_image = np.asarray(original_image)

    resized_image = original_image.resize(IMAGE_SIZE)
    input_arr = np.asarray(resized_image).astype(np.float32)
    input_tensor = tf.expand_dims(input_arr, axis=0)

    # Build a model that returns the selected convolutional activations
    # and the final classifier output.
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor, training=False)

        # Support both standard classification output and nested output.
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        class_channel = predictions[:, predicted_index]

    # Compute gradients of the predicted class with respect to the
    # convolutional feature maps.
    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        raise ValueError(
            "Grad-CAM gradients could not be calculated for the selected layer."
        )

    # Global-average-pool the gradients to obtain channel importance.
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

    # Weight the feature maps by the pooled gradients.
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads[tf.newaxis, tf.newaxis, :],
        axis=-1
    )

    # Keep only positive activations and normalize to 0-1.
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)

    if float(max_value) > 0:
        heatmap = heatmap / max_value

    heatmap = heatmap.numpy()

    # Resize heatmap to the uploaded image dimensions.
    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255)
    ).resize(
        (display_image.shape[1], display_image.shape[0]),
        Image.Resampling.BILINEAR
    )

    heatmap_array = np.asarray(heatmap_image).astype(np.float32) / 255.0

    # Create a matplotlib colormap overlay without changing the
    # original uploaded image.
    cmap = plt.get_cmap("jet")
    colored_heatmap = cmap(heatmap_array)[..., :3]

    overlay = (
        0.55 * display_image.astype(np.float32) / 255.0
        + 0.45 * colored_heatmap
    )
    overlay = np.clip(overlay, 0, 1)

    return display_image, heatmap, overlay, last_conv_layer.name


def display_gradcam(test_image, predicted_index):
    """Display the Grad-CAM explanation below the prediction results."""
    original, heatmap, overlay, layer_name = generate_gradcam(
        test_image,
        predicted_index
    )

    st.markdown("### 🔥 Grad-CAM Visualization")
    st.caption(
        "Highlighted regions show the image areas that contributed most "
        "strongly to the model's predicted class."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            original,
            caption="Original image",
            use_container_width=True
        )

    with col2:
        st.image(
            overlay,
            caption="Grad-CAM attention map",
            use_container_width=True
        )

    st.caption(f"Grad-CAM feature layer: {layer_name}")

# ============================================================
# MODEL PREDICTION
# ============================================================

def model_prediction(test_image):

    model = load_model()

    image = Image.open(test_image).convert("RGB")

    image = image.resize(IMAGE_SIZE)

    input_arr = np.asarray(image).astype(np.float32)

    # Convert single image to batch
    input_arr = np.expand_dims(input_arr, axis=0)

    prediction = model.predict(
        input_arr,
        verbose=0
    )

    # Handle models with softmax output
    probabilities = prediction[0]

    # If output is not normalized, apply softmax
    if not np.isclose(np.sum(probabilities), 1.0, atol=0.01):
        probabilities = tf.nn.softmax(probabilities).numpy()

    result_index = int(np.argmax(probabilities))

    confidence = float(probabilities[result_index]) * 100

    return result_index, probabilities, confidence


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("""
<div class="sidebar-title">
🌽 Maize AI
</div>

<div class="sidebar-subtitle">
Disease Recognition System<br>
UAS Bangalore
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔬 Disease Recognition",
        "📘 About",
        "🌱 Management Strategies"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
### 🌽 Supported Classes

✓ Common Rust  
✓ Fusarium Stalk Rot  
✓ Healthy Maize  
✓ Maize Downy Mildew  
✓ Southern Corn Blight  
✓ Turcicum Leaf Blight
""")

st.sidebar.markdown("---")

st.sidebar.caption(
    "AI-assisted maize disease recognition research platform"
)


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    
    # --------------------------------------------------------
    # Optional UAS Logo
    # --------------------------------------------------------

    col_logo1, col_logo2, col_logo3 = st.columns([1, 3, 1])

    with col_logo2:

        try:

            st.image(
                "coverimage.png",
                width=1000
            )

        except:

            pass
       


    st.markdown(
        '<div class="section-title">🌱 About the System</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">

    The <b>Maize Disease Recognition System</b> is an AI-assisted
    research platform designed to support rapid preliminary
    identification of maize diseases from plant images.

    The system uses a trained deep learning image-classification
    model to recognize visual patterns associated with different
    disease categories.

    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # How it works
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🔍 How It Works</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        <div class="info-card">

        <h3>📤 1. Upload</h3>

        Upload a clear image of a maize leaf or plant showing
        disease symptoms.

        </div>
        """, unsafe_allow_html=True)


    with col2:

        st.markdown("""
        <div class="info-card">

        <h3>🧠 2. AI Analysis</h3>

        The deep learning model analyzes visual features
        present in the uploaded image.

        </div>
        """, unsafe_allow_html=True)


    with col3:

        st.markdown("""
        <div class="info-card">

        <h3>📊 3. Prediction</h3>

        The system displays the predicted class and model
        confidence together with probability scores.

        </div>
        """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # Supported diseases
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🌽 Supported Maize Classes</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(3)

    for i, disease in enumerate(CLASS_NAMES):

        with cols[i % 3]:

            info = DISEASE_INFO[disease]

            st.markdown(f"""
            <div class="disease-card">

            <h3>
            {info["emoji"]} {disease}
            </h3>

            <p>
            {info["description"]}
            </p>

            </div>
            """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">⭐ Key Features</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">

    ✓ Image-based disease classification  
    ✓ Six maize disease/health categories  
    ✓ Rapid prediction  
    ✓ Model confidence estimation  
    ✓ Class probability visualization  
    ✓ User-friendly interface  
    ✓ Research-oriented platform  
    ✓ Designed for agricultural applications  

    </div>
    """, unsafe_allow_html=True)


    st.info(
        "💡 Use the 'Disease Recognition' page from the sidebar "
        "to upload an image and obtain a model prediction."
    )


# ============================================================
# DISEASE RECOGNITION PAGE
# ============================================================

elif page == "🔬 Disease Recognition":

    st.markdown(
        '<div class="section-title">🔬 Maize Disease Recognition</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a maize plant image for AI-assisted disease classification."
    )

    st.markdown("---")


    # --------------------------------------------------------
    # Upload
    # --------------------------------------------------------

    test_image = st.file_uploader(
        "📤 Upload Maize Image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a maize leaf or plant."
    )


    if test_image is not None:

        # ----------------------------------------------------
        # Image and prediction columns
        # ----------------------------------------------------

        col1, col2 = st.columns([1, 1])


        with col1:

            st.markdown("### 🖼️ Uploaded Image")

            image = Image.open(test_image)

            st.image(
                image,
                caption="Uploaded maize image",
                use_container_width=True
            )


        with col2:

            st.markdown("### 🔍 Analysis")

            st.write(
                "Click the button below to analyze the uploaded image."
            )

            predict_button = st.button(
                "🧠 Analyze Image",
                use_container_width=True,
                type="primary"
            )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        if predict_button:

            try:

                with st.spinner(
                    "Analyzing the maize image..."
                ):

                    result_index, probabilities, confidence = (
                        model_prediction(test_image)
                    )


                predicted_class = CLASS_NAMES[result_index]

                disease_info = DISEASE_INFO[predicted_class]


                # ------------------------------------------------
                # Prediction Result
                # ------------------------------------------------

                st.markdown(f"""<div class="prediction-card">
<div class="prediction-label">🌽 &nbsp; MODEL PREDICTION</div>
<div class="prediction-icon">{disease_info["emoji"]}</div>
<div class="prediction-name">{predicted_class}</div>
<div class="prediction-status">✓ &nbsp; Disease identified successfully</div>
</div>""", unsafe_allow_html=True)


                # ------------------------------------------------
                # Confidence
                # ------------------------------------------------

                st.markdown("### 📈 Prediction Confidence")

                st.progress(
                    min(confidence / 100, 1.0)
                )

                st.metric(
                    "Model confidence",
                    f"{confidence:.2f}%"
                )


                # ------------------------------------------------
                # Interpretation
                # ------------------------------------------------

                if confidence >= 80:

                    st.success(
                        "High model confidence. The image contains "
                        "features strongly associated with the predicted class."
                    )

                elif confidence >= 60:

                    st.warning(
                        "Moderate model confidence. Consider confirming "
                        "the prediction through expert field diagnosis."
                    )

                else:

                    st.warning(
                        "Low model confidence. The image may contain "
                        "ambiguous symptoms or conditions outside the "
                        "training distribution."
                    )


                # ------------------------------------------------
                # Disease description
                # ------------------------------------------------

                st.markdown("### 📝 About the Prediction")

                st.info(
                    disease_info["description"]
                )


                # ------------------------------------------------
                # Probability chart
                # ------------------------------------------------

                st.markdown("### 📊 Class Probability Distribution")


                probability_df = pd.DataFrame({

                    "Disease": CLASS_NAMES,

                    "Probability": (
                        probabilities * 100
                    )

                })

                probability_df = probability_df.sort_values(
                    "Probability",
                    ascending=True
                )


                fig = px.bar(

                    probability_df,

                    x="Probability",

                    y="Disease",

                    orientation="h",

                    text="Probability",

                    labels={
                        "Probability": "Probability (%)",
                        "Disease": ""
                    }
                )


                fig.update_traces(

                    texttemplate="%{text:.2f}%",

                    textposition="outside"
                )


                fig.update_layout(

                    height=430,

                    margin=dict(
                        l=10,
                        r=40,
                        t=20,
                        b=20
                    ),

                    plot_bgcolor="white",

                    paper_bgcolor="white",

                    xaxis=dict(
                        range=[
                            0,
                            max(
                                100,
                                float(
                                    probability_df[
                                        "Probability"
                                    ].max()
                                ) * 1.15
                            )
                        ]
                    )
                )


                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


                # ------------------------------------------------
                # Grad-CAM visualization
                # ------------------------------------------------

                try:
                    display_gradcam(
                        test_image,
                        result_index
                    )
                except Exception as gradcam_error:
                    st.warning(
                        "Grad-CAM visualization could not be generated "
                        "for this model."
                    )
                    st.caption(str(gradcam_error))


                # ------------------------------------------------
                # Recommended management
                # ------------------------------------------------

                st.markdown(
                    "### 🌱 General Management Considerations"
                )

                for item in disease_info["management"]:

                    st.markdown(
                        f"✓ {item}"
                    )


                st.caption(
                    "Model confidence represents the model's output "
                    "probability and should not be interpreted as "
                    "definitive field diagnosis."
                )


            except Exception as e:

                st.error(
                    "Prediction could not be completed."
                )

                st.exception(e)


    else:

        st.info(
            "👆 Upload a maize image above to begin."
        )


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "📘 About":

    st.markdown(
        '<div class="section-title">📘 About the Project</div>',
        unsafe_allow_html=True
    )


     


    # --------------------------------------------------------
    # Campus image
    # --------------------------------------------------------

    try:

        st.image(
            "complete deployment.png",
            caption="University of Agricultural Sciences, Bangalore – GKVK Campus",
            use_container_width=True
        )

    except:

        st.info(
            "Add the UAS campus image as 'uas_campus.jpg' "
            "to display it here."
        )


    # --------------------------------------------------------
    # Project overview
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🌽 Project Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">

    The <b>Maize Disease Recognition System</b> is being developed
    as an AI-assisted platform for image-based identification of
    important maize diseases.

    The system combines computer vision, deep learning and an
    interactive web interface to provide rapid preliminary
    disease classification from maize plant images.

    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # Objectives
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🎯 Objectives</div>',
        unsafe_allow_html=True
    )

    objectives = [

        "Develop an automated maize disease recognition system.",

        "Apply deep learning for image-based disease classification.",

        "Provide a simple interface for researchers and agricultural users.",

        "Support rapid preliminary disease identification.",

        "Explore explainable and interpretable AI for plant disease diagnosis.",

        "Develop a research-oriented platform for future AI-based agricultural applications."
    ]


    for objective in objectives:

        st.markdown(
            f"✓ {objective}"
        )


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">📊 Dataset Information</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">

    The current maize image dataset contains six classes:

    <br><br>

    <b>Common Rust</b><br>
    <b>Fusarium Stalk Rot</b><br>
    <b>Healthy Maize</b><br>
    <b>Maize Downy Mildew</b><br>
    <b>Southern Corn Blight</b><br>
    <b>Turcicum Leaf Blight</b>

    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # Dataset numbers
    # --------------------------------------------------------

    dataset_data = {

        "Class": [
            "Common Rust",
            "Fusarium Stalk Rot",
            "Healthy Maize",
            "Maize Downy Mildew",
            "Southern Corn Blight",
            "Turcicum Leaf Blight"
        ],

        "Total Images": [
            303,
            703,
            3435,
            321,
            1549,
            8123
        ]
    }


    dataset_df = pd.DataFrame(dataset_data)


    st.dataframe(
        dataset_df,
        use_container_width=True,
        hide_index=True
    )


    st.caption(
        "Dataset counts represent the current image collection "
        "provided for the six maize classes."
    )


    # --------------------------------------------------------
    # Institutional background
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🏫 Institutional Background</div>',
        unsafe_allow_html=True
    )

    st.write("""
    The University of Agricultural Sciences, Bangalore (UAS-B)
    is an agricultural education and research institution in
    Karnataka.

    The GKVK campus provides an important environment for
    agricultural research, education, extension and technology
    development.
    """)


    # --------------------------------------------------------
    # Development team
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">👨‍🔬 Development Team</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">

    <b>Maruthi Prasad B. P.</b><br>
    Department of Genetics and Plant Breeding<br>
    University of Agricultural Sciences, Bangalore

    <br><br>

    <b>Harish J.</b><br>
    Department of Plant Pathology<br>
    University of Agricultural Sciences, Bangalore

    <br><br>

    <b>M. K. Prasannakumar</b><br>
    Department of Plant Pathology<br>
    University of Agricultural Sciences, Bangalore

    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # Disclaimer
    # --------------------------------------------------------

    st.warning("""
    ⚠️ **Research and decision-support disclaimer**

    This platform is intended for research, educational and
    preliminary disease-recognition purposes. AI-based image
    classification should not replace field diagnosis or
    expert assessment. Management decisions should consider
    local disease conditions, crop stage, cultivar and
    recommendations from qualified agricultural experts.
    """)


# ============================================================
# MANAGEMENT STRATEGIES PAGE
# ============================================================

elif page == "🌱 Management Strategies":

    st.markdown(
        '<div class="section-title">🌱 Maize Disease Management Strategies</div>',
        unsafe_allow_html=True
    )


    st.write("""
    Effective maize disease management requires an integrated
    approach involving resistant cultivars, healthy seed,
    field sanitation, balanced nutrition, crop rotation,
    regular monitoring and timely intervention.
    """)


    st.markdown("---")


    # --------------------------------------------------------
    # Common Rust
    # --------------------------------------------------------

    st.markdown("""
    <div class="disease-card">

    <h3>🟤 Common Rust</h3>

    <ul>
        <li>Use resistant or tolerant maize hybrids where available.</li>
        <li>Monitor fields regularly for early symptoms.</li>
        <li>Maintain appropriate plant density and crop aeration.</li>
        <li>Manage infected crop residues appropriately.</li>
        <li>Use recommended fungicides when disease pressure warrants chemical control.</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # Fusarium Stalk Rot
    # --------------------------------------------------------

    st.markdown("""
    <div class="disease-card">

    <h3>🟠 Fusarium Stalk Rot</h3>

    <ul>
        <li>Use resistant or tolerant maize hybrids where available.</li>
        <li>Maintain balanced plant nutrition.</li>
        <li>Avoid excessive nitrogen application.</li>
        <li>Minimize drought and other environmental stresses.</li>
        <li>Manage insect damage that can provide infection sites.</li>
        <li>Remove or properly manage infected crop residues.</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # Healthy maize
    # --------------------------------------------------------

    st.markdown("""
    <div class="disease-card">

    <h3>🌱 Healthy Maize</h3>

    <ul>
        <li>Use quality and disease-free seed.</li>
        <li>Maintain balanced crop nutrition.</li>
        <li>Ensure proper irrigation and drainage.</li>
        <li>Monitor the crop regularly for disease symptoms.</li>
        <li>Maintain good field sanitation.</li>
        <li>Follow recommended integrated crop management practices.</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # Downy Mildew
    # --------------------------------------------------------

    st.markdown("""
    <div class="disease-card">

    <h3>🟡 Maize Downy Mildew</h3>

    <ul>
        <li>Use certified, disease-free seed.</li>
        <li>Prefer resistant or tolerant hybrids.</li>
        <li>Avoid continuous maize cultivation where practical.</li>
        <li>Maintain appropriate field drainage.</li>
        <li>Remove severely infected plants where recommended.</li>
        <li>Follow locally recommended seed-treatment practices.</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # Southern Corn Blight
    # --------------------------------------------------------

    st.markdown("""
    <div class="disease-card">

    <h3>🔴 Southern Corn Blight</h3>

    <ul>
        <li>Use resistant or tolerant maize hybrids.</li>
        <li>Practice crop rotation with suitable non-host crops.</li>
        <li>Manage infected crop residues.</li>
        <li>Maintain balanced plant nutrition.</li>
        <li>Avoid excessive crop density where possible.</li>
        <li>Use recommended fungicides when economically justified.</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # Turcicum Leaf Blight
    # --------------------------------------------------------

    st.markdown("""
    <div class="disease-card">

    <h3>🟣 Turcicum Leaf Blight</h3>

    <ul>
        <li>Use resistant or tolerant maize hybrids.</li>
        <li>Practice crop rotation.</li>
        <li>Manage infected crop residues.</li>
        <li>Maintain balanced fertilization.</li>
        <li>Regularly inspect lower leaves for early symptoms.</li>
        <li>Apply recommended fungicides when disease pressure is high.</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # General recommendations
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🌾 General Recommendations</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">

    ✓ Use certified and healthy seed<br>
    ✓ Prefer disease-resistant cultivars<br>
    ✓ Practice crop rotation<br>
    ✓ Maintain field sanitation<br>
    ✓ Maintain balanced fertilization<br>
    ✓ Avoid unnecessary crop stress<br>
    ✓ Monitor fields regularly<br>
    ✓ Identify disease symptoms at an early stage<br>
    ✓ Use chemical control only when appropriate and according to local recommendations

    </div>
    """, unsafe_allow_html=True)


    st.warning("""
    ⚠️ Always follow locally approved agricultural recommendations
    and consult qualified agricultural experts before applying
    plant-protection products.
    """)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

🌽 <b>Maize Disease Recognition System</b><br>

Artificial Intelligence-assisted Agricultural Research Platform<br>

University of Agricultural Sciences, Bangalore

<br><br>

© 2026 | Research and Educational Use

</div>
""", unsafe_allow_html=True)
