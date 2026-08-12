import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image

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
        background: linear-gradient(
            135deg,
            #e8f7ed,
            #ffffff
        );
        padding: 28px;
        border-radius: 18px;
        border: 2px solid #a7d8b6;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        text-align: center;
        margin-top: 20px;
    }

    .prediction-label {
        font-size: 16px;
        color: #4b6354;
        margin-bottom: 8px;
    }

    .prediction-name {
        font-size: 32px;
        font-weight: 800;
        color: #087432;
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

MODEL_PATH = "best_model.keras"

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

    model = tf.keras.models.load_model(MODEL_PATH)

    return model


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
                width=800
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

                st.markdown(f"""
                <div class="prediction-card">

                    <div class="prediction-label">
                    🌽 Model Prediction
                    </div>

                    <div class="prediction-name">
                    {disease_info["emoji"]}
                    {predicted_class}
                    </div>

                </div>
                """, unsafe_allow_html=True)


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
    # UAS Logo
    # --------------------------------------------------------

    try:

        st.image(
            "uas_logo.png",
            width=180
        )

    except:

        pass


    st.markdown("""
    <div class="hero">

        <h1>University of Agricultural Sciences, Bangalore</h1>

        <p>
        Artificial intelligence-assisted maize disease
        recognition research platform.
        </p>

    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # Campus image
    # --------------------------------------------------------

    try:

        st.image(
            "uas_campus.jpg",
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
