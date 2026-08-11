import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Maize Disease AI | UAS Bangalore",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ---------- MAIN BACKGROUND ---------- */

    .stApp {
        background:
        linear-gradient(
            180deg,
            #f7fbf5 0%,
            #ffffff 45%,
            #f4f9f2 100%
        );
    }


    /* ---------- SIDEBAR ---------- */

    [data-testid="stSidebar"] {
        background:
        linear-gradient(
            180deg,
            #123d27 0%,
            #1d5b37 55%,
            #0f3321 100%
        );
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }


    /* ---------- HEADINGS ---------- */

    h1 {
        color: #123d27;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    h2 {
        color: #174c2e;
        font-weight: 750;
    }

    h3 {
        color: #24623c;
        font-weight: 700;
    }


    /* ---------- HERO ---------- */

    .hero {
        padding: 45px 40px;
        border-radius: 24px;

        background:
        linear-gradient(
            135deg,
            rgba(18,61,39,0.96),
            rgba(39,104,60,0.88)
        );

        color: white;

        box-shadow:
        0 15px 35px rgba(0,0,0,0.12);

        margin-bottom: 30px;
    }

    .hero h1 {
        color: white;
        font-size: 46px;
        margin-bottom: 10px;
    }

    .hero p {
        font-size: 19px;
        line-height: 1.7;
        color: #eef8ef;
    }


    /* ---------- UAS HEADER ---------- */

    .uas-header {
        background: white;
        padding: 15px 25px;
        border-radius: 15px;

        box-shadow:
        0 4px 18px rgba(0,0,0,0.08);

        margin-bottom: 25px;

        border-left:
        6px solid #1d6b3b;
    }

    .uas-title {
        font-size: 23px;
        font-weight: 800;
        color: #174c2e;
    }

    .uas-subtitle {
        font-size: 14px;
        color: #65746b;
    }


    /* ---------- CARDS ---------- */

    .feature-card {
        background: white;

        padding: 28px 24px;

        border-radius: 18px;

        min-height: 190px;

        box-shadow:
        0 8px 25px rgba(0,0,0,0.07);

        border-top:
        4px solid #3b8f52;

        transition: 0.25s;
    }

    .feature-card:hover {
        transform: translateY(-4px);

        box-shadow:
        0 14px 30px rgba(0,0,0,0.12);
    }

    .feature-icon {
        font-size: 38px;
    }

    .feature-title {
        font-size: 20px;
        font-weight: 750;
        color: #174c2e;
        margin-top: 10px;
    }

    .feature-text {
        color: #65746b;
        line-height: 1.6;
    }


    /* ---------- RESULT CARD ---------- */

    .prediction-card {
        background:
        linear-gradient(
            135deg,
            #edf8ef,
            #ffffff
        );

        padding: 30px;

        border-radius: 20px;

        border:
        1px solid #cfe4d3;

        box-shadow:
        0 10px 30px rgba(0,0,0,0.08);
    }

    .prediction-label {
        color: #66806e;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .prediction-name {
        color: #174c2e;
        font-size: 31px;
        font-weight: 800;
    }

    .confidence {
        color: #277442;
        font-size: 21px;
        font-weight: 700;
    }


    /* ---------- INFO BOX ---------- */

    .info-box {
        background: #f1f8f3;

        padding: 22px;

        border-radius: 15px;

        border-left:
        5px solid #32834b;

        margin: 15px 0;
    }


    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;

        padding: 30px;

        margin-top: 50px;

        color: #6b776f;

        border-top:
        1px solid #dce8df;
    }


    /* ---------- BUTTON ---------- */

    .stButton > button {
        border-radius: 12px;

        border: none;

        background:
        linear-gradient(
            90deg,
            #1d6339,
            #32884d
        );

        color: white;

        font-weight: 700;

        padding: 10px 25px;

        min-height: 48px;
    }

    .stButton > button:hover {
        background:
        linear-gradient(
            90deg,
            #154b2c,
            #276d3f
        );

        color: white;
    }


    /* ---------- FILE UPLOADER ---------- */

    [data-testid="stFileUploader"] {
        background: white;

        padding: 15px;

        border-radius: 15px;

        border:
        2px dashed #9bc4a4;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "best_model.keras"

# Change this after your maize model is trained.
CLASS_NAMES = [
    "Maize Disease 1",
    "Maize Disease 2",
    "Maize Disease 3",
    "Maize Disease 4",
    "Maize Disease 5",
    "Maize Disease 6",
    "Maize Disease 7",
    "Maize Disease 8"
]


# ============================================================
# UAS HEADER
# ============================================================

st.markdown("""
<div class="uas-header">

    <div class="uas-title">
        🌱 UNIVERSITY OF AGRICULTURAL SCIENCES, BANGALORE
    </div>

    <div class="uas-subtitle">
        GKVK • Bengaluru • Karnataka, India
        &nbsp; | &nbsp;
        Artificial Intelligence in Agriculture
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_disease(uploaded_file):

    model = load_model()

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    prediction = model.predict(
        image_array,
        verbose=0
    )[0]

    index = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[index] * 100
    )

    return (
        index,
        confidence,
        prediction
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## 🌽 Maize Disease AI"
)

st.sidebar.markdown(
    "### UAS Bangalore"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "🔬 Disease Recognition",
        "📘 About",
        "🌱 Management Strategies"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "AI-assisted maize disease recognition"
)

st.sidebar.caption(
    "Research-oriented agricultural application"
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.markdown("""
    <div class="hero">

        <h1>🌽 Maize Disease Recognition</h1>

        <p>
        An AI-powered image analysis platform developed
        for rapid and accessible recognition of maize
        diseases from plant images.
        </p>

        <p>
        Developed in the academic and research environment
        of the University of Agricultural Sciences, Bangalore.
        </p>

    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # HERO IMAGE
    # --------------------------------------------------------

    try:

        st.image(
            "maize_field.jpg",
            use_container_width=True
        )

    except:

        st.info(
            "Add a high-quality maize field image "
            "as 'maize_field.jpg'."
        )


    st.markdown(
        "## 🌱 Intelligent Plant Disease Detection"
    )

    st.write(
        """
        The system uses deep learning-based image
        classification to identify disease-associated
        visual patterns in maize plant images.
        """
    )


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        <div class="feature-card">

            <div class="feature-icon">
                🧠
            </div>

            <div class="feature-title">
                AI-Powered
            </div>

            <div class="feature-text">
                Deep learning-based image classification
                for automated maize disease recognition.
            </div>

        </div>
        """, unsafe_allow_html=True)


    with c2:

        st.markdown("""
        <div class="feature-card">

            <div class="feature-icon">
                ⚡
            </div>

            <div class="feature-title">
                Rapid Analysis
            </div>

            <div class="feature-text">
                Upload a maize image and obtain a prediction
                within seconds.
            </div>

        </div>
        """, unsafe_allow_html=True)


    with c3:

        st.markdown("""
        <div class="feature-card">

            <div class="feature-icon">
                🔬
            </div>

            <div class="feature-title">
                Research Based
            </div>

            <div class="feature-text">
                Designed as an agricultural research and
                decision-support platform.
            </div>

        </div>
        """, unsafe_allow_html=True)


    st.markdown("---")


    st.markdown("## 🔍 How it works")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 01 · Upload")
        st.write(
            "Upload a clear maize leaf or plant image."
        )

    with c2:
        st.markdown("### 02 · Analyze")
        st.write(
            "The trained deep learning model analyzes "
            "visual disease patterns."
        )

    with c3:
        st.markdown("### 03 · Predict")
        st.write(
            "Receive the predicted disease and confidence."
        )


# ============================================================
# DISEASE RECOGNITION
# ============================================================

elif page == "🔬 Disease Recognition":

    st.header(
        "🔬 Maize Disease Recognition"
    )

    st.write(
        """
        Upload a maize plant image for AI-assisted
        disease classification.
        """
    )


    uploaded_file = st.file_uploader(
        "📤 Upload maize image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        help="Upload a clear image showing the maize plant or leaf."
    )


    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        col1, col2 = st.columns(
            [1.1, 0.9]
        )


        with col1:

            st.image(
                image,
                caption="Uploaded maize image",
                use_container_width=True
            )


        with col2:

            st.markdown("""
            <div class="info-box">

            <b>Image ready for analysis</b>

            <br><br>

            The image will be processed using
            the trained maize disease classification model.

            </div>
            """, unsafe_allow_html=True)


            analyze = st.button(
                "🔍 Analyze Image",
                use_container_width=True
            )


        if analyze:

            try:

                with st.spinner(
                    "🧠 Analyzing maize image..."
                ):

                    (
                        index,
                        confidence,
                        probabilities
                    ) = predict_disease(
                        uploaded_file
                    )


                predicted_class = CLASS_NAMES[
                    index
                ]


                st.markdown(
                    "## 📊 Prediction Result"
                )


                st.markdown(f"""
                <div class="prediction-card">

                    <div class="prediction-label">
                        Most probable disease
                    </div>

                    <div class="prediction-name">
                        🌽 {predicted_class}
                    </div>

                    <br>

                    <div class="confidence">
                        Confidence: {confidence:.2f}%
                    </div>

                </div>
                """, unsafe_allow_html=True)


                # ------------------------------------------------
                # PROBABILITY CHART
                # ------------------------------------------------

                st.markdown(
                    "### 📈 Prediction probabilities"
                )


                probability_df = pd.DataFrame({
                    "Disease": CLASS_NAMES,
                    "Probability":
                        probabilities * 100
                })


                probability_df = (
                    probability_df
                    .sort_values(
                        "Probability",
                        ascending=True
                    )
                )


                fig = px.bar(
                    probability_df,
                    x="Probability",
                    y="Disease",
                    orientation="h",
                    text="Probability",
                    labels={
                        "Probability":
                            "Probability (%)",
                        "Disease":
                            ""
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
                        r=30,
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


                st.caption(
                    "Prediction confidence represents "
                    "the model's output probability and "
                    "should not be interpreted as definitive "
                    "field diagnosis."
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
# ABOUT
# ============================================================

elif page == "📘 About":

    st.header(
        "📘 About the Project"
    )


    st.markdown("""
    <div class="hero">

        <h1>University of Agricultural Sciences, Bangalore</h1>

        <p>
        Artificial intelligence-assisted maize disease
        recognition research platform.
        </p>

    </div>
    """, unsafe_allow_html=True)


    try:

        st.image(
            "uas_campus.jpg",
            caption="University of Agricultural Sciences, Bangalore – GKVK campus",
            use_container_width=True
        )

    except:

        st.info(
            "Add the UAS campus image as 'uas_campus.jpg'."
        )


    st.markdown(
        "## 🌽 Project Overview"
    )

    st.write(
        """
        The Maize Disease Recognition System is being
        developed as an AI-assisted platform for automated
        identification of maize diseases from plant images.

        The system combines computer vision, deep learning
        and an interactive web interface to provide rapid
        image-based disease predictions.
        """
    )


    st.markdown(
        "## 🎯 Objectives"
    )

    objectives = [
        "Develop an automated maize disease recognition system.",
        "Apply deep learning for image-based disease classification.",
        "Provide a simple interface for researchers and agricultural users.",
        "Support rapid preliminary disease identification.",
        "Explore explainable and interpretable AI for plant disease diagnosis."
    ]

    for objective in objectives:

        st.markdown(
            f"✓ {objective}"
        )


    st.markdown(
        "## 🏫 Institutional Background"
    )

    st.write(
        """
        The University of Agricultural Sciences, Bangalore
        (UAS-B) is a major agricultural education and research
        institution in Karnataka. The university's GKVK campus
        provides an important environment for agricultural
        research, education, extension and technology development.
        """
    )


    st.markdown(
        "## 👨‍🔬 Development Team"
    )

    st.write(
        """
        **Maruthi Prasad B. P.**  
        Department of Genetics and Plant Breeding  
        University of Agricultural Sciences, Bangalore

        **Harish J.**  
        Department of Plant Pathology  
        University of Agricultural Sciences, Bangalore

        **M.K. Prasannakumar**  
        Department of Plant Pathology  
        University of Agricultural Sciences, Bangalore
        """
    )


# ============================================================
# MANAGEMENT
# ============================================================

elif page == "🌱 Management Strategies":

    st.header(
        "🌱 Maize Disease Management"
    )

    st.info(
        """
        Management recommendations should be updated
        according to the final maize diseases included
        in the trained model and locally validated
        agricultural recommendations.
        """
    )


    st.markdown("""
    ## 🌾 Integrated Disease Management

    ### 1. Resistant Cultivars
    Use cultivars with resistance or tolerance to
    locally important maize diseases whenever available.

    ### 2. Healthy Seed
    Use healthy, high-quality seed and avoid planting
    visibly infected seed lots.

    ### 3. Field Sanitation
    Remove or appropriately manage infected plant
    residues to reduce pathogen carry-over.

    ### 4. Crop Rotation
    Where appropriate, rotate maize with suitable
    non-host crops.

    ### 5. Balanced Nutrition
    Maintain balanced crop nutrition and avoid
    excessive fertilizer application.

    ### 6. Field Monitoring
    Regularly inspect plants for early disease symptoms.

    ### 7. Responsible Chemical Management
    Use plant protection products only when required,
    following locally approved recommendations and
    product-label instructions.
    """)


    st.warning(
        """
        ⚠️ The AI prediction is intended as a
        decision-support aid and should not replace
        confirmation by a qualified plant pathologist
        or agricultural expert.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    <b>🌽 Maize Disease Recognition System</b>

    <br>

    University of Agricultural Sciences, Bangalore

    <br><br>

    AI • Agriculture • Plant Health • Research

</div>
""", unsafe_allow_html=True)
