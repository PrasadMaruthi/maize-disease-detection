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
    page_title="MaizeVision AI | UAS Bangalore",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');
:root{--g950:#062b1a;--g900:#073b23;--g800:#0b5d32;--g700:#0d7a3d;--g600:#16934b;--g100:#e9f8ef;--line:#dfeae3;--ink:#102218;--muted:#65756b;}
html,body,[class*="css"]{font-family:'Inter',sans-serif}.stApp{background:radial-gradient(circle at 90% 4%,rgba(22,147,75,.09),transparent 24rem),radial-gradient(circle at 0% 28%,rgba(229,185,63,.06),transparent 20rem),#f5faf7;color:var(--ink)}
.block-container{max-width:1450px;padding-top:1.5rem;padding-bottom:3rem}[data-testid="stHeader"]{background:rgba(245,250,247,.82);backdrop-filter:blur(12px)}
section[data-testid="stSidebar"]{background:radial-gradient(circle at 20% 0%,rgba(42,166,91,.28),transparent 18rem),linear-gradient(180deg,#073b23,#042719);border-right:1px solid rgba(255,255,255,.08)}
section[data-testid="stSidebar"] *{color:#f5fff8!important}.sidebar-title{text-align:center;font-family:'Manrope',sans-serif;font-size:24px;font-weight:800;letter-spacing:-.5px}.sidebar-subtitle{text-align:center;font-size:12px;line-height:1.55;opacity:.78;margin-bottom:18px}.sidebar-title:before{content:'🌽';display:block;margin:0 auto 10px;width:58px;height:58px;line-height:58px;border-radius:18px;background:linear-gradient(145deg,#2bb967,#0d6735);box-shadow:0 12px 28px rgba(0,0,0,.25);font-size:30px}.sidebar-title:after{content:'AI-ASSISTED RESEARCH PLATFORM';display:block;margin-top:8px;font-size:9px;letter-spacing:1.3px;color:#bce8ca!important}
section[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.10)}div[role="radiogroup"]{gap:6px}div[role="radiogroup"] label{border-radius:12px!important;padding:8px 10px!important;transition:.2s}div[role="radiogroup"] label:hover{background:rgba(255,255,255,.09)}
.hero{position:relative;overflow:hidden;min-height:300px;padding:44px 48px;border-radius:30px;background:radial-gradient(circle at 88% 24%,rgba(94,224,143,.30),transparent 17rem),radial-gradient(circle at 70% 120%,rgba(229,185,63,.18),transparent 18rem),linear-gradient(135deg,#073b23,#0b6836 55%,#12944b);color:#fff;box-shadow:0 24px 60px rgba(7,59,35,.18);margin-bottom:28px}.hero:after{content:'🌽';position:absolute;right:5%;bottom:-48px;font-size:220px;opacity:.11;transform:rotate(-12deg)}.hero-kicker{display:inline-block;padding:7px 12px;border:1px solid rgba(255,255,255,.18);border-radius:999px;background:rgba(255,255,255,.09);font-size:11px;font-weight:800;letter-spacing:.8px}.hero h1{position:relative;z-index:1;max-width:820px;margin:18px 0 10px;font-family:'Manrope',sans-serif;font-size:clamp(36px,5vw,60px);line-height:1.03;letter-spacing:-2.4px}.hero p{position:relative;z-index:1;max-width:760px;color:rgba(255,255,255,.87);font-size:17px;line-height:1.7;margin:0}.hero-pills{position:relative;z-index:2;display:flex;gap:9px;flex-wrap:wrap;margin-top:24px}.hero-pill{padding:9px 13px;border-radius:12px;background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.13);font-size:11px;font-weight:700}
.section-title{color:var(--g900);font-family:'Manrope',sans-serif;font-weight:800;font-size:29px;letter-spacing:-.8px;margin-top:30px;margin-bottom:14px}.section-title:after{content:'';display:block;width:42px;height:3px;border-radius:4px;background:linear-gradient(90deg,#16934b,#b9d94b);margin-top:7px}.info-card,.disease-card{background:rgba(255,255,255,.94);border:1px solid var(--line);box-shadow:0 8px 28px rgba(16,34,24,.055);border-radius:20px;transition:.2s}.info-card{padding:23px}.info-card:hover,.disease-card:hover{transform:translateY(-2px);box-shadow:0 15px 36px rgba(16,34,24,.09);border-color:#c7dfcf}.info-card h3,.disease-card h3{color:var(--g900);font-family:'Manrope',sans-serif}.disease-card{padding:21px;border-left:4px solid #16934b}.disease-card h3{margin-top:0;font-size:16px}.disease-card p{color:var(--muted);font-size:12.5px;line-height:1.65}
.prediction-card{background:radial-gradient(circle at 100% 0%,rgba(22,147,75,.13),transparent 15rem),linear-gradient(145deg,#effaf2,#fff);padding:30px;border-radius:22px;border:1px solid #b9dec4;box-shadow:0 14px 38px rgba(11,93,50,.09);text-align:center;margin-top:20px}.prediction-label{font-size:11px;color:var(--muted);font-weight:800;letter-spacing:1.3px;text-transform:uppercase}.prediction-name{font-family:'Manrope',sans-serif;font-size:31px;font-weight:800;color:var(--g800);margin-top:7px}
.stButton>button{min-height:45px;border-radius:13px;font-weight:800;border:1px solid #cfe2d4;transition:.2s}.stButton>button:hover{transform:translateY(-1px)}.stButton>button[kind="primary"]{background:linear-gradient(135deg,#0b6a38,#159a50);border:0;box-shadow:0 9px 22px rgba(11,106,56,.18)}
div[data-testid="stFileUploader"]{border:1px dashed #9bcaae;border-radius:20px;padding:10px;background:linear-gradient(145deg,#f2fbf5,#fff)}div[data-testid="stAlert"]{border-radius:14px}div[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:16px;padding:13px;box-shadow:0 6px 20px rgba(16,34,24,.045)}
.footer{text-align:center;padding:28px 10px 8px;margin-top:45px;border-top:1px solid var(--line);color:#718078;font-size:12px;line-height:1.7}#MainMenu{visibility:hidden}footer{visibility:hidden}
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
<div class="sidebar-title">MaizeVision AI</div>

<div class="sidebar-subtitle">
Intelligent maize disease recognition<br>
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

    st.markdown("""
    <div class="hero">
        <div class="hero-kicker">● AI-ASSISTED AGRICULTURAL RESEARCH PLATFORM</div>
        <h1>See the symptom.<br>Understand the disease.</h1>
        <p>MaizeVision AI uses deep-learning image classification to provide rapid preliminary recognition of important maize disease classes from plant images.</p>
        <div class="hero-pills">
            <span class="hero-pill">🌽 6 classification classes</span>
            <span class="hero-pill">🧠 Deep learning</span>
            <span class="hero-pill">📊 Confidence analysis</span>
            <span class="hero-pill">🏫 UAS Bangalore</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
       


    total_images = 303 + 703 + 3435 + 321 + 1549 + 8123
    s1, s2, s3, s4 = st.columns(4)
    with s1: st.metric("Recognition classes", "6")
    with s2: st.metric("Current images", f"{total_images:,}")
    with s3: st.metric("Model input", "224 × 224")
    with s4: st.metric("Platform", "AI-assisted")

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
        "Upload a clear maize plant image for rapid AI-assisted preliminary classification."
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

🌽 <b>MaizeVision AI · Maize Disease Recognition System</b><br>

Artificial Intelligence-assisted Agricultural Research Platform<br>

University of Agricultural Sciences, Bangalore

<br><br>

© 2026 | Research and Educational Use

</div>
""", unsafe_allow_html=True)
