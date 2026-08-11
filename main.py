import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Maize Disease Recognition System",
    page_icon="🌽",
    layout="wide"
)


# ============================================================
# MAIZE DISEASE CLASSES
# ============================================================
# IMPORTANT:
# Replace these names with the exact class order used during
# maize model training.

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
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "best_model.keras"
    )

    return model


# ============================================================
# MODEL PREDICTION
# ============================================================

def model_prediction(test_image):

    model = load_model()

    image = Image.open(test_image).convert("RGB")

    image = image.resize((224, 224))

    input_arr = np.array(image)

    # Convert single image into batch
    input_arr = np.expand_dims(input_arr, axis=0)

    # IMPORTANT:
    # Do NOT divide by 255 here if the final model contains
    # its own preprocessing/rescaling layer.
    prediction = model.predict(
        input_arr,
        verbose=0
    )

    result_index = np.argmax(prediction[0])

    confidence = float(
        prediction[0][result_index] * 100
    )

    return result_index, confidence, prediction[0]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌽 Dashboard")

app_mode = st.sidebar.selectbox(
    "Select Page",
    [
        "Home",
        "About",
        "Disease Recognition",
        "Management Strategies"
    ]
)


# ============================================================
# HOME PAGE
# ============================================================

if app_mode == "Home":

    st.header(
        "🌽 MAIZE DISEASE RECOGNITION SYSTEM"
    )

    image_path = "cover page.jpg"

    try:
        st.image(
            image_path,
            use_container_width=True
        )
    except:
        st.info(
            "Add your maize cover image as "
            "'cover page.jpg' in the application folder."
        )


    st.markdown(
        """
## Welcome to the **Maize Disease Recognition System**!

An intelligent platform designed to assist in the
**early and accurate identification of maize plant diseases**
using deep learning-based image classification.

The system is intended to support farmers, researchers,
plant pathologists and agricultural professionals in
rapidly identifying disease symptoms from plant images.

---

### 🔍 How It Works

**1. 📤 Upload Image**

Upload an image of a maize leaf or plant showing
visible disease symptoms.

**2. 🧠 AI-Based Analysis**

The uploaded image is processed using a trained
deep learning image-classification model.

**3. 📊 Prediction**

The system identifies the most probable disease class
and provides the corresponding prediction confidence.

---

### 🌟 Key Features

- 🎯 Automated maize disease classification
- 🧠 Deep learning-based image analysis
- ⚡ Rapid prediction
- 🖥️ Simple and user-friendly interface
- 📊 Prediction confidence
- 🌱 Agricultural disease-management information
- 🔬 Research-oriented platform

---

### 🌽 Supported Maize Diseases

The final disease classes will be displayed here
after the maize model is trained.

---

### 🚀 Get Started

Navigate to **Disease Recognition** from the sidebar
to upload a maize plant image and obtain an AI-based
disease prediction.

---

### 👨‍🔬 About the Project

This project aims to apply artificial intelligence
and computer vision to maize disease diagnosis.

The system is intended to:

- Improve rapid disease identification
- Reduce dependence on preliminary visual inspection
- Support agricultural decision-making
- Facilitate AI-assisted plant disease diagnosis

---

**🌱 Empowering maize disease diagnosis with artificial intelligence.**
"""
    )


# ============================================================
# ABOUT PAGE
# ============================================================

elif app_mode == "About":

    st.header("📘 About the Project")

    image_path_1 = "complete deployment.png"

    try:
        st.image(
            image_path_1,
            use_container_width=True
        )
    except:
        st.info(
            "Add your project workflow image as "
            "'complete deployment.png'."
        )


    st.markdown(
        """
## 📊 Dataset Information

The maize disease dataset will consist of labelled
maize plant images representing different disease
categories.

The final dataset description, number of images,
training-validation split and augmentation strategy
will be updated after completion of model development.

---

## 🌽 About the Project

The **Maize Disease Recognition System** is designed
to assist in the identification of maize diseases using
deep learning and computer vision.

A convolutional neural network will be trained using
labelled maize disease images and subsequently
integrated into this web-based application.

---

## 🎯 Objectives

- Enable rapid maize disease identification
- Develop an automated image-based diagnostic system
- Support agricultural research and decision-making
- Reduce dependence on preliminary manual inspection
- Provide an accessible AI-based disease recognition tool

---

## 👨‍🔬 Development Team

**Maruthi Prasad B. P.**  
Department of Genetics and Plant Breeding  
University of Agricultural Sciences, Bangalore

**Harish J.**  
Department of Plant Pathology  
University of Agricultural Sciences, Bangalore

**M.K. Prasannakumar**  
Department of Plant Pathology  
University of Agricultural Sciences, Bangalore

---

## 🏫 Acknowledgement

This work is supported by the
**University of Agricultural Sciences, Bangalore**,
which provides an academic and research environment
for advancing innovations in agriculture and plant sciences.

---

🌱 *Empowering maize disease detection with AI.*
"""
    )


# ============================================================
# DISEASE RECOGNITION PAGE
# ============================================================

elif app_mode == "Disease Recognition":

    st.header("🌽 Disease Recognition")

    st.markdown(
        """
Upload a maize plant image to obtain a disease prediction
using the trained deep learning model.
"""
    )

    # --------------------------------------------------------
    # IMAGE UPLOAD
    # --------------------------------------------------------

    test_image = st.file_uploader(
        "📤 Choose a maize plant image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    # --------------------------------------------------------
    # IMAGE DISPLAY
    # --------------------------------------------------------

    if test_image is not None:

        image = Image.open(
            test_image
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Maize Image",
            use_container_width=True
        )


        # ----------------------------------------------------
        # PREDICTION BUTTON
        # ----------------------------------------------------

        if st.button(
            "🔍 Predict Disease",
            use_container_width=True
        ):

            try:

                result_index, confidence, probabilities = (
                    model_prediction(test_image)
                )

                predicted_class = CLASS_NAMES[
                    result_index
                ]


                # ------------------------------------------------
                # RESULT
                # ------------------------------------------------

                st.write(
                    "### 🧠 Prediction Result"
                )

                st.success(
                    f"🌽 Predicted Disease: "
                    f"**{predicted_class}**"
                )

                st.info(
                    f"🎯 Prediction Confidence: "
                    f"**{confidence:.2f}%**"
                )


                # ------------------------------------------------
                # PROBABILITY TABLE
                # ------------------------------------------------

                st.write(
                    "### 📊 Class Probabilities"
                )

                probability_data = []

                for i, probability in enumerate(
                    probabilities
                ):

                    probability_data.append(
                        {
                            "Disease": CLASS_NAMES[i],
                            "Probability (%)":
                                round(
                                    float(probability) * 100,
                                    2
                                )
                        }
                    )


                probability_data = sorted(
                    probability_data,
                    key=lambda x:
                        x["Probability (%)"],
                    reverse=True
                )


                st.dataframe(
                    probability_data,
                    use_container_width=True,
                    hide_index=True
                )


            except Exception as e:

                st.error(
                    "Prediction could not be performed."
                )

                st.exception(e)

    else:

        st.warning(
            "⚠️ Please upload a maize plant image to proceed."
        )


# ============================================================
# MANAGEMENT STRATEGIES PAGE
# ============================================================

elif app_mode == "Management Strategies":

    st.header(
        "🌽 Maize Disease Management Strategies"
    )

    st.markdown(
        """
Effective management of maize diseases requires
integrated approaches involving resistant varieties,
healthy planting material, field sanitation, appropriate
crop management and timely disease monitoring.

The management recommendations below will be updated
according to the final disease classes included in the
trained maize model.

---

## 🌱 General Management Recommendations

### 🌾 Use Resistant Varieties

Where available, use maize cultivars with resistance
or tolerance to locally important diseases.

### 🌱 Use Healthy Planting Material

Use healthy, high-quality seed and avoid planting
material showing disease symptoms.

### 🧹 Field Sanitation

Remove and properly manage infected crop residues
that may serve as sources of primary inoculum.

### 💧 Proper Crop Management

Maintain appropriate plant density, irrigation and
nutrient management to reduce conditions favourable
for disease development.

### 🔄 Crop Rotation

Where appropriate, practise crop rotation with
non-host crops to reduce pathogen carry-over.

### 🔎 Regular Monitoring

Regularly inspect maize fields for early symptoms
and take appropriate management measures when required.

### 🧪 Responsible Chemical Management

Where fungicides are recommended, select registered
products and follow locally approved label instructions,
application timings and safety precautions.

---

## ⚠️ Important

Disease management recommendations should be based
on accurate disease identification, local conditions
and recommendations from agricultural extension
services or plant pathology experts.

The AI prediction should be considered a
**decision-support tool and not a replacement for
expert diagnosis**.
"""
    )