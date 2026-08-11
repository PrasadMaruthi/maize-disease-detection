import streamlit as st
import tensorflow as tf
import numpy as np

###Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('rice_disease_mobilenetv3.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) ##Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

##Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Management Strategies"])

###Home Page
if(app_mode=="Home"):
    st.header("🌾 RICE DISEASE RECOGNITION SYSTEM")
    image_path = "cover page.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown(""" 
    Welcome to the **Rice Disease Recognition System**! 

    An intelligent platform designed to assist in the **early and accurate identification of rice plant diseases** using advanced deep learning techniques.  
    
    Our goal is to support farmers, researchers, and agricultural professionals in making **timely and informed decisions** to improve crop health and productivity.  
    
    ---  
    ### 🔍 How It Works  
    1. **📤 Upload Image:** Upload an image of a rice leaf or plant showing symptoms of disease.  
    2. **🧠 AI-Based Analysis:** The system processes the image using a trained Convolutional Neural Network (CNN) model to detect patterns associated with different rice diseases.  
    3. **📊 Results & Insights:** Instantly receive the predicted disease class along with relevant information to guide further action.  
    
    ---  
    
    ### 🌟 Why Choose This System?  
    - **🎯 High Accuracy:** Built using modern deep learning techniques for reliable disease classification.  
    - **🖥️ User-Friendly Interface:** Simple and intuitive design for easy usage by both technical and non-technical users.  
    - **⚡ Fast & Efficient:** Get results within seconds to enable quick response and management.  
    - **🌱 Research-Oriented:** Developed with scientific methodology, suitable for academic and practical applications.  
    
    ---  
    
    ### 🌾 Supported Rice Diseases 
    
    The system is trained to identify multiple rice diseases, including:  
        **Brown Spot**  
        **Rice Blast**  
        **Neck Blast**  
        **Sheath Blight**  
        **Sheath Rot**  
        **False Smut**  
        **Udabatta**  
        **Bakanae**  
        **Baharkholidaria**  
        ---  
    ### 🚀 Get Started  
    Navigate to the **"Disease Recognition"** section from the sidebar to upload your image and experience real-time disease detection.  
    
    ---  
    ### 👨‍🔬 About the Project  
    This project focuses on leveraging **artificial intelligence in agriculture** to address challenges in plant disease diagnosis.  
    
    It aims to:  
        Enhance disease detection accuracy  
        Reduce dependency on manual inspection  
        Support sustainable agricultural practice  
        
    For more details about the methodology, dataset, and development, visit the **"About"** section.  
        
    ---  
    
    **✨ Empowering agriculture with AI-driven disease diagnosis.** 
        
        
       


""")
###About Page
elif(app_mode=="About"):
    st.header("About")
    image_path_1 = "complete deployment.png"
    st.image(image_path_1, use_column_width=True)
    st.markdown("""
   ## 📊 Dataset Information

    This dataset has been developed using **offline data augmentation techniques** applied to an original rice disease dataset. 
    Augmentation improves model performance by increasing diversity and reducing overfitting.

    ### 📁 Dataset Content
    - **Training Set:** ~4000 images    

    The dataset includes multiple rice disease classes along with healthy plant samples for effective classification.

    ---

    ## 🌾 About the Project

    The **Rice Disease Recognition System** is designed to assist in the **early and accurate identification of rice plant diseases** 
    using deep learning techniques.

    A **Convolutional Neural Network (CNN)** model has been trained on labeled image data to detect disease patterns from rice leaf images.

    ---

    ## 🎯 Objectives

    - Enable **fast and accurate disease detection**  
    - Reduce dependency on manual diagnosis  
    - Support **precision agriculture**  
    - Improve crop health and productivity  

    ---

    ## 👨‍🔬 Development Team

    **Maruthi Prasad B P**  
    Department of Genetics and Plant Breeding  
    University of Agricultural Sciences, Bangalore  

    **Harish J**  
    Department of Plant Pathology  
    University of Agricultural Sciences, Bangalore  

    **[Developer Name 3]**  
    Department: *(Add here)*  
    University: *(Add here)*  

    ---

    ## 🏫 Acknowledgement

    This work is supported by the **University of Agricultural Sciences, Bangalore**, 
    which provides an excellent academic and research environment for advancing innovations 
    in agriculture and plant sciences.

    ---

    🌱 *Empowering agriculture with AI-driven disease detection.*
   
""")

### Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("🌾 Disease Recognition")

    st.markdown("Upload a rice plant image to detect disease using our trained deep learning model.")

    # Upload Image
    test_image = st.file_uploader("📤 Choose an Image:", type=["jpg", "jpeg", "png"])

    # Show Image
    if test_image is not None:
        if st.button("👁️ Show Image"):
            st.image(test_image, caption="Uploaded Image", use_column_width=True)

        # Predict Button
        if st.button("🔍 Predict"):
            st.write("### 🧠 Prediction Result")

            # Call prediction function
            result_index = model_prediction(test_image)

            # Define Classes (update according to your dataset)
            class_names = [
                "Bharkholdaria blight disease",
                "Brown spot disease",
                "False smut disease",
                "Healthy Rice Plant",
                "Neck blast disease",
                "Rice Blast disease",
                "Sheath blight disease",
                "Sheath rot disease",
                "Udabatta disease"               
            ]

            # Display Result
            st.success(f"🌱 Model Prediction: **{class_names[result_index]}**")

            # Optional: Confidence (if your function returns it)
            # confidence = ...
            # st.info(f"Confidence: {confidence:.2f}%")

    else:
        st.warning("⚠️ Please upload an image to proceed.")


### Management Strategies Page
elif(app_mode=="Management Strategies"):
    st.header("🌾 Disease Management Strategies")

    st.markdown("""
    Effective management of rice diseases is essential to ensure **healthy crop growth, higher yield, and sustainable agriculture**.  
    Below are recommended strategies for major rice diseases identified by this system.

    ---

    ## 🟤 Brown Spot
    - Use **disease-free certified seeds**  
    - Apply balanced fertilization (especially potassium)  
    - Spray fungicides like **Mancozeb** if severe  

    ## 🔴 Rice Blast / Neck Blast
    - Grow **resistant varieties**  
    - Avoid excessive nitrogen fertilization  
    - Apply fungicides such as **Tricyclazole** at early stages  

    ## ⚪ False Smut
    - Use clean seeds and avoid dense planting  
    - Apply fungicides like **Propiconazole** at booting stage  

    ## 🟢 Sheath Blight
    - Maintain proper spacing to reduce humidity  
    - Avoid excessive nitrogen  
    - Apply fungicides such as **Validamycin**  

    ## 🟡 Sheath Rot
    - Use healthy seeds and ensure good drainage  
    - Spray fungicides like **Carbendazim**  

    ## 🟠 Udabatta
    - Treat seeds with fungicides before sowing  
    - Remove and destroy infected plants  

    ## 🌿 Baharkholidaria (Bakanae / Foot Rot)
    - Use **hot water seed treatment** or fungicide-treated seeds  
    - Rogue out infected seedlings early  

    ## 🌱 Healthy Plants
    - Maintain proper crop management practices  
    - Regular monitoring for early disease detection  

    ---

    ## 🌟 General Recommendations
    - Practice **crop rotation**  
    - Use **certified disease-free seeds**  
    - Maintain proper field sanitation  
    - Monitor fields regularly  
    - Apply fungicides only when necessary  

    ---

    ⚠️ *Note: Always follow recommended agricultural guidelines and consult local experts before applying chemicals.*
    """)