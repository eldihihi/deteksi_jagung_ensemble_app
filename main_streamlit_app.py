# main_streamlit_app.py (Untuk Model Ensemble VGG16, ResNet50, InceptionV3 dengan Gemini API)

import streamlit as st
import os
import numpy as np
import gdown # Used to download models from Google Drive
from tensorflow.keras.models import load_model # To load Keras/TensorFlow models
from PIL import Image # Used for image manipulation (e.g., resizing)
import requests # Used to make HTTP requests to Gemini API
import json # Used to parse JSON responses from Gemini API
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg # VGG16 preprocessing
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet # ResNet50 preprocessing
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception # InceptionV3 preprocessing

# --- Model Configuration ---
MODEL_DIR = "models"

# Google Drive URL for your fine-tuned ensemble model
# IMPORTANT: This URL is for the 'ensemble_fine_tuned_model.h5'
MODEL_URLS = {
    "ensemble": "https://drive.google.com/uc?id=13jUW2aYyiNblInQnrK9UR8f0EcUalSTQ" # YOUR ENSEMBLE MODEL URL
}

# Local filename for the model
MODEL_FILENAMES = {
    "ensemble": "ensemble_fine_tuned_model.h5"
}

# Target image sizes for each model in the ensemble
IMG_SIZE_VGG_RESNET = 224 # VGG16 and ResNet50 input size
IMG_SIZE_INCEPTION = 299 # InceptionV3 input size

# Class names for prediction results (ensure the order matches your training)
CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# --- Gemini API Configuration ---
GEMINI_API_KEY = "" # This will be automatically filled by Canvas/Streamlit Secrets
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# --- Helper Function for Image Preprocessing ---
def preprocess_image_for_ensemble(image_path):
    """
    Loads and preprocesses an image for each branch of the ensemble model.
    Returns a tuple: (vgg_resnet_image_array, inception_image_array)
    """
    img = Image.open(image_path).convert('RGB')

    # Preprocess for VGG16/ResNet50 branch (224x224)
    img_vgg_resnet = img.resize((IMG_SIZE_VGG_RESNET, IMG_SIZE_VGG_RESNET))
    img_array_vgg_resnet = np.array(img_vgg_resnet)
    img_array_vgg_resnet = np.expand_dims(img_array_vgg_resnet, axis=0) # Add batch dimension
    
    # Apply VGG/ResNet specific preprocessing (rescaling to [-1, 1] or similar)
    # Keras applications' preprocess_input functions handle normalization.
    # If your training used rescale=1./255, ensure consistency here.
    # For ensemble, it's safer to use the specific preprocess_input for each base model.
    processed_vgg_resnet_image = preprocess_vgg(img_array_vgg_resnet) # VGG and ResNet preprocessing are similar

    # Preprocess for InceptionV3 branch (299x299)
    img_inception = img.resize((IMG_SIZE_INCEPTION, IMG_SIZE_INCEPTION))
    img_array_inception = np.array(img_inception)
    img_array_inception = np.expand_dims(img_array_inception, axis=0) # Add batch dimension
    
    # Apply InceptionV3 specific preprocessing
    processed_inception_image = preprocess_inception(img_array_inception)

    return processed_vgg_resnet_image, processed_inception_image

# --- Model Loading with Streamlit Caching ---
@st.cache_resource(show_spinner=False, hash_funcs={"_thread.RLock": lambda _: None})
def load_ensemble_model_cached():
    """
    Downloads and loads the ensemble model. This function is cached by Streamlit.
    """
    loaded_model = None
    
    # Create models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    status_placeholder = st.empty() # Placeholder for dynamic messages

    name = "ensemble" # Model name to download
    model_filepath = os.path.join(MODEL_DIR, MODEL_FILENAMES[name])
    
    # Download model if it doesn't exist
    if not os.path.exists(model_filepath):
        status_placeholder.info(f"⬇️ Mengunduh model {name}...")
        try:
            # Use fuzzy=True for gdown to be more robust with potential redirects
            gdown.download(url=MODEL_URLS[name], output=model_filepath, quiet=True, fuzzy=True)
            status_placeholder.success(f"✅ Model {name} berhasil diunduh.")
        except Exception as e:
            status_placeholder.error(f"❌ Gagal mengunduh model {name}: {e}")
            st.exception(e) # Display full exception in Streamlit
            return None # Indicate failure
            
    # Load model
    status_placeholder.info(f"🧠 Memuat model {name}...")
    try:
        # For custom layers or functions (like 'swish' in EfficientNet), you might need
        # custom_objects argument in load_model. For VGG/ResNet/Inception, usually not needed.
        # If you encounter issues with custom activations (e.g., 'swish' in EfficientNet),
        # ensure the necessary library (e.g., `efficientnet.tfkeras`) is imported BEFORE load_model.
        # For this ensemble, we are using standard Keras applications, so it should be fine.
        model = load_model(model_filepath)
        loaded_model = model
        status_placeholder.success(f"✅ Model {name} berhasil dimuat.")
    except Exception as e:
        status_placeholder.error(f"❌ Gagal memuat model {name}: {e}")
        st.exception(e) # Display full exception in Streamlit
        return None # Indicate failure
            
    return loaded_model

# --- Function to get treatment suggestions from Gemini API ---
def get_treatment_suggestions(plant_disease):
    """
    Calls the Gemini API to get treatment suggestions for a given plant disease.
    """
    api_key = os.getenv("GEMINI_API_KEY", "") # Get API key from environment variable
    if not api_key:
        st.error("API Key untuk Gemini tidak ditemukan. Harap pastikan GEMINI_API_KEY diatur di Streamlit Secrets.")
        return "Tidak dapat memberikan saran perawatan tanpa API Key."

    prompt = f"Berikan saran perawatan singkat dan praktis untuk penyakit daun jagung: {plant_disease}. Fokus pada langkah-langkah yang bisa dilakukan petani. Berikan dalam bentuk poin-poin singkat."
    
    chat_history = []
    chat_history.append({"role": "user", "parts": [{"text": prompt}]})
    
    payload = {"contents": chat_history}
    
    try:
        with st.spinner(f"Mencari saran perawatan untuk {plant_disease} dari AI..."):
            api_url_with_key = f"{GEMINI_API_URL}?key={api_key}"
            
            response = requests.post(api_url_with_key, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            result = response.json()

            if result and result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                st.warning("Tidak dapat memperoleh saran perawatan dari AI. Struktur respons tidak sesuai.")
                st.json(result) # Display full response for debugging
                return "Tidak ada saran perawatan yang ditemukan."
    except requests.exceptions.RequestException as e:
        st.error(f"Terjadi kesalahan saat memanggil API Gemini: {e}")
        return "Gagal mendapatkan saran perawatan karena masalah koneksi atau API."
    except json.JSONDecodeError as e:
        st.error(f"Terjadi kesalahan saat mengurai respons JSON dari Gemini: {e}")
        return "Gagal mendapatkan saran perawatan karena masalah format data."
    except Exception as e:
        st.error(f"Terjadi kesalahan tidak terduga: {e}")
        return "Terjadi kesalahan saat mendapatkan saran perawatan."


# --- Streamlit Application UI ---

# Set basic page configuration
st.set_page_config(
    page_title="Deteksi Penyakit Daun Jagung (Ensemble CNN + Gemini AI)",
    page_icon="🌽",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Inject Tailwind CSS and custom styles for Option 3
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #e0f2f7; /* Light cyan background */
        }
        .card-base {
            background-color: #ffffff;
            border-radius: 1.5rem;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        }
        .card-upload {
            border-left: 8px solid #38b2ac; /* Teal accent */
        }
        .card-preview {
            border-left: 8px solid #63b3ed; /* Blue accent */
        }
        .card-prediction {
            border-left: 8px solid #f6ad55; /* Orange accent */
        }
        .card-advice {
            border-left: 8px solid #48bb78; /* Green accent */
        }
        .chart-container {
            position: relative;
            width: 100%;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
            height: 300px;
            background-color: #edf2f7; /* Placeholder background */
            border-radius: 0.75rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
            color: #6b7280;
        }
    </style>
""", unsafe_allow_html=True)

# Main content container
st.markdown('<div class="container max-w-xl w-full mx-auto">', unsafe_allow_html=True)

st.markdown("""
    <h1 class="text-3xl md:text-4xl font-bold text-center text-gray-800 mb-4">
        🌽 Deteksi Penyakit Daun Jagung
    </h1>
    <p class="text-center text-gray-600 mb-8">
        Aplikasi cerdas untuk diagnosis cepat dan saran perawatan menggunakan Model Ensemble.
    </p>
""", unsafe_allow_html=True)

# Call the cached model loading function
ensemble_model = load_ensemble_model_cached()

# Check if model was loaded successfully
if ensemble_model is None:
    st.error("Aplikasi tidak dapat berfungsi penuh karena ada masalah dalam memuat model. Silakan periksa log deployment.")
    st.stop()

# --- Image Upload Section ---
st.markdown('<div class="card-base card-upload p-6 md:p-8 mb-6">', unsafe_allow_html=True)
st.markdown('<h2 class="text-2xl font-semibold text-teal-700 mb-4">Unggah Gambar Daun</h2>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Pilih Gambar Anda:",
    type=["jpg", "jpeg", "png"],
    help="Hanya file JPG, JPEG, atau PNG.",
    key="file_uploader_ensemble" # Add unique key for Streamlit
)
st.markdown('<p class="mt-2 text-sm text-gray-500">Hanya file JPG, JPEG, atau PNG.</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True) # Close card-upload div

# Initialize session state for results
if 'predicted_result_ensemble' not in st.session_state:
    st.session_state.predicted_result_ensemble = None
if 'confidence_level_ensemble' not in st.session_state:
    st.session_state.confidence_level_ensemble = None
if 'treatment_advice_ensemble' not in st.session_state:
    st.session_state.treatment_advice_ensemble = None
if 'uploaded_image_data_ensemble' not in st.session_state:
    st.session_state.uploaded_image_data_ensemble = None

if uploaded_file is not None:
    # Store uploaded image data in session state
    st.session_state.uploaded_image_data_ensemble = uploaded_file.getvalue()

    # --- Image Preview Section ---
    st.markdown('<div class="card-base card-preview p-6 md:p-8 mb-6 text-center">', unsafe_allow_html=True)
    st.markdown('<h2 class="text-2xl font-semibold text-blue-700 mb-4">Gambar yang Diunggah</h2>', unsafe_allow_html=True)
    st.image(st.session_state.uploaded_image_data_ensemble, caption='Gambar yang Diunggah', use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True) # Close card-preview div

    st.write("")
    st.write("---")
    st.write("Menganalisis gambar...")

    # Create a temporary file path
    temp_image_path = os.path.join(MODEL_DIR, "uploaded_temp_image_ensemble.jpg")
    with open(temp_image_path, "wb") as f:
        f.write(st.session_state.uploaded_image_data_ensemble)
    
    prediction_status_placeholder = st.empty()
    prediction_status_placeholder.info("Melakukan prediksi...")

    try:
        # Preprocess image for ensemble model (returns two arrays)
        processed_vgg_resnet_image, processed_inception_image = preprocess_image_for_ensemble(temp_image_path)
        
        # Predict with the ensemble model (requires a list of inputs)
        prediction = ensemble_model.predict([processed_vgg_resnet_image, processed_vgg_resnet_image, processed_inception_image])
        
        # Get predicted class index and confidence level
        predicted_class_index = np.argmax(prediction)
        result = CLASS_NAMES[predicted_class_index]
        confidence = np.max(prediction) * 100 # Convert to percentage

        st.session_state.predicted_result_ensemble = result
        st.session_state.confidence_level_ensemble = f"{confidence:.2f}%"

        prediction_status_placeholder.success("✅ Prediksi Selesai!")
        st.write("---")
        
        # --- Get and store treatment suggestions from Gemini ---
        if st.session_state.predicted_result_ensemble == "Healthy":
            st.session_state.treatment_advice_ensemble = "Daun jagung terlihat sehat. Pertahankan praktik perawatan yang baik!"
        else:
            st.session_state.treatment_advice_ensemble = get_treatment_suggestions(st.session_state.predicted_result_ensemble)
        
    except Exception as e:
        prediction_status_placeholder.error("❌ Terjadi kesalahan saat memproses gambar atau melakukan prediksi.")
        st.exception(e) # Display full exception for debugging
        st.session_state.predicted_result_ensemble = None
        st.session_state.confidence_level_ensemble = None
        st.session_state.treatment_advice_ensemble = None
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

# Display results if available in session state
if st.session_state.predicted_result_ensemble:
    # --- Prediction Result Section ---
    st.markdown('<div class="card-base card-prediction p-6 md:p-8 mb-6">', unsafe_allow_html=True)
    st.markdown('<h2 class="text-2xl font-semibold text-orange-700 mb-4">Hasil Deteksi</h2>', unsafe_allow_html=True)
    st.markdown(f"""
        <p class="text-gray-800 text-lg mb-2">
            Penyakit Terdeteksi: <span class="font-bold text-orange-600">{st.session_state.predicted_result_ensemble}</span>
        </p>
        <p class="text-gray-800 text-lg">
            Tingkat Keyakinan: <span class="font-bold text-orange-600">{st.session_state.confidence_level_ensemble}</span>
        </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True) # Close card-prediction div

    # --- Treatment Advice Section ---
    st.markdown('<div class="card-base card-advice p-6 md:p-8 mb-6">', unsafe_allow_html=True)
    st.markdown('<h2 class="text-2xl font-semibold text-green-700 mb-4">Saran Perawatan dari AI</h2>', unsafe_allow_html=True)
    st.markdown(f'<div class="text-gray-800 leading-relaxed">{st.session_state.treatment_advice_ensemble}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True) # Close card-advice div

# Placeholder for Chart (if needed in the future)
st.markdown("""
    <div class="chart-container mt-8 hidden">
        <p>Placeholder untuk Grafik (misal: distribusi keyakinan kelas)</p>
    </div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True) # Close main container
