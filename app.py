import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "paddy_disease_model.h5"
CLASS_FILE = "class_names.txt"
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Paddy Disease", page_icon="🌾", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# Load class names
@st.cache_data
def load_classes():
    with open(CLASS_FILE, "r") as f:
        return [line.strip() for line in f.readlines()]

model = load_model()
class_names = load_classes()

# 🌐 Language selection
language = st.selectbox(
    "🌐 Select Language / భాష ఎంచుకోండి / भाषा चुनें",
    ["English", "Telugu", "Hindi"]
)

# UI text
ui_text = {
    "English": {
        "title": "🌾 Paddy Disease Recognition",
        "upload": "Upload Paddy Leaf Image",
        "image": "Uploaded Image",
        "disease": "Disease Name",
        "confidence": "Confidence",
        "confidence_info": "This percentage shows how sure the AI model is about its prediction.",
        "fertilizer": "Fertilizer Suggestion",
        "wait": "Please upload a paddy leaf image."
    },
    "Telugu": {
        "title": "🌾 వరి వ్యాధి గుర్తింపు",
        "upload": "వరి ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "image": "అప్‌లోడ్ చేసిన చిత్రం",
        "disease": "వ్యాధి పేరు",
        "confidence": "అంచనా నమ్మకం",
        "confidence_info": "ఈ శాతం AI మోడల్ తన అంచనాపై ఎంత నమ్మకంగా ఉందో చూపిస్తుంది.",
        "fertilizer": "ఎరువు సూచన",
        "wait": "దయచేసి వరి ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి."
    },
    "Hindi": {
        "title": "🌾 धान रोग पहचान",
        "upload": "धान के पत्ते की छवि अपलोड करें",
        "image": "अपलोड की गई छवि",
        "disease": "रोग का नाम",
        "confidence": "विश्वास स्तर",
        "confidence_info": "यह प्रतिशत बताता है कि AI मॉडल अपने अनुमान को लेकर कितना निश्चित है।",
        "fertilizer": "उर्वरक सुझाव",
        "wait": "कृपया धान के पत्ते की छवि अपलोड करें।"
    }
}

# Disease names (PURE local language)
disease_translation = {
    "Bacterial_leaf_blight": {
        "English": "Bacterial Leaf Blight",
        "Telugu": "బ్యాక్టీరియా వల్ల వచ్చే ఆకు ఎండ వ్యాధి",
        "Hindi": "बैक्टीरिया से होने वाला पत्ती झुलसा रोग"
    },
    "Brown_spot": {
        "English": "Brown Spot",
        "Telugu": "గోధుమ మచ్చల వ్యాధి",
        "Hindi": "भूरे धब्बों का रोग"
    },
    "Leaf_smut": {
        "English": "Leaf Smut",
        "Telugu": "నల్ల మచ్చల ఆకు వ్యాధి",
        "Hindi": "काले धब्बों वाली पत्ती का रोग"
    }
}

# Fertilizer + farmer suggestions (DIFFERENT for each disease)
fertilizer = {
    "Bacterial_leaf_blight": {
        "English": """Use balanced NPK fertilizer. Avoid excess nitrogen. 
Maintain proper drainage. Remove infected leaves and keep field clean.""",

        "Telugu": """సమతుల్యమైన NPK ఎరువులు వాడండి. నైట్రోజన్ ఎక్కువగా వాడకండి. 
పొలం లో నీటి ప్రవాహం సరిగా ఉంచండి. 
రోగగ్రస్త ఆకులను తొలగించి పొలాన్ని శుభ్రంగా ఉంచండి.""",

        "Hindi": """संतुलित NPK उर्वरक का उपयोग करें। अधिक नाइट्रोजन न दें। 
खेत में पानी का सही निकास बनाए रखें। 
संक्रमित पत्तियों को हटाएं और खेत को साफ रखें।"""
    },

    "Brown_spot": {
        "English": """Apply balanced NPK fertilizer. Add potassium and zinc. 
Use healthy seeds and avoid water stress. Maintain soil nutrients.""",

        "Telugu": """సమతుల్యమైన NPK ఎరువులు వాడండి. పొటాషియం మరియు జింక్ ఇవ్వండి. 
ఆరోగ్యమైన విత్తనాలు వాడండి మరియు నీటి కొరత రాకుండా చూడండి.""",

        "Hindi": """संतुलित NPK उर्वरक का उपयोग करें। पोटाश और जिंक दें। 
स्वस्थ बीज का उपयोग करें और पानी की कमी न होने दें।"""
    },

    "Leaf_smut": {
        "English": """Use balanced fertilizer and maintain potassium level. 
Avoid excess nitrogen and keep field clean.""",

        "Telugu": """సమతుల్యమైన ఎరువులు వాడండి మరియు పొటాషియం స్థాయిని నిలబెట్టండి. 
నైట్రోజన్ ఎక్కువగా వాడకండి మరియు పొలం శుభ్రంగా ఉంచండి.""",

        "Hindi": """संतुलित उर्वरक का उपयोग करें और पोटाश स्तर बनाए रखें। 
अधिक नाइट्रोजन से बचें और खेत को साफ रखें।"""
    }
}

# Title
st.title(ui_text[language]["title"])

# Upload
uploaded_file = st.file_uploader(
    ui_text[language]["upload"],
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader(ui_text[language]["image"])
    st.image(image, width=300)

    # Preprocess
    img = image.resize(IMG_SIZE)
    arr = np.array(img)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    # Prediction
    prediction = model.predict(arr)
    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    # IMPORTANT FIX
    class_key = predicted_class.strip().replace(" ", "_")

    confidence = np.max(prediction) * 100

    # Get data
    disease_name = disease_translation.get(class_key, {}).get(language, predicted_class)
    fertilizer_text = fertilizer.get(class_key, {}).get(language)

    # Fallback (never show empty)
    if fertilizer_text is None:
        fertilizer_text = {
            "English": "Use balanced fertilizer and maintain field properly.",
            "Telugu": "సమతుల్యమైన ఎరువులు వాడండి మరియు పొలం సరిగా ఉంచండి.",
            "Hindi": "संतुलित उर्वरक का उपयोग करें और खेत को सही रखें।"
        }[language]

    # Display
    st.subheader("🌱 " + ui_text[language]["disease"])
    st.success(disease_name)

    st.subheader("🎯 " + ui_text[language]["confidence"])
    st.info(f"{confidence:.2f}%")
    st.write(ui_text[language]["confidence_info"])

    st.subheader("🌾 " + ui_text[language]["fertilizer"])
    st.success(fertilizer_text)

else:
    st.info(ui_text[language]["wait"])