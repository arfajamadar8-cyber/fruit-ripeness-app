import streamlit as st
from PIL import Image
import numpy as np
import h5py
from tensorflow.keras.models import load_model

# Load model
model = load_model("fruit_model.h5", compile=False)

# Classes (IMPORTANT - same order as training)
classes = ['overripe', 'ripe', 'unripe']

# 🎨 Stylish Title
st.markdown("""
<h1 style='text-align: center; color: green;'>
🍎 AI Fruit Ripeness Detector
</h1>
""", unsafe_allow_html=True)

st.write("### Upload or Capture a fruit image to detect ripeness")

# 🔘 Input option
option = st.radio("Choose Input Method:", ["Upload Image", "Use Camera"])

file = None

if option == "Upload Image":
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
else:
    file = st.camera_input("Take a Photo")

# If image provided
if file:
    img = Image.open(file)
    st.image(img, caption="Selected Image", use_column_width=True)

    # Preprocess
    img = img.resize((100, 100))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    pred = model.predict(img_array)
    class_index = np.argmax(pred)
    result = classes[class_index]
    confidence = np.max(pred)

    # 🎯 Result Display
    st.subheader("🔍 Prediction Result")

    if result == "ripe":
        st.success(f"🍌 Ripe - Ready to eat 😋 ({confidence*100:.2f}%)")
        st.info("👉 Suggestion: Perfect for eating now!")
    elif result == "unripe":
        st.warning(f"🥭 Unripe - Not ready ⏳ ({confidence*100:.2f}%)")
        st.info("👉 Suggestion: Keep for 2–3 days")
    else:
        st.error(f"🍎 Overripe - Use quickly ⚠️ ({confidence*100:.2f}%)")
        st.info("👉 Suggestion: Use for juice or compost")

    # 📊 Confidence Chart
    st.subheader("📊 Confidence Levels")
    st.bar_chart(pred[0])

    # 🧠 Detailed breakdown
    st.subheader("🧠 Model Analysis")
    for i, val in enumerate(pred[0]):
        st.write(f"{classes[i]}: {val*100:.2f}%")
