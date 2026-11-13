import io
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.cm as cm
from PIL import Image, ImageChops, ImageEnhance


st.set_page_config(page_title="Image Forgery Detection", layout="wide")
st.title("🔍 Digital Image Forgery Detection (ELA + CNN + FFT)")
st.markdown(
    "Upload an image to analyze **JPEG compression residuals (ELA)**, "
    "**frequency-domain artifacts (2D FFT)**, and **Grad-CAM visual explanations**."
)


@st.cache_resource
def load_forgery_model():
    return tf.keras.models.load_model("best_model.h5")

model = load_forgery_model()


def compute_ela(pil_image, quality=90):
    original = pil_image.convert("RGB")
    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)

    ela_image = ImageChops.difference(original, resaved)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image


def compute_fft(ela_pil):
    gray = np.array(ela_pil.convert("L"), dtype=np.float32)
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = 20 * np.log(np.abs(f_shift) + 1e-8)
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    colored_fft = cm.viridis(magnitude)[:, :, :3]  # Match notebook viridis colormap
    return Image.fromarray((colored_fft * 255).astype(np.uint8))


def compute_gradcam(model, img_array, last_conv_layer_name="conv2d_2"):
    x = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs = None
        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                conv_outputs = x
                tape.watch(conv_outputs)
        loss = x[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(original_pil, heatmap, alpha=0.45):
    orig_resized = original_pil.convert("RGB").resize((128, 128))
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = cm.jet
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = Image.fromarray(np.uint8(jet_colors[heatmap_uint8] * 255)).resize((128, 128))
    blended = Image.blend(orig_resized, jet_heatmap, alpha=alpha)
    return blended


st.sidebar.header("Settings")
ela_quality = st.sidebar.slider("ELA Re-save Quality", min_value=70, max_value=98, value=90)
uploaded_file = st.file_uploader("Upload an Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    orig_img = Image.open(uploaded_file)

    # Preprocess & Predict
    ela_img = compute_ela(orig_img, quality=ela_quality)
    ela_resized = ela_img.resize((128, 128))
    input_tensor = np.expand_dims(np.array(ela_resized, dtype=np.float32) / 255.0, axis=0)

    prob = float(model.predict(input_tensor, verbose=0)[0][0])
    is_tampered = prob >= 0.5
    label = "TAMPERED (Forged)" if is_tampered else "AUTHENTIC (Original)"
    confidence = prob * 100 if is_tampered else (1 - prob) * 100

    # Display Verdict Banner
    st.divider()
    if is_tampered:
        st.error(f"### Prediction: **{label}** — Confidence: **{confidence:.2f}%** (Raw Score: {prob:.4f})")
    else:
        st.success(f"### Prediction: **{label}** — Confidence: **{confidence:.2f}%** (Raw Score: {prob:.4f})")

    # Generate FFT & Grad-CAM
    fft_img = compute_fft(ela_resized)
    heatmap = compute_gradcam(model, input_tensor, last_conv_layer_name="conv2d_2")
    gradcam_img = overlay_gradcam(orig_img, heatmap)

    # 4-Column Visual Comparison
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(orig_img.resize((256, 256)), caption="1. Original Image", use_container_width=True)
    with col2:
        st.image(ela_img.resize((256, 256)), caption=f"2. ELA Map (Q={ela_quality})", use_container_width=True)
    with col3:
        st.image(fft_img.resize((256, 256)), caption="3. 2D FFT Spectrum", use_container_width=True)
    with col4:
        st.image(gradcam_img.resize((256, 256)), caption="4. Grad-CAM Focus", use_container_width=True)