# BLA BLE BLA RED O
import streamlit as st
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import pandas as pd
from scipy.interpolate import PchipInterpolator
import os
import requests
import matplotlib.patches as mpatches


def download_from_google_drive(file_id, dest_path):
    URL = "https://docs.google.com/uc?export=download"

    with requests.Session() as session:
        response = session.get(URL, params={"id": file_id}, stream=True)
        token = get_confirm_token(response)

        if token:
            response = session.get(
                URL, params={"id": file_id, "confirm": token}, stream=True
            )

        save_response_content(response, dest_path)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


@st.cache_resource
def load_trained_model(model_name):
    model_info = MODEL_OPTIONS[model_name]
    model_path = model_info["filename"]

    if not os.path.exists(model_path):
        with st.spinner(f"Downloading {model_name} model..."):
            download_from_google_drive(model_info["file_id"], model_path)
    return load_model(model_path)


st.set_page_config(layout="wide", page_title="Image Adjuster")

st.title("Image Augmentation & Analysis App")

st.sidebar.subheader("IDEA")
st.sidebar.markdown(
    """
**Problem**\n
Can adjusting image inputs to match training conditions improve prediction accuracy?

**Description**
* The left image is untouched 
* The center image is controlled by you!
* The right image is automatically pushed into the relevant range
    (brightness/saturation/contrast)

"""
)


MODEL_OPTIONS = {
    "Xception": {
        "file_id": "1Fz2RuXvM54Ym3QCanvMySKME0Y4-GgEb",
        # https://drive.google.com/file/d/1Fz2RuXvM54Ym3QCanvMySKME0Y4-GgEb/view?usp=sharing
        "filename": "xception_model.h5",
    },
    "Custom CNN": {
        "file_id": "1Epj1vPIFnj8_snHTmUvO6msHogRy737U",
        # https://drive.google.com/file/d/1Epj1vPIFnj8_snHTmUvO6msHogRy737U/view?usp=sharing
        "filename": "custom_cnn_unified_model.h5",
    },
    # "RandomForest": {
    #     "file_id": "15g2u9wFJKLY2cvXkUWtqCSRhl3ekN6Eo",
    #     # https://drive.google.com/file/d/15g2u9wFJKLY2cvXkUWtqCSRhl3ekN6Eo/view?usp=sharing
    #     "filename": "random_forest_model.h5",
    # },
}

st.sidebar.subheader("Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose a model:", list(MODEL_OPTIONS.keys())
)

model = load_trained_model(selected_model_name)

image_source = st.sidebar.radio("Choose image source:", ["Example", "Upload"])

example_options = {
    "DCIS 1": "images/DCIS_1.png",
    "DCIS 2": "images/DCIS_2.png",
    "Invasive Tumor": "images/Invasive_Tumor.png",
    "Proliferative Invasive Tumor": "images/Prolif_Tumor.png",
}

selected_example = None
uploaded_file = None

if image_source == "Upload":
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"]
    )
else:
    selected_example = st.sidebar.selectbox(
        "Select an example image:", list(example_options.keys())
    )

image = None
if image_source == "Upload" and uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
elif image_source == "Example" and selected_example:
    image_path = example_options[selected_example]
    image = Image.open(image_path).convert("RGB")


def calculate_image_stats(image):
    """Calculates brightness, saturation, and contrast for an image."""
    image_np = np.array(image.convert("RGB")) / 255.0
    hsv = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)

    brightness = np.mean(hsv[:, :, 2])
    saturation = np.mean(hsv[:, :, 1])
    contrast = np.std(
        cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    )

    return brightness, saturation, contrast


def plot_metric_chart(metric_name, data_dict):
    df = pd.DataFrame(data_dict)
    df[["start", "end"]] = df["Range"].str.split("–", expand=True).astype(int)
    df["mid"] = (df["start"] + df["end"]) / 2

    combined_max = df[["Xception", "Custom CNN", "RandomForest"]].max(axis=1)
    best_idx = combined_max.idxmax()
    best_start = df.loc[best_idx, "start"]
    best_end = df.loc[best_idx, "end"]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    colors = {"Xception": "cyan", "Custom CNN": "orange", "RandomForest": "lime"}

    for model in ["Xception", "Custom CNN", "RandomForest"]:
        x = df["mid"].to_numpy()
        y = df[model].to_numpy()
        interpolator = PchipInterpolator(x, y)
        x_smooth = np.linspace(x.min(), x.max(), 300)
        y_smooth = interpolator(x_smooth)

        ax.plot(x_smooth, y_smooth, label=model, linewidth=2, color=colors[model])
        ax.scatter(x, y, color=colors[model], edgecolor="white", zorder=5, label=None)

    ax.axvspan(best_start, best_end, color="purple", alpha=0.3, label=None)

    best_patch = mpatches.Patch(
        color="purple", alpha=0.3, label=f"Best Range: {best_start}–{best_end}"
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(best_patch)
    labels.append(f"Best Range: {best_start}–{best_end}")
    ax.legend(handles, labels, fontsize=10)

    ax.set_xlim(0, 127 if metric_name.lower() == "contrast" else 255)
    ax.set_xlabel(metric_name, fontsize=12, color="white")
    ax.set_ylabel("Accuracy", fontsize=12, color="white")
    ax.set_title(
        f"Model Accuracy Across {metric_name} Range", fontsize=16, color="white"
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(colors="white")

    st.pyplot(fig)


def auto_adjust_image(
    image,
    feature="All",
    brightness_range=(153, 178),
    saturation_range=(51, 76),
    contrast_range=(26, 38),
):
    brightness, saturation, contrast = calculate_image_stats(image)

    def get_factor(current, range):
        mean = np.mean(range)
        if mean > 0:
            return mean / current
        else:
            return 1.0

    def clamp(f, min_f=0.5, max_f=1.5):
        # return max(min(f, max_f), min_f)
        return f

    brightness_factor = 1.0
    saturation_factor = 1.0
    contrast_factor = 1.0

    if feature in ("Brightness", "All"):
        brightness_factor = clamp(get_factor(brightness, brightness_range))
    if feature in ("Saturation", "All"):
        saturation_factor = clamp(get_factor(saturation, saturation_range))
    if feature in ("Contrast", "All"):
        contrast_factor = clamp(get_factor(contrast, contrast_range))

    image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)
    image = ImageEnhance.Color(image).enhance(saturation_factor)

    return image


if image:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
        st.markdown("**Original Image**")
        orig_bright, orig_sat, orig_con = calculate_image_stats(image)
        st.write(f"Brightness: {orig_bright:.2f}")
        st.write(f"Saturation: {orig_sat:.2f}")
        st.write(f"Contrast: {orig_con:.2f}")

    st.sidebar.subheader("Augmentation Options")

    feature_choice = st.sidebar.selectbox(
        "Select a feature to adjust (for both manual and auto adjust):",
        ["Brightness", "Saturation", "Contrast", "All"],
        key="selected_metric",
    )

    brightness_factor = 1.0
    contrast_factor = 1.0
    saturation_factor = 1.0

    if feature_choice in ("Brightness", "All"):
        brightness_factor = st.sidebar.slider("Brightness", 0.0, 3.0, 1.0)

    if feature_choice in ("Saturation", "All"):
        saturation_factor = st.sidebar.slider("Saturation", 0.0, 3.0, 1.0)

    if feature_choice in ("Contrast", "All"):
        contrast_factor = st.sidebar.slider("Contrast", 0.0, 3.0, 1.0)

    augmented = image
    augmented = ImageEnhance.Brightness(augmented).enhance(brightness_factor)
    augmented = ImageEnhance.Contrast(augmented).enhance(contrast_factor)
    augmented = ImageEnhance.Color(augmented).enhance(saturation_factor)

    adjusted = auto_adjust_image(image, feature=feature_choice)

    with col2:
        st.image(augmented, caption="Augmented Image", use_container_width=True)
        aug_bright, aug_sat, aug_con = calculate_image_stats(augmented)
        st.markdown("**Augmented Image**")
        st.write(f"Brightness: {aug_bright:.2f}")
        st.write(f"Saturation: {aug_sat:.2f}")
        st.write(f"Contrast: {aug_con:.2f}")

    orig_bright, orig_sat, orig_con = calculate_image_stats(image)
    aug_bright, aug_sat, aug_con = calculate_image_stats(augmented)
    adj_bright, adj_sat, adj_con = calculate_image_stats(adjusted)

    with col3:
        st.image(adjusted, caption="Auto Adjusted Image", use_container_width=True)
        st.markdown("**Auto Augmented**")
        st.write(f"Brightness: {adj_bright:.2f}")
        st.write(f"Saturation: {adj_sat:.2f}")
        st.write(f"Contrast: {adj_con:.2f}")

    st.markdown("---")
    st.markdown("### Run Prediction")

    prediction_placeholder = st.container()

    with st.form("predict_form"):
        submitted = st.form_submit_button("Predict", use_container_width=True)

    with prediction_placeholder:
        st.subheader(f"{selected_model_name} - Model Predictions")
        pred_cols = st.columns(3)

        if submitted:
            try:
                if model:

                    def preprocess(img):
                        arr = np.array(img.resize((100, 100))) / 255.0
                        return np.expand_dims(arr, axis=0)

                    img_input = preprocess(image)
                    aug_input = preprocess(augmented)
                    adj_input = preprocess(adjusted)

                    img_pred = model.predict(img_input)
                    aug_pred = model.predict(aug_input)
                    adj_pred = model.predict(adj_input)

                    img_label = np.argmax(img_pred, axis=1)[0]
                    aug_label = np.argmax(aug_pred, axis=1)[0]
                    adj_label = np.argmax(adj_pred, axis=1)[0]

                    index_to_label = {
                        0: "DCIS_1",
                        1: "DCIS_2",
                        2: "Invasive_Tumor",
                        3: "Prolif_Invasive_Tumor",
                    }

                    labels = [img_label, aug_label, adj_label]
                    for col, title, label in zip(
                        pred_cols, ["Original", "Augmented", "Auto Adjusted"], labels
                    ):
                        with col:
                            st.markdown(f"#### {title}")
                            st.success(f"**Prediction:** {index_to_label[label]}")
                else:
                    st.warning("No model found.")
            except Exception as e:
                st.error("Something went wrong during prediction.")
                st.text(str(e))
        else:
            for col, title in zip(
                pred_cols, ["Original", "Augmented", "Auto Adjusted"]
            ):
                with col:
                    st.markdown(f"#### {title}")
                    st.info("Awaiting prediction...")

        brightness_data = {
            "Range": [
                "0–26",
                "26–51",
                "51–76",
                "76–102",
                "102–128",
                "128–153",
                "153–178",
                "178–204",
                "204–230",
                "230–255",
            ],
            "Xception": [
                0.299950,
                0.329120,
                0.423164,
                0.489199,
                0.520523,
                0.541884,
                0.544601,
                0.530387,
                0.436752,
                0.343688,
            ],
            "Custom CNN": [
                0.196294,
                0.265914,
                0.266787,
                0.270927,
                0.302661,
                0.420729,
                0.545070,
                0.514601,
                0.364530,
                0.292796,
            ],
            "RandomForest": [
                0.244867,
                0.251016,
                0.250563,
                0.242574,
                0.248985,
                0.259347,
                0.444601,
                0.316496,
                0.293590,
                0.256444,
            ],
        }

    saturation_data = {
        "Range": [
            "0–26",
            "26–51",
            "51–76",
            "76–102",
            "102–128",
            "128–153",
            "153–178",
            "178–204",
            "204–230",
            "230–255",
        ],
        "Xception": [
            0.505130,
            0.509309,
            0.538218,
            0.538197,
            0.508876,
            0.439278,
            0.438820,
            0.418295,
            0.406023,
            0.374892,
        ],
        "Custom CNN": [
            0.255884,
            0.299202,
            0.407525,
            0.372532,
            0.341375,
            0.339658,
            0.305905,
            0.302421,
            0.295356,
            0.282145,
        ],
        "RandomForest": [
            0.314424,
            0.344858,
            0.478416,
            0.439914,
            0.340919,
            0.330573,
            0.314640,
            0.310993,
            0.305881,
            0.289583,
        ],
    }

    contrast_data = {
        "Range": [
            "0–13",
            "13–26",
            "26–38",
            "38–51",
            "51–64",
            "64–76",
            "76–89",
            "89–102",
            "102–115",
            "115–128",
        ],
        "Xception": [
            0.306745,
            0.480054,
            0.543546,
            0.532003,
            0.512400,
            0.498200,
            0.491100,
            0.470000,
            0.460000,
            0.450000,
        ],
        "Custom CNN": [
            0.259970,
            0.266999,
            0.530147,
            0.287274,
            0.246400,
            0.240000,
            0.238000,
            0.237000,
            0.235000,
            0.232000,
        ],
        "RandomForest": [
            0.262925,
            0.320943,
            0.476552,
            0.376130,
            0.314000,
            0.290000,
            0.285000,
            0.278000,
            0.271000,
            0.265000,
        ],
    }

    if (
        "last_metric" not in st.session_state
        or feature_choice != st.session_state.last_metric
    ):
        st.session_state.last_metric = feature_choice

    if feature_choice in ["Brightness", "All"]:
        plot_metric_chart("Brightness", brightness_data)
    if feature_choice in ["Saturation", "All"]:
        plot_metric_chart("Saturation", saturation_data)
    if feature_choice in ["Contrast", "All"]:
        plot_metric_chart("Contrast", contrast_data)
