import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Skin Lesion Segmentation",
    page_icon="🧬",
    layout="wide"
)

st.title("Skin Lesion Segmentation using U-Net")

st.write("""
This app demonstrates a deep learning model for skin lesion segmentation. 
Upload a skin lesion image and the model will generate a predicted binary mask showing the lesion area.
""")

st.warning(
    "This app is for educational and portfolio demonstration purposes only. "
    "It is not a medical device and should not be used for diagnosis."
)


# -----------------------------
# U-Net model architecture
# -----------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.encoder1 = self.contracting_block(in_channels, 64)
        self.encoder2 = self.contracting_block(64, 128)
        self.encoder3 = self.contracting_block(128, 256)

        # Bottleneck
        self.bottleneck = self.contracting_block(256, 512)

        # Decoder
        self.decoder3 = self.expansive_block(512, 256)
        self.decoder2 = self.expansive_block(512, 128)
        self.decoder1 = self.expansive_block(256, 64)

        # Final output layer
        self.final_output = nn.Conv2d(128, out_channels, kernel_size=1)

    def contracting_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        return block

    def expansive_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        return block

    def forward(self, x):
        encoder1 = self.encoder1(x)
        encoder2 = self.encoder2(nn.MaxPool2d(2)(encoder1))
        encoder3 = self.encoder3(nn.MaxPool2d(2)(encoder2))

        bottleneck = self.bottleneck(nn.MaxPool2d(2)(encoder3))

        decoder3 = self.decoder3(bottleneck)
        decoder3 = torch.cat([encoder3, decoder3], dim=1)

        decoder2 = self.decoder2(decoder3)
        decoder2 = torch.cat([encoder2, decoder2], dim=1)

        decoder1 = self.decoder1(decoder2)
        decoder1 = torch.cat([encoder1, decoder1], dim=1)

        out = self.final_output(decoder1)

        return out


# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    device = torch.device("cpu")

    model = UNet(in_channels=3, out_channels=1)

    checkpoint = torch.load(
        "best_model_v2_unet.pth",
        map_location=device
    )

    # Handles common save formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    model.to(device)
    model.eval()

    return model


# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


# -----------------------------
# Prediction
# -----------------------------
def predict_mask(model, image, threshold=0.5):
    image_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(image_tensor)
        probability_mask = torch.sigmoid(output)
        probability_mask = probability_mask.squeeze().cpu().numpy()

    binary_mask = (probability_mask > threshold).astype(np.uint8)

    return probability_mask, binary_mask


# -----------------------------
# Overlay function
# -----------------------------
def create_overlay(image, binary_mask):
    image_resized = image.resize((256, 256))
    image_array = np.array(image_resized).astype(np.uint8)

    mask_rgb = np.zeros_like(image_array)
    mask_rgb[:, :, 0] = binary_mask * 255

    overlay = (0.7 * image_array + 0.3 * mask_rgb).astype(np.uint8)

    return overlay


# -----------------------------
# App interface
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)

threshold = st.slider(
    "Prediction threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    try:
        model = load_model()
        probability_mask, binary_mask = predict_mask(model, image, threshold)
        overlay = create_overlay(image, binary_mask)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Original image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Predicted mask")
            st.image(binary_mask * 255, use_container_width=True, clamp=True)

        with col3:
            st.subheader("Overlay")
            st.image(overlay, use_container_width=True)

        st.success("Segmentation completed successfully.")

        with st.expander("Model details"):
            st.write("""
            **Model:** U-Net style encoder-decoder convolutional neural network  
            **Task:** Binary skin lesion segmentation  
            **Input size:** 256 × 256 RGB image  
            **Output:** Predicted binary lesion mask  
            """)

    except Exception as e:
        st.error("The app interface works, but the model could not be loaded or used yet.")
        st.write("Error details:")
        st.code(str(e))

else:
    st.info("Upload an image to generate a predicted lesion mask.")
