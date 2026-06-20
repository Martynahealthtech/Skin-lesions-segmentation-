import streamlit as st

st.set_page_config(
    page_title="Skin Lesion Segmentation",
    page_icon="🧬",
    layout="wide"
)

st.title("Skin Lesion Segmentation using Deep Learning")

st.write("""
This app demonstrates a deep learning project for skin lesion segmentation.

Upload a skin lesion image and the app will display it. 
The model prediction will be added in the next version.
""")

st.warning(
    "This app is for educational and portfolio demonstration purposes only. "
    "It is not a medical device and should not be used for diagnosis."
)

uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.subheader("Uploaded image")
    st.image(uploaded_file, use_container_width=True)
    st.success("Image uploaded successfully. Model prediction will be added next.")
else:
    st.info("Upload an image to test the app interface.")
