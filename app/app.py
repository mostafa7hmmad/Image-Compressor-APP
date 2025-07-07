import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Title
st.title("KMeans Image Compression App")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and resize the image
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((128, 128))  # Resize to 128x128
    img_array = np.array(img)     # Shape: (128, 128, 3)

    # Normalize (rescale) to 1.0/255
    rescaled_img = img_array / 255.0

    # Flatten the image to (128*128, 3)
    pixel_data = rescaled_img.reshape(-1, 3)

    # Apply KMeans
    n_clusters = st.slider("Choose number of clusters", 2, 20, 8)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(pixel_data)
    centroids = km.cluster_centers_

    # Reconstruct the compressed image
    compressed_img = centroids[labels].reshape(128, 128, 3)

    # Plot original and compressed images
    st.subheader("Original vs Compressed Image")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(rescaled_img)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(compressed_img)
    ax[1].set_title(f"Compressed Image ({n_clusters} colors)")
    ax[1].axis("off")

    st.pyplot(fig)
