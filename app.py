import streamlit as st
from PIL import Image
from vector_utility import Embedding_Creation
from time import time
st.title('Image Search Application')

# Display a loading spinner while initializing the model and index
with st.spinner('Loading model and initializing index...'):
    embeddings_processing = Embedding_Creation()

# Section for entering a path and pressing enter
st.sidebar.title('Enter Path')
path_input = st.sidebar.text_input('Enter path:', '')

if st.sidebar.button('Enter'):
    embeddings_processing._dataset_path = path_input
    st.write(f'Path entered: {path_input}')
    progress_bar = st.progress(0)
    embeddings_processing.process_dataset(progress_bar)
    progress_bar.empty()



# Section for uploading an image and pressing search button
st.sidebar.title('Upload Image')
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if st.sidebar.button('Search'):
    start = time()
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        # Show another spinner while searching for similar images
        with st.spinner('Searching for similar images...'):
            similar_images, D, I = embeddings_processing.search_index(image)
        if similar_images:
            st.write('Showing similar images:')
            for img_path in similar_images:
                st.image(str(img_path), width=150)
        else:
            st.write("No similar images found.")
    st.write(f"total time {round((time() - start), 2)} seconds")