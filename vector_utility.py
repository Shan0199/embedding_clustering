# vector_utility.py
import os
import faiss
import torch
import joblib
import pathlib
import imagehash
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTModel


class Embedding_Creation:
    """
    A class to create and manage embeddings of images using a Vision Transformer (ViT) model.

    Attributes:
        dataset_path (str): The path to the dataset containing images.
        index_path (str): The path to save or load the FAISS index.
        image_mapping_path (str): The path to save or load the mapping of image IDs to image names.
        image_hashes_path (str): The path to save or load the set of image hashes.
        dim (int): The dimensionality of the feature vectors.
    """

    def __init__(self, dataset_path="", index_path='./faiss_index.idx', image_mapping_path='./image_mapping.npy', image_hashes_path='image_hashes.npy', dim=768) -> None:
        self.feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k') 
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.index_path = index_path
        self.index = faiss.read_index(index_path) if os.path.exists(index_path) else faiss.IndexFlatL2(dim) 
        self.image_mapping = joblib.load(image_mapping_path) if os.path.exists(image_mapping_path) else dict()
        self.image_hashes = set(np.load(image_hashes_path, allow_pickle=True).tolist()) if os.path.exists(image_hashes_path) else set()
        self._dataset_path = dataset_path
        self.image_mapping_path = image_mapping_path
        self.image_hashes_path = image_hashes_path

    @property
    def dataset_path(self):
        return self._dataset_path

    dataset_path.setter
    def dataset_path_setter(self,path):
        self._dataset_path = path
    
    
    def get_feature_vector(self,images):
        """
        Get the feature vector for a given image.

        Args:
            images (PIL.Image): The input image.

        Returns:
            numpy.ndarray: The feature vector of the input image.
        """
        inputs = self.feature_extractor(images, return_tensors="pt")
        outputs = self.model(**inputs)
        feature_vector = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        return feature_vector

    def save_embeddings(self,image,image_name):
        """
        Save embeddings of an image to the index.

        Args:
            image (PIL.Image): The input image.
            image_name (str): The name of the image.
        """
        embeddings = self.get_feature_vector(image)
        self.index.add(embeddings)

        image_id = len(self.image_mapping)
        self.image_mapping[image_id] = image_name
        
        return None
        
    def process_dataset(self, progress_bar):
        """
        Process a dataset of images to create embeddings.

        Args:
            path_to_dataset (pathlib.Path): The path to the dataset directory.
        """
        try:
            files_list = list(pathlib.Path(self._dataset_path).iterdir())
            for index,files in enumerate(files_list):
                input_image = Image.open(files)
                image_hash = str(imagehash.average_hash(input_image))
                if image_hash in self.image_hashes:
                    continue
                self.image_hashes.add(image_hash)
                self.save_embeddings(input_image, files)
                progress_bar.progress(int(100 * index / len(files_list)), text=f"Processing! {index} images processed out of {len(files_list)}")
            
        except Exception as e:
            print(e)
        
        finally:

            faiss.write_index(self.index, self.index_path)
            np.save(self.image_hashes_path, list(self.image_hashes))
            joblib.dump(self.image_mapping,self.image_mapping_path)

    
    def search_index(self, input_image, k=5):
        """
        Search the index for similar images to the given input image.

        Args:
            input_image (str): The path to the input image.
            k (int, optional): The number of similar images to retrieve. Defaults to 5.

        Returns:
            tuple: A tuple containing lists of similar image names, distances, and indices.
        """
        if self.image_mapping and self.image_hashes:
            embeddings = self.get_feature_vector(input_image)
            actual_k = min(k + 1, len(self.image_mapping))
            D, I = self.index.search(embeddings, actual_k)
            images = [self.image_mapping[i] for i in I[0] if i != len(self.image_mapping) - 1][:k]
            return images, D, I
        else:
            raise "Dataset is empty"




if __name__=="__main__":
    pass