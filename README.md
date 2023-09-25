# Exploring the Dataset

## Team Name
TODO - Update your team name tagging all contributors

TODO - Add the following badges
- PyTorch
- PyTorch Lightning
- Python
- Weights and Biases

These badges can be found with a simple Google search
## Assignment Overview
This homework is designed to help your team with the final project. The objective is to use dimensionality reduction of the images to identify patterns using traditional unsupervised methods. We will then visualize the reduced dimensionality data to intuitively assess the extent of similarity and differences between target labels. This step will inform your model design choices for your team project. For example, are local image features more dominant in discriminating between labels or are global features more important. You are encouraged to use this assignment as content towards your stretch goal objectives.

In this assignment we will:

- [ ] Build a PyTorch Lighting DataModule to encapsulate training, validation, and testing dataloaders utilizing the custom PyTorch Dataset from the previous homework.
- [ ] Build an unsupervised learning **baseline model** to identify patterns in the BPS Microscopy data.
  - [ ] Reduce the dimensionality of the data to a vector representation using Principal Component Analysis (PCA).
  - [ ] Feed the vector representation to a T-distributed stochastic neighbor embedding (TSNE) model.
  - [ ] Visualize the output and associated labels in two and three dimensions.

### Files to Adapt from Last Assignment
- `src/dataset/bps_dataset.py`
- `src/augmentation.py`
- `src/data_utils.py`

### Files to Work On
- `src/dataset/bps_datamodule.py`
- `src/unsupervised/pca_tsne.py`
- `src/vis_utils.py`

### Images to Save and Inspect
- `tsne_direct_4hr_Gy_hi.png`
- `tsne_pca_4hr_Gy_hi.png`
- `tsne_resnet_2d_4hr_Gy_hi.png`
- `tsne_resnet_3d_4hr_Gy_hi.png`
- `Cell_Gallery_from_1D_Array.png`
- `PCA_Cell_Gallery_from_1D_Array.png`


## PyTorch Lightning DataModule:

In the previous assignment you built a custom Dataset class with PyTorch callable transformations. In this assignment you will build upon the previous assignment by using the `LightningDataModule` to manage your training, validation, and testing dataloaders, as well as steps that may be required for transformations. By doing so, you alleviate custom code that may be present in your driver function and make your dataset splits reusable across different models you may test within your team.

To create a DataModule you must define the following methods to create your training, validation, and test dataloaders.
- `prepare_data`: This function is called within a single process on CPU enabling you to add logic to download the data and tokenize it once. In this function we will be calling our
  `save_tiffs_local_from_s3` function from `src/data_utils.py`
- `setup`: This function expects a `stage` string argument that calls the constructor of your PyTorch DataSet class with all the necessary input parameters. You can also perfrom training, validation, and testing splits on the full dataset if you like.
- `train_dataloader`: This function generates the training set dataloader using the dataset you defined in `setup`.
- `val_dataloader`: This function generates the validation set dataloader using the dataset you defined in `setup`.
- `test_dataloader`: This function generates the testing set dataloader using the dataset you defined in `setup`.

## Dimensionality Reduction Techniques for Image Feature Embedding Intuition
### Principal Component Analysis (PCA)
PCA is a popular dimensionality reduction technique in computer vision. It can be used to reduce the number of features in an image dataset without losing too much information. This can be useful for tasks such as object recognition and image classification. PCA can also be used to find the most important features in an image dataset. This can be done by calculating the eigenvalues of the covariance matrix of the dataset. The eigenvalues represent the variance of each feature. The features with the largest eigenvalues are the most important features. In other words, PCA can be used to compress images by reducing the number of pixels in each image. This can be done by finding the principal components of the image and then projecting the image onto the subspace spanned by the first few principal components.

Since the most important features have the most variance, PCA focuses more on preserving global trends in the data and less on preserving local relationships between specific points.

We will visualize the feature embeddings found by PCA by fitting the compressed representation of the image to t-SNE plot which will allow us to visualize our images as points on a 2D and 3D plane which we can then color based on the target label of particle type.


## Visualizing High Dimensional Data in Lower Dimensions
### T-distributed Stochastic Neighbor Embedding (t-SNE) 
T-SNE is a non-linear dimensionality reduction technique that is commonly used for visualization of high-dimensional data. It was developed by Laurens van der Maaten and Geoffrey Hinton in 2008.

t-SNE works by finding a low-dimensional representation of the data that preserves the local structure of the high-dimensional data. This means that points that are close together in the high-dimensional space will also be close together in the low-dimensional space. In other words, t-SNE preserves local patterns.

t-SNE is a popular choice for visualizing high-dimensional data in computer vision because it can be used to visualize data that is too high-dimensional to be visualized directly. For example, t-SNE can be used to visualize the features of images, such as the color and texture of the pixels.

We will be using t-SNE to visualize the results of our raw images, our PCA data representation, as well as our pre-trained deep learning representation for both particle_type labels.


Based on the images what features are important in differentiating Fe tracks from X-ray tracks? Global or local features? Which dimensionality reduction techniques seem better suited for differentiating between particle types at the high Gy dose after 4 hours?

## IMPORTANT: The Data
The data is not provided in the repository because 7,091 images in a github repo is a verifiable way to be labeled as an ML newb. While we had to compromise and provide you with two csv files for training and validation, storing tiffs is a little too crazy for us.

To download the data use the `main` function provided in the `bps_datamodule.py` source code. Make sure that your Lightning Data
Module `prepare_data()` function is implemented. While running this, the files will save to the `data/processed` directory. We have added this directory to the `.gitignore` so you can also avoid rookie mistakes. Make sure that after you run it once, re-comment `prepare_data()` so you don't accidentally keep downloading images everytime you test `bps_datamodule.py`.

## Considerations
This assignment relies on visualizations instead of test cases for credit. Many function
definitions have been provided to you for your use in the context of the driver code in 
each file's respective `main`. Please treat `main` as your test cases to build confidence in this assignment. Feel free to adapt and understand the codebase to use for
your final project as needed.

## NOTE
- It is required that you add your names and badges to your readme.
- The initial code will not necessarily run. You will have to write the necessary code.
- Commit all changes as you develop the code in your individual private repo. Please provide descriptive commit messages and push from local to your repository. If you do not stage, commit, and push git classroom will not receive your code at all.
- Make sure your last push is before the deadline. Your last push will be considered as your final submission.
- There is no partial credit for code that does not run.
- Use the driver code in the `main` functions of each file to work on as a means of testing your code.
- If you need to be considered for partial grade for any reason, then message the staff on canvas before the deadline. Late email requests may not be considered.

## References:
- [Image t-SNE](https://notebook.community/ml4a/ml4a-guides/notebooks/image-tsne)
- [Faces Dataset Decomposition Tutorial](https://github.com/olekscode/Examples-PCA-tSNE/blob/master/Python/Faces%20Dataset%20Decomposition.ipynb) 
- [Using t-SNE for Data Visualisation](https://medium.com/analytics-vidhya/using-t-sne-for-data-visualisation-8a83f46fbad3)

## Authors:
- Nadia Ahmed (@nadia-eecs)
- Jacob Campbell (@campjake)
