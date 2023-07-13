import matplotlib.pyplot as plt
import torch
import typing as typing
import numpy as np
from torchvision import utils
import pandas as pd


# Faces Dataset Decomposition Tutorial https://github.com/olekscode/Examples-PCA-tSNE/blob/master/Python/Faces%20Dataset%20Decomposition.ipynb
def plot_gallery_from_2D(title, images, n_row, n_col, img_shape: tuple):
    """Helper function to plot a gallery of portraits from 2D data images"""
    raise NotImplementedError

def plot_gallery_from_1D(title, images, n_row, n_col, img_shape: tuple):
    """Helper function to plot a gallery of portraits from 1D array"""
    raise NotImplementedError

def show_image_and_label(image: torch.Tensor, label: str):
    """Show an image with a label"""
    raise NotImplementedError
    
def plot_dataloader_batch(img_batch, label_batch):
    """Plot a batch of images with labels"""
    raise NotImplementedError

def show_label_batch(images_batch: torch.Tensor, label_batch: str):
    """Show image with label for a batch of samples."""
    raise NotImplementedError

def plot_2D_scatter_plot(cps_df: pd.DataFrame,
                         scatter_fname: str) -> None:
    """
    Create a 2D scatter plot of the t-SNE components when the number of components is 2.
    https://medium.com/analytics-vidhya/using-t-sne-for-data-visualisation-8a83f46fbad3

    Args:
        cps_df: A dataframe that contains the lower dimensional t-SNE components and the labels for each image.
    
    Returns:
        None
    """
    raise NotImplementedError

def plot_3D_scatter_plot(cps_df: pd.DataFrame,
                         scatter_fname: str) -> None:
    """
    Create a 2D scatter plot of the t-SNE components when the number of components is 2.
    https://medium.com/analytics-vidhya/using-t-sne-for-data-visualisation-8a83f46fbad3

    Args:
        cps_df: A dataframe that contains the lower dimensional t-SNE components and the labels for each image.
    
    Returns:
        None
    """
    raise NotImplementedError

