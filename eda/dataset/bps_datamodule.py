""" This module contains the PyTorch Lightning BPSMouseDataModule to use with the PyTorch
BPSMouseDataset class."""

# Add additional necessary imports

from eda.data_utils import save_tiffs_local_from_s3
import boto3
from botocore import UNSIGNED
from botocore.config import Config

from eda.dataset.bps_dataset import BPSMouseDataset

from eda.dataset.augmentation import (
    NormalizeBPS,
    ResizeBPS,
    ToTensor
)

class BPSDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_csv_file: str,
                 train_dir: str,
                 val_csv_file: str,
                 val_dir: str,
                 resize_dims: tuple,
                 test_csv_file: str = None,
                 test_dir: str = None,
                 file_on_prem: bool = True,
                 batch_size: int = 4,
                 num_workers: int = 2,
                 meta_csv_file: str = None,
                 meta_root_dir: str = None,
                 s3_client: boto3.client = None,
                 s3_path: str = None,
                 bucket_name: str = None):
        """
        PyTorch Lightning DataModule for the BPS microscopy data.

        Args:
            train_csv_file (str): The name of the csv file containing the training data.
            train_dir (str): The directory where the training data is stored.
            val_csv_file (str): The name of the csv file containing the validation data.
            val_dir (str): The directory where the validation data is stored.
            resize_dims (tuple): The dimensions to resize the images to during Transform.
            test_csv_file (str): The name of the csv file containing the test data.
            test_dir (str): The directory where the test data is stored.
            file_on_prem (bool): Whether the data is stored on-prem or in the cloud 
                                 (needed by BPSDataset)
            batch_size (int): The batch size to use for the DataLoader.
            num_workers (int): The number of workers to use for the DataLoader.
            meta_csv_file (str): The name of the csv file containing the all the metadata
                                (needed by BPSDataset when file_on_prem = False)
            meta_root_dir (str): The directory where the metadata is stored
                                (needed by BPSDataset when file_on_prem = False)
        """
        super().__init__()
        self.train_csv = train_csv_file
        self.train_dir = train_dir
        self.val_csv = val_csv_file
        self.val_dir = val_dir
        self.test_csv = test_csv_file
        self.test_dir = test_dir
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.s3_path = s3_path
        self.on_prem = file_on_prem
        self.meta_csv = meta_csv_file
        self.meta_dir = meta_root_dir
        self.resize_dims = resize_dims
        self.transform = transforms.Compose([
                            NormalizeBPS(),
                            ResizeBPS(resize_dims[0], resize_dims[1]),
                            ToTensor()
                ])
        self.on_prem = file_on_prem
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def prepare_data(self) -> None:
        """
        Download data if needed. This method is called only from a single CPU.
        """
        raise NotImplementedError
        
    def setup(self, stage: str):
        """
        Assign train/val datasets for use in dataloaders. Requires that train,
        val, and test csv files be stored locally. Image tiffs will be stored
        in the same directory as the csv files.
        """
        raise NotImplementedError


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Returns the training dataloader.
        """
        raise NotImplementedError
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Returns the validation dataloader.
        """
        raise NotImplementedError
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        """
        Returns the test dataloader. In our case, we will only use the val_dataloader
        since NASA GeneLab has not released the test set.
        """
        raise NotImplementedError
