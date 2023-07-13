# Import necessary modules
from eda.dataset.bps_dataset import BPSMouseDataset 
from eda.dataset.augmentation import(
    NormalizeBPS,
    ResizeBPS,
    ToTensor
)
from eda.dataset.bps_datamodule import BPSDataModule

from eda.vis_utils import(
    plot_gallery_from_1D,
    plot_2D_scatter_plot,
)

def main():
    """
    You may use this function to test your code.
    """
    # Fix the random seed to ensure reproducibility across runs
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    data_dir = root / 'data' / 'processed'
    train_meta_fname = 'meta_dose_hi_hr_4_post_exposure_train.csv'
    val_meta_fname = 'meta_dose_hi_hr_4_post_exposure_test.csv'
    save_dir = root / 'models' / 'baselines'
    batch_size = 2
    max_epochs = 3
    accelerator = 'auto'
    num_workers = 1
    acc_devices = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dm_stage = 'train'
    
    # Instantiate BPSDataModule 
    bps_datamodule = BPSDataModule(train_csv_file=train_meta_fname,
                                   train_dir=data_dir,
                                   val_csv_file=val_meta_fname,
                                   val_dir=data_dir,
                                   resize_dims=(64, 64),
                                   batch_size=batch_size,
                                   num_workers=num_workers)
    
    # Setup BPSDataModule which will instantiate the BPSMouseDataset objects
    # to be used for training and validation depending on the stage ('train' or 'val')
    bps_datamodule.setup(stage=dm_stage)

    image_stream_1d, all_labels = preprocess_images(lt_datamodule=bps_datamodule.train_dataloader)
    print(f'image_stream_1d.shape: {image_stream_1d.shape}')
    # Project the flattened images onto the principal components
    IMAGE_SHAPE = (64, 64)
    N_ROWS = 5
    N_COLS = 7
    N_COMPONENTS = N_ROWS * N_COLS

    # Perform PCA on the flattened images and specify the number of components to keep as 35
    pca, X_pca = perform_pca(X_flat=image_stream_1d, n_components=N_COMPONENTS)
    print(f'X_pca: {X_pca.shape}')

    # Plot the 1d array of flattened images
    plot_gallery_from_1D(title="Cell_Gallery_from_1D_Array",
                     images=image_stream_1d[:N_COMPONENTS],
                     n_row=N_ROWS,
                     n_col=N_COLS,
                     img_shape=IMAGE_SHAPE)
    print(f'X_pca.shape: {X_pca.shape}')

    # Plot the 1d array of flattened images after reducing the dimensionality using PCA
    plot_gallery_from_1D(title="PCA_Cell_Gallery_from_1D_Array",
                     images=pca.components_,
                     n_row=N_ROWS,
                     n_col=N_COLS,
                     img_shape=IMAGE_SHAPE)

    # Perform t-SNE on the flattened images before reducing the dimensionality using PCA
    X_tsne_direct = perform_tsne(X_reduced_dim=image_stream_1d, perplexity=30, n_components=2)
    print(f'X_tsne_direct.shape: {X_tsne_direct.shape}')
    # Perform t-SNE on the flattened images after reducing the dimensionality using PCA
    X_tsne_pca = perform_tsne(X_reduced_dim=X_pca, perplexity=30, n_components=2)
    print(f'X_tsne_pca.shape: {X_tsne_pca.shape}')
    tsne_df_direct = create_tsne_cp_df(X_tsne_direct, all_labels, 1000)
    print(tsne_df_direct.head())
    print(f'tsne_df_direct.shape: {tsne_df_direct.shape}')
    tsne_df_pca = create_tsne_cp_df(X_tsne_pca, all_labels, 1000)
    print(tsne_df_pca.head())
    print(f'tsne_df_pca.shape: {tsne_df_pca.shape}')
    plot_2D_scatter_plot(tsne_df_direct, 'tsne_direct_4hr_Gy_hi')
    plot_2D_scatter_plot(tsne_df_pca, 'tsne_pca_4hr_Gy_hi')

if __name__ == "__main__":
    main()