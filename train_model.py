
from src.model import RowleyFaceNN, FaceCNN, train_model, FaceDetectionModule, FaceDataModule
from src.config.config import Configuration


def main():
    CONFIG = Configuration()
    data_module  = FaceDataModule(CONFIG)
    model_module = FaceDetectionModule(
        CONFIG=CONFIG,
        model=FaceCNN(
            in_channels=1 if CONFIG.gray_scale else 3,
            num_classes=1,
            out_size=(1,1)
        ),
    )

    train_model(
        CONFIG, 
        data_module,
        model_module
    )


if __name__ == "__main__":
    main()

