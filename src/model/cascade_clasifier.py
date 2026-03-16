import numpy as np

from src.data.filter import get_integral_image, get_integral_sum
from src.data.crops import get_all_image_crops
from .haar_cascade_parser import HaarCascade


class CascadeClassifier:
    def __init__(self, CONFIG, cascade: HaarCascade):
        self.CONFIG = CONFIG
        self.cascade = cascade

    def predict(self, img=None, img_path=None):
        assert img is not None or img_path is not None, "Either img or img_path must be provided"

        crops = get_all_image_crops(
            self.CONFIG, 
            img=img,
            img_path=img_path
        )

        faces = [
            crop
            for crop in crops
            if self._predict_crop(crop["img"])
        ]

        return faces
    
    def _predict_crop(self, crop_img):
        integral_img = get_integral_image(crop_img)

        # TODO:with integral image
        std_dev = np.std(crop_img)
        if std_dev <= 0:
            std_dev = 1.0

        for stage in self.cascade.stages:
            if not self._predict_stage(integral_img, stage):
                return False
        
        return True

    def _predict_stage(self, integral_img, stage, std_dev=1.0):
        total_sum = 0.0
        for clf in stage.weak_classifiers:
            feature_val = self._compute_feature(integral_img, clf.feature)

            norm_thrs = clf.threshold * std_dev
            total_sum += (
                clf.left_value 
                if feature_val >= norm_thrs 
                else clf.right_value
            )

        return total_sum >= stage.threshold
    
    def _compute_feature(self, integral_img, feature):
        feature_sum = 0.0
        for rec in feature.rectangles:
            rec_sum = get_integral_sum(
                integral_img, 
                rec.x, rec.y, 
                rec.x+rec.width-1, rec.y+rec.height-1
            )
            feature_sum += rec_sum * rec.weight
    
        return feature_sum
    