from dataclasses import dataclass
from typing import Dict, List

# =====================================================================
#                           HaarCascade 
# =====================================================================

@dataclass
class Rectangle:
    """Represents a rectangle in a feature."""
    x: int
    y: int
    width: int
    height: int
    weight: float  # The weight/scaling factor
    
    def __repr__(self):
        return f"Rect(x={self.x}, y={self.y}, w={self.width}, h={self.height}, weight={self.weight})"


@dataclass
class Feature:
    """Represents a Haar feature (composed of rectangles)."""
    feature_id: int
    rectangles: List[Rectangle]
    
    def __repr__(self):
        return f"Feature({self.feature_id}): {len(self.rectangles)} rects"


@dataclass
class WeakClassifier:
    """Represents a weak classifier within a stage."""
    classifier_id: int
    feature_id: int
    threshold: float
    left_value: float
    right_value: float
    feature: Feature = None  # Reference to the actual feature
    
    def __repr__(self):
        return (f"WeakClassifier({self.classifier_id}): "
                f"Feature {self.feature_id}, "
                f"threshold={self.threshold:.6f}, "
                f"left={self.left_value:.6f}, right={self.right_value:.6f}")


@dataclass
class Stage:
    """Represents a cascade stage."""
    stage_id: int
    threshold: float
    weak_classifiers: List[WeakClassifier]
    max_weak_count: int = None
    
    def __repr__(self):
        return (f"Stage({self.stage_id}): "
                f"threshold={self.threshold:.6f}, "
                f"{len(self.weak_classifiers)} classifiers")


@dataclass
class HaarCascade:
    """Complete Haar cascade classifier."""
    cascade_type: str
    feature_type: str
    height: int
    width: int
    stage_count: int
    stages: List[Stage]
    features: Dict[int, Feature]
    max_weak_count: int = None
    
    def __repr__(self):
        return (f"HaarCascade: {self.width}x{self.height}, "
                f"{len(self.stages)} stages, "
                f"{len(self.features)} features")

