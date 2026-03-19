"""
Made by copilot

Parser for Haar Cascade Classifier XML files used in Viola-Jones face detection.
Extracts stages, weak classifiers, features, and thresholds.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Dict
import os


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


class HaarCascadeParser:
    """Parser for Haar Cascade XML files."""
    
    def __init__(self, xml_path: str):
        """
        Initialize the parser with a Haar cascade XML file.
        
        Args:
            xml_path: Path to the Haar cascade XML file
        """
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Cascade file not found: {xml_path}")
        
        self.xml_path = xml_path
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
    
    def parse(self) -> HaarCascade:
        """
        Parse the entire cascade structure.
        
        Returns:
            HaarCascade object containing all cascade information
        """
        # Handle both formats:
        # 1. Root is 'cascade' directly
        # 2. Root is 'opencv_storage' and cascade is first child
        cascade_elem = None
        
        if self.root.tag == 'cascade':
            cascade_elem = self.root
        elif self.root.tag == 'opencv_storage':
            # Find the first child with cascade data
            for child in self.root:
                if 'type_id' in child.attrib and 'classifier' in child.attrib['type_id']:
                    cascade_elem = child
                    break
        else:
            # Try to find 'cascade' as a child
            cascade_elem = self.root.find('cascade')
        
        if cascade_elem is None:
            raise ValueError("No 'cascade' element found in XML")
        
        # Extract basic cascade properties
        cascade_type = cascade_elem.get('type_id', 'unknown')
        
        feature_type_elem = cascade_elem.find('featureType')
        feature_type = feature_type_elem.text if feature_type_elem is not None else 'HAAR'
        
        # Handle 'size' element (height width format) and separate height/width
        size_elem = cascade_elem.find('size')
        if size_elem is not None and size_elem.text:
            parts = size_elem.text.split()
            width = int(parts[0]) if len(parts) > 0 else 24
            height = int(parts[1]) if len(parts) > 1 else 24
        else:
            height_elem = cascade_elem.find('height')
            height = int(height_elem.text) if height_elem is not None else 24
            
            width_elem = cascade_elem.find('width')
            width = int(width_elem.text) if width_elem is not None else 24
        
        stage_num_elem = cascade_elem.find('stageNum')
        stage_count = int(stage_num_elem.text) if stage_num_elem is not None else 0
        
        stage_params = cascade_elem.find('stageParams')
        max_weak_count = None
        if stage_params is not None:
            max_weak_elem = stage_params.find('maxWeakCount')
            if max_weak_elem is not None:
                max_weak_count = int(max_weak_elem.text)
        
        # Parse features first (needed by classifiers)
        features = self._parse_features(cascade_elem)
        
        # Parse stages
        stages = self._parse_stages(cascade_elem, features)
        
        return HaarCascade(
            cascade_type=cascade_type,
            feature_type=feature_type,
            height=height,
            width=width,
            stage_count=stage_count,
            stages=stages,
            features=features,
            max_weak_count=max_weak_count
        )
    
    def _parse_features(self, cascade_elem) -> Dict[int, Feature]:
        """
        Parse all features from the cascade.
        
        Args:
            cascade_elem: The cascade XML element
            
        Returns:
            Dictionary mapping feature IDs to Feature objects
        """
        features = {}
        features_elem = cascade_elem.find('.//features')
        
        if features_elem is None:
            return features
        
        for feature_id, feature_elem in enumerate(features_elem.findall('_')):
            rects_elem = feature_elem.find('rects')
            
            rectangles = []
            if rects_elem is not None:
                for rect_elem in rects_elem.findall('_'):
                    # Parse rectangle definition
                    # Format: "x y width height weight"
                    if rect_elem.text is None:
                        continue
                    rect_text = rect_elem.text.strip()
                    parts = rect_text.split()
                    
                    if len(parts) >= 5:
                        x = int(parts[0])
                        y = int(parts[1])
                        width = int(parts[2])
                        height = int(parts[3])
                        weight = float(parts[4])
                        
                        rectangles.append(Rectangle(x, y, width, height, weight))
            
            features[feature_id] = Feature(feature_id, rectangles)
        
        return features
    
    def _parse_stages(self, cascade_elem, features: Dict[int, Feature]) -> List[Stage]:
        """
        Parse all stages from the cascade.
        Handles both old format (weakClassifiers) and new format (trees).
        
        Args:
            cascade_elem: The cascade XML element
            features: Dictionary of features
            
        Returns:
            List of Stage objects
        """
        stages = []
        stages_elem = cascade_elem.find('stages')
        
        if stages_elem is None:
            return stages
        
        stage_id = 0
        for stage_elem in stages_elem.findall('_'):
            # Parse stage threshold
            threshold_elem = stage_elem.find('stage_threshold')
            threshold = float(threshold_elem.text) if threshold_elem is not None else 0.0
            
            max_weak_elem = stage_elem.find('maxWeakCount')
            max_weak_count = int(max_weak_elem.text) if max_weak_elem is not None else None
            
            # Try new format first (trees), then fall back to old format (weakClassifiers)
            weak_classifiers = self._parse_weak_classifiers_trees(stage_elem, features)
            if not weak_classifiers:
                weak_classifiers = self._parse_weak_classifiers(stage_elem, features)
            
            stage = Stage(
                stage_id=stage_id,
                threshold=threshold,
                weak_classifiers=weak_classifiers,
                max_weak_count=max_weak_count
            )
            stages.append(stage)
            stage_id += 1
        
        return stages
    
    def _parse_weak_classifiers(
        self,
        stage_elem,
        features: Dict[int, Feature]
    ) -> List[WeakClassifier]:
        """
        Parse weak classifiers from a stage (old format).
        
        Args:
            stage_elem: The stage XML element
            features: Dictionary of features
            
        Returns:
            List of WeakClassifier objects
        """
        weak_classifiers = []
        weak_classifiers_elem = stage_elem.find('weakClassifiers')
        
        if weak_classifiers_elem is None:
            return weak_classifiers
        
        classifier_id = 0
        for classifier_elem in weak_classifiers_elem.findall('_'):
            # Parse internal node (contains feature_id and threshold)
            internal_nodes_elem = classifier_elem.find('internalNodes')
            internal_nodes_text = internal_nodes_elem.text.strip() if internal_nodes_elem is not None else ""
            
            # OpenCV format: "left_node right_node feature_index threshold"
            # For simple stumps this is commonly "0 -1 <feature_id> <threshold>".
            parts = internal_nodes_text.split()
            feature_id = int(parts[2]) if len(parts) > 2 else -1
            threshold = float(parts[3]) if len(parts) > 3 else 0.0
            
            # Parse leaf values
            leaf_values_elem = classifier_elem.find('leafValues')
            leaf_values_text = (
                leaf_values_elem.text.strip()
                if leaf_values_elem is not None and leaf_values_elem.text is not None
                else ""
            )
            leaf_values = list(map(float, leaf_values_text.split()))
            
            left_value = leaf_values[0] if len(leaf_values) > 0 else 0.0
            right_value = leaf_values[1] if len(leaf_values) > 1 else 0.0
            
            weak_clf = WeakClassifier(
                classifier_id=classifier_id,
                feature_id=feature_id,
                threshold=threshold,
                left_value=left_value,
                right_value=right_value,
                feature=features.get(feature_id, None)
            )
            weak_classifiers.append(weak_clf)
            classifier_id += 1
        
        return weak_classifiers
    
    def _parse_weak_classifiers_trees(
        self,
        stage_elem,
        features: Dict[int, Feature]
    ) -> List[WeakClassifier]:
        """
        Parse weak classifiers from a stage (new tree-based format).
        
        Args:
            stage_elem: The stage XML element
            features: Dictionary of features
            
        Returns:
            List of WeakClassifier objects
        """
        weak_classifiers = []
        trees_elem = stage_elem.find('trees')
        
        if trees_elem is None:
            return weak_classifiers
        
        classifier_id = 0
        feature_id_counter = 0
        
        for tree_elem in trees_elem.findall('_'):
            # Each tree contains a single weak classifier
            # with embedded feature, threshold, and leaf values
            
            # Find the actual classifier element (nested "_" within the tree)
            classifier_node = tree_elem.find('_')
            if classifier_node is None:
                classifier_node = tree_elem
            
            # Parse feature
            feature_elem = classifier_node.find('feature')
            rectangles = []
            feature_id_for_clf = feature_id_counter
            
            if feature_elem is not None:
                rects_elem = feature_elem.find('rects')
                if rects_elem is not None:
                    for rect_elem in rects_elem.findall('_'):
                        if rect_elem.text is None:
                            continue
                        rect_text = rect_elem.text.strip()
                        parts = rect_text.split()
                        
                        if len(parts) >= 5:
                            x = int(parts[0])
                            y = int(parts[1])
                            width = int(parts[2])
                            height = int(parts[3])
                            weight = float(parts[4])
                            
                            rectangles.append(Rectangle(x, y, width, height, weight))
                
                # Store the feature
                if rectangles:
                    features[feature_id_for_clf] = Feature(feature_id_for_clf, rectangles)
                    feature_id_counter += 1
            
            # Parse threshold
            threshold_elem = classifier_node.find('threshold')
            threshold = float(threshold_elem.text) if threshold_elem is not None else 0.0
            
            # Parse leaf values
            left_val_elem = classifier_node.find('left_val')
            right_val_elem = classifier_node.find('right_val')
            
            left_value = float(left_val_elem.text) if left_val_elem is not None else 0.0
            right_value = float(right_val_elem.text) if right_val_elem is not None else 0.0
            
            weak_clf = WeakClassifier(
                classifier_id=classifier_id,
                feature_id=feature_id_for_clf,
                threshold=threshold,
                left_value=left_value,
                right_value=right_value,
                feature=features.get(feature_id_for_clf, None)
            )
            weak_classifiers.append(weak_clf)
            classifier_id += 1
        
        return weak_classifiers
    
    def print_summary(self, cascade: HaarCascade):
        """Print a summary of the cascade structure."""
        print(f"\n{'='*70}")
        print(f"Haar Cascade Summary")
        print(f"{'='*70}")
        print(f"Type: {cascade.cascade_type}")
        print(f"Feature Type: {cascade.feature_type}")
        print(f"Size: {cascade.width}x{cascade.height}")
        print(f"Total Stages: {len(cascade.stages)}")
        print(f"Total Features: {len(cascade.features)}")
        if cascade.max_weak_count:
            print(f"Max Weak Count: {cascade.max_weak_count}")
        print(f"\n{'='*70}")
    
    def print_stages_summary(self, cascade: HaarCascade):
        """Print summary of all stages and their classifiers."""
        print(f"\n{'='*70}")
        print(f"Stages Summary")
        print(f"{'='*70}\n")
        
        for stage in cascade.stages:
            print(f"Stage {stage.stage_id}:")
            print(f"  Threshold: {stage.threshold:.10f}")
            print(f"  Weak Classifiers: {len(stage.weak_classifiers)}")
            if stage.max_weak_count:
                print(f"  Max Count: {stage.max_weak_count}")
            
            # Print first 5 classifiers
            for clf in stage.weak_classifiers[:5]:
                print(f"    {clf}")
            
            if len(stage.weak_classifiers) > 5:
                print(f"    ... and {len(stage.weak_classifiers) - 5} more")
            print()
    
    def print_features_summary(self, cascade: HaarCascade):
        """Print summary of features."""
        print(f"\n{'='*70}")
        print(f"Features Summary")
        print(f"{'='*70}\n")
        
        for feat_id, feature in list(cascade.features.items())[:10]:
            print(f"Feature {feat_id}:")
            for rect in feature.rectangles:
                print(f"  {rect}")
            print()
        
        if len(cascade.features) > 10:
            print(f"... and {len(cascade.features) - 10} more features")


def load_cascade(cascade_path: str) -> HaarCascade:
    """
    Convenience function to load a cascade.
    
    Args:
        cascade_path: Path to cascade XML file
        
    Returns:
        Parsed HaarCascade object
    """
    print(f"Loading Haar cascade from: {cascade_path}")
    parser = HaarCascadeParser(cascade_path)
    return parser.parse()
