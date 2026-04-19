from tqdm import tqdm

from src.config import Configuration
from src.model import CascadeClassifier

from maikol_utils.print_utils import print_separator, print_color
from maikol_utils.file_utils import list_dir_files


def test_cascade_fpr(CONFIG: Configuration, classifier: CascadeClassifier):
    print_separator("Testing cascade FPR on training faces", sep_type="LONG")

    all_crops, cn = list_dir_files(CONFIG.no_faces_crops_path)
    print(f" - Found {cn} no-face crops in {CONFIG.no_faces_crops_path}\n")
    
    all_fps, total_candidates = [], 0
    for img_path in tqdm(all_crops):  # Test on the first 5 images
        fps, candidates = classifier.predict_no_merge(img_path=img_path, return_candidate_count=True)
        all_fps.extend(fps)
        total_candidates += candidates

    fpr = len(all_fps) / total_candidates if total_candidates > 0 else 0
    print_color(f" - False Positive Rate: {fpr:.6f}", color="green")
    return fpr