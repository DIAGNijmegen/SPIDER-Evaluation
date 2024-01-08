import SimpleITK
from collections import OrderedDict, defaultdict
import numpy as np
import SimpleITK as sitk
import pandas

from scipy.ndimage.morphology import (
    distance_transform_edt,
    binary_erosion,
    generate_binary_structure,
)

from typing import Iterable, Optional, List, Union
import json
from pathlib import Path


def dice_score(mask1: Iterable[bool], mask2: Iterable[bool]) -> float:
    """Dice volume overlap score for two binary masks"""
    m1 = np.asarray(mask1, dtype="bool").flatten()
    m2 = np.asarray(mask2, dtype="bool").flatten()

    try:
        return 2 * np.count_nonzero(m1 & m2) / float(np.count_nonzero(m1) + np.count_nonzero(m2))
    except ZeroDivisionError:
        raise ValueError("Cannot compute dice score on empty masks")


class Spider:

    def __init__(self):

        self.ground_truth_path = Path("/opt/evaluation/ground-truth/")

        self.input_path = Path("/input/")
        self.output_path = Path("/output/")

        with open(self.input_path / "predictions.json", "r") as f:
            self.predictions_json = json.load(f)


    def score_case(self, gt_path, pred_path):

        # global surface_distance_vert

        # Load the images for this case
        gt = sitk.ReadImage(str(gt_path))
        pred = sitk.ReadImage(str(pred_path))

        mask_manual = sitk.GetArrayFromImage(gt)
        mask_automatic = sitk.GetArrayFromImage(pred)

        image = sitk.ReadImage(str(gt_path))
        spacing = np.array(image.GetSpacing())

        # Construct containers for the per-scan results
        all_dice_scores = defaultdict(list)
        # all_surface_distances = defaultdict(list)

        # Check if manual and automatic mask have the same dimensions
        if mask_manual.shape != mask_automatic.shape:
            print(' > Manual and automatic masks have different shapes: {} vs {}'.format(
                sitk.GetArrayFromImage(mask_manual).shape,
                sitk.GetArrayFromImage(mask_automatic).shape))

        # build lookup table for all labels
        label_lut = OrderedDict()
        all_labels_manual = sorted(list(np.unique(mask_manual[mask_manual > 0])))
        print(all_labels_manual)
        for label_manual in all_labels_manual:
            # Determine label in automatic mask with which this label overlaps the most
            overlap_automatic = mask_automatic[mask_manual == label_manual]
            overlap_automatic_foreground = overlap_automatic > 0
            if np.any(overlap_automatic_foreground):
                label_automatic = np.bincount(overlap_automatic[overlap_automatic_foreground]).argmax()
                label_lut[label_manual] = label_automatic

        dice_scores_vert = []
        dice_scores_discs = []
        total_vert = 0
        total_discs = 0
        missed_vert = 0
        missed_discs = 0
        detection_threshold = 0.1

        for label_manual in all_labels_manual:
            if label_manual not in label_lut:
                score = 0
            else:
                label_automatic = label_lut[label_manual]
                score = dice_score(mask_manual == label_manual, mask_automatic == label_automatic)

            if 'dice_score_SC' in locals():
                pass
            else:
                dice_score_SC = 999

            if label_manual > 0 and label_manual < 100:
                total_vert += 1
                if score < detection_threshold:
                    missed_vert += 1
                else:
                    dice_scores_vert.append(score)
            elif label_manual > 200:
                total_discs += 1
                if score < detection_threshold:
                    missed_discs += 1
                else:
                    dice_scores_discs.append(score)
            elif label_manual == 100:
                dice_score_SC = score
            all_dice_scores[label_manual].append(score)

        dice_score_vert = np.mean(dice_scores_vert)
        dice_score_discs = np.mean(dice_scores_discs)
        overall_dice_score = np.mean([v for vs in all_dice_scores.values() for v in vs])

        detection_rate_vert = (total_vert - missed_vert) / total_vert
        detection_rate_discs = (total_discs - missed_discs) / total_discs

        return {
            'DiceScoreVertebrae': dice_score_vert,
            'DiceScoreDiscs': dice_score_discs,
            'DiceScoreSpinalCanal': dice_score_SC,
            'OveralDiceScore': overall_dice_score,
            'DetectionRateVertebrae': detection_rate_vert,
            'DetectionRateDiscs': detection_rate_discs,
            'pred_fname': pred_path.name,
            'gt_fname': gt_path.name,
        }
        

    def compute_metrics(self):

        metric_accumulator = []

        for i in range(len(self.predictions_json)):
            entry = self.predictions_json[i]
            job_pk = entry['pk']
            pk = entry['outputs'][0]["image"]["pk"]
            name = entry['inputs'][0]["image"]["name"]

            gt_path = self.ground_truth_path / name
            pred_path = self.input_path / f"{job_pk}" / "output" / "images" / "sagittal-spine-mr-segmentation" / f"{pk}.mha"
            # pred_path = self.predictions_path / f"{pk}.mha"

            metric = self.score_case(gt_path, pred_path)
            metric_accumulator.append(metric)

        df = pandas.DataFrame(metric_accumulator)
        metric_columns = ['DiceScoreVertebrae', 'DiceScoreDiscs', 'DiceScoreSpinalCanal',
                          'OveralDiceScore', 'DetectionRateVertebrae', 'DetectionRateDiscs']#, 'ASDVertebrae', 'ASDDiscs', 'ASDSpinalCanal', 'OveralASD']

        results_metric = {}
        for metric_column in metric_columns:
            results_metric[metric_column] = {
                "mean": df[metric_column].mean(),
                "std": df[metric_column].std(),
            }

        with open("/output/metrics.json", "w") as f:
            json.dump(results_metric, f)


if __name__ == "__main__":
    evaluator = Spider()
    evaluator.compute_metrics()

