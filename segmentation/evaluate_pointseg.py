import os
import argparse
import json
import tempfile
import sys
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from scipy import ndimage
from skimage.filters import threshold_otsu
import cv2

from inv.segmentation.utils import (
    resize_segments_np,
    compute_segment_centroids,
    offset_single_centroid,
    get_baseline_metrics,
    plot_segments
)

# CutLER imports
from maskcut import maskcut, dino
from types import SimpleNamespace

class CutLERWrapper:
    # Fixed hyperparameters based on the demo notebook
    HYPERPARAMS = {
        'vit_arch': 'base',
        'vit_feat': 'k',
        'patch_size': 8,
        'url': 'https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth',
        'feat_dim': 768,
        'fixed_size': 480,  # Original was 480
        'tau': 0.15,
        'N': 3
    }

    def __init__(self, tmp_dir, device='cuda'):
        # Change to the CutLER maskcut directory to ensure imports work
        self.dino = dino

        self.args = SimpleNamespace(**self.HYPERPARAMS)

        self.backbone = self.dino.ViTFeat(
            self.args.url,
            self.args.feat_dim,
            self.args.vit_arch,
            self.args.vit_feat,
            self.args.patch_size
        )
        self.backbone.eval()

        self.tmp_dir = tmp_dir
        self.device = device
        self.cpu_eval = device == 'cpu'

        if not self.cpu_eval:
            self.backbone.cuda()

    def segment_from_image(self, image_array, pt, num_segs, attention=False):
        """
        Process an image array and return segmentation masks

        Args:
            image_array: RGB numpy array (H, W, 3)

        Returns:
            predicted_masks: Binary mask numpy array (N, H, W)
        """
        # Save image to temporary file since CutLER requires a file path
        with tempfile.NamedTemporaryFile(suffix='.png', dir=self.tmp_dir, delete=False) as tmp:
            temp_path = tmp.name
            img = Image.fromarray(image_array)
            img.save(temp_path)

        try:
            # Import CRF and metrics inside the function to avoid import issues
            from maskcut.crf import densecrf
            from third_party.TokenCut.unsupervised_saliency_detection import metric
            from maskcut.maskcut import maskcut

            bipartitions, _, I_new, attention_map = maskcut(
                temp_path,
                self.backbone,
                self.args.patch_size,
                self.args.tau,
                N=num_segs,
                fixed_size=self.args.fixed_size,
                cpu=self.cpu_eval,
                pt=[pt[0], pt[1]],
                attention=attention
            )

            if attention:
                attention_map = cv2.resize(attention_map, (256, 256), interpolation=cv2.INTER_AREA)
                bipartitions = [attention_map]

            # Get original dimensions
            height, width = image_array.shape[:2]

            pseudo_mask_list = []

            for idx, bipartition in enumerate(bipartitions):
                # Post-process pseudo-masks with CRF
                pseudo_mask = densecrf(np.array(I_new), bipartition)
                pseudo_mask = ndimage.binary_fill_holes(pseudo_mask >= 0.5)

                pseudo_mask = Image.fromarray(np.uint8(pseudo_mask * 255))
                pseudo_mask = np.asarray(pseudo_mask.resize((width, height)))

                pseudo_mask = pseudo_mask.astype(np.uint8)
                upper = np.max(pseudo_mask)
                lower = np.min(pseudo_mask)
                thresh = upper / 2.0
                pseudo_mask[pseudo_mask > thresh] = upper
                pseudo_mask[pseudo_mask <= thresh] = lower

                pseudo_mask_list.append(pseudo_mask)

            # Stack masks and normalize to [0, 1]
            predicted_masks = (np.stack(pseudo_mask_list, 0) / 255)

            return predicted_masks

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def segment_at_points(self, image, points, num_segs, attention=False):
        """
        Similar interface to SAM2_Multimask.segment_at_points but for CutLER

        Args:
            image: H×W×3 uint8
            points: N×2 float, (x,y) pixel coordinates of points of interest

        Returns:
            mask: HxW binary mask (mask containing the point or empty mask)
            point: the point used for segmentation
        """
        # CutLER doesn't use points for segmentation, it's unsupervised
        # But we can use the points to select which mask to return


        # Get all predicted masks
        all_masks = self.segment_from_image(image, points, num_segs, attention)
        if attention:
            return all_masks[0], [points[0], points[1]]

        # Extract the first point (x, y) from points
        x, y = int(points[0]), int(points[1])

        # Make sure x, y are within image bounds
        height, width = image.shape[:2]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))

        # Check which mask contains the point
        mask_found = False
        for mask in all_masks:
            if mask[y, x] > 0:  # If the point is inside this mask
                best_mask = mask
                mask_found = True
                break

        # If no mask contains the point, return an empty mask
        if not mask_found:
            best_mask = np.zeros((height, width), dtype=np.float32)

        # Return the selected mask and the point
        return best_mask, [points[0], points[1]]


def main(args, h5_save_file):
    # Create temporary directory for CutLER intermediates
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Build CutLER wrapper
        cutler_wrapper = CutLERWrapper(tmp_dir=tmp_dir, device=args.device)

        os.makedirs(os.path.dirname(h5_save_file), exist_ok=True)

        # Open annotations file
        dataset = h5py.File(args.annotations_h5, 'r')

        with h5py.File(h5_save_file, "w") as h5f:
            for img_idx, img_name in enumerate(dataset.keys()):
                print(f"Processing {img_name}...")

                # Load image
                image = dataset[img_name]
                I = image['rgb'][:]

                # Load GT segments and centroids
                gt_segs = image['segment'][:]
                centroids = compute_segment_centroids(torch.tensor(gt_segs))

                # Create group for this image
                img_grp = h5f.create_group(f"img{img_idx}")
                img_grp.create_dataset("image_rgb", data=I, compression="gzip")
                img_grp.create_dataset("segments_gt", data=gt_segs, compression="gzip")

                for si, cent in enumerate(centroids):
                    offsets = offset_single_centroid(
                        cent, N=args.num_offset_points,
                        min_mag=args.min_mag, max_mag=args.max_mag
                    )
                    seg_grp = img_grp.create_group(f"seg{si}")

                    for pi, pt in enumerate(offsets):
                        # Get segment from CutLER
                        num_seg = max(args.num_seg, gt_segs.shape[0])
                        mask, pt_mv = cutler_wrapper.segment_at_points(I, pt.detach().cpu().numpy(), num_seg, args.attention)

                        # Store result
                        pt_grp = seg_grp.create_group(f"pt{pi}")
                        pt_grp.create_dataset("segment", data=mask, compression="gzip")
                        pt_grp.create_dataset("centroid", data=pt_mv, compression="gzip")

        print(f"Done writing all segments to '{h5_save_file}'.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # dataset
    p.add_argument("--save_dir", default='./test_vis_pointseg_spectral/')

    p.add_argument("--annotations_h5", default='/ccn2/u/lilianch/data/entityseg_100.h5')

    # test
    p.add_argument("--test", type=str,  nargs='+',
                        choices=['h5', 'vis', 'metrics'], default =['h5', 'vis', 'metrics'],
                        help='h5 for saving segments, vis for visualizing, metrics to save/print metrics')

    # h5 test
    p.add_argument("--h5_file", default='./segments.h5',
                   help='location to retrieve segments h5 file (assumes existing)')

    # CutLER specific args
    p.add_argument("--n_segments", type=int, default=3,
                   help='Number of segments for CutLER to generate')

    # misc args
    p.add_argument("--num_offset_points", type=int, default=1)
    p.add_argument("--min_mag", type=float, default=10.0)
    p.add_argument("--max_mag", type=float, default=25.0)
    p.add_argument("--num_seg", type=int, default=7)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attention", action='store_true', help='Test with attention matrix as segments')

    args = p.parse_args()

    save_dir = args.save_dir
    h5_save_file = os.path.join(save_dir, 'segments.h5')
    if 'h5' in args.test:
        if os.path.exists(h5_save_file):
            response = input(f"File '{h5_save_file}' already exists. Overwrite? (y/[n]): ").strip().lower()
            if response != 'y':
                print("Aborting.")
                sys.exit(0)
        main(args, h5_save_file)

    if 'vis' in args.test:
        assert os.path.exists(h5_save_file), f"Expected file '{h5_save_file}' to exist, but it was not found. Make sure the file is generated or the correct path is provided."
        img_save_dir = os.path.join(save_dir, 'vis/')
        os.makedirs(os.path.dirname(img_save_dir), exist_ok=True)
        plot_segments(h5_save_file, img_save_dir)

    if 'metrics' in args.test:
        assert os.path.exists(h5_save_file), f"Expected file '{h5_save_file}' to exist, but it was not found. Make sure the file is generated or the correct path is provided."
        metrics_save_json = os.path.join(save_dir, 'metrics.json')
        os.makedirs(os.path.dirname(metrics_save_json), exist_ok=True)
        if not os.path.exists(metrics_save_json):
            with open(metrics_save_json, 'w') as f:
                pass
        get_baseline_metrics(h5_save_file, metrics_save_json)

