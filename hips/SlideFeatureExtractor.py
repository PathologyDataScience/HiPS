import os
from os.path import join as opj
import logging
import warnings
from pandas import DataFrame, read_csv, concat
from copy import deepcopy
import numpy as np
import matplotlib.pylab as plt
from typing import Iterable, Callable, Tuple, Union, Any, Dict, Set, List
from scipy.stats import entropy
from functools import cached_property
from collections import namedtuple
import large_image
from skimage.morphology import (
    binary_dilation, binary_erosion, label as sklabel,
    remove_small_holes, remove_small_objects,
)
from skimage.color import rgb2gray
from skimage.feature import canny
from scipy.sparse import csr_matrix
from skimage.measure import regionprops_table
from PIL import Image
from histomicstk.features.compute_morphometry_features import (
    _fractal_dimension
)
from histomicstk.features.compute_intensity_features import (
    compute_intensity_features
)

from MuTILs_Panoptic.utils.GeneralUtils import (
    CollectErrors, load_json, flatten_dict, normalize_to_zero_one_range,
    weighted_avg_and_std, append_row_to_df_or_create_it,
)
from MuTILs_Panoptic.configs.panoptic_model_configs import (
    RegionCellCombination, VisConfigs
)
from hips.RipleysK import FastRipleyK
from MuTILs_Panoptic.utils.MiscRegionUtils import (
    get_region_within_x_pixels, _pixsum,
)


collect_errors = CollectErrors()


class SlideCollagenFeatureExtractor(object):
    """
    Extracts collagen features from MuTIsWSIRunner ouput.
    """
    def __init__(
        self,
        slide_dir: str,
        output_dir: str,
        wsi_file: str,
        *,
        slide_name: str = None,
        mpp: float = 0.5,
        roi_size: int = 512,
        topk_salient_rois: int = 256,
        min_stroma_ratio=0.3,
        min_tumor_ratio=0.2,
        region_prop_names: Iterable[str] = None,
        max_axis_ratio_for_fiber=0.2,
        monitor: str = "",
        logger: Any = None,
        _region_class_map: Dict[str, str] = None,
        _nucleus_class_map: Dict[str, str] = None,
        _debug: bool = False,
    ):
        """
        This represents the slide as a feature matrix summarizing
        the collagen with tumor-associated stroma.
        """
        self.slide_dir = slide_dir
        self.output_dir = output_dir
        self.wsi_file = wsi_file
        self.slide_name = slide_name or os.path.basename(slide_dir)
        self.mpp = mpp
        self.roi_size = roi_size
        self.topk_salient_rois = topk_salient_rois
        self.min_stroma_ratio = min_stroma_ratio
        self.min_tumor_ratio = min_tumor_ratio
        self.max_axis_ratio_for_fiber = max_axis_ratio_for_fiber
        self.monitor = monitor
        self.logger = logger or logging.getLogger(__name__)
        self._debug = _debug

        # codes and class map for most relevant classes
        self._region_class_map = _region_class_map or {
            'TILS': 'STROMA',  # tils-dense stroma is still stroma!
        }
        self._rcd = RegionCellCombination.REGION_CODES
        self._nucleus_class_map = _nucleus_class_map or {
            'ActiveStromalCellNOS': 'StromalCellNOS',
        }
        self._ncd = RegionCellCombination.NUCLEUS_CODES

        # region props to extract per collagen fibril edge
        self._region_prop_names = region_prop_names or (
            'label',
            'area',  # fibril edge length -- a bit noisy d.t. looping
            'major_axis_length',  # more robust than area visually
            'minor_axis_length',
            # directionality -- Angle between the 0th axis (rows) and the
            # major axis of the ellipse that has the same second moments
            # as the region, ranging from -pi/2 to pi/2 counter-clockwise
            'orientation',
        )

        if self._debug:
            os.makedirs(
                opj(
                    self.output_dir,
                    "perSlideCollagenFeatures",
                    f"DEBUG.{self.slide_name}",
                ),
                exist_ok=True
            )

        # internal
        self._roiname = None

    def run(self):
        """"""
        # summarize features for each roi in the slide (one row per roi)
        features = self._load_or_extract_collagen_features_for_all_rois()
        # summary of roi features .. whole slide is one row now
        self._summarize_collagen_features_for_slide(features)

    # HELPERS -----------------------------------------------------------------

    def _load_or_extract_collagen_features_for_all_rois(self):
        """"""
        where = opj(
            self.output_dir, 'perSlideCollagenFeatures', f"{self.slide_name}.csv"
        )
        # maybe this was already done
        if os.path.isfile(where):
            try:
                return read_csv(where, index_col=0)
            except Exception as whoknows:
                pass

        features = {}
        tidx = -1
        for tile_mask_info in self._mask_tilesource.tileIterator(
                tile_size={
                    'width': self._mask_tile_size,
                    'height': self._mask_tile_size,
                },
                tile_overlap={'x': 0, 'y': 0},
                format=large_image.tilesource.TILE_FORMAT_PIL
        ):
            tidx += 1
            if tidx % 100 == 0:
                self.logger.info(f"{self.monitor}: loading tile {tidx}")

            feats_this_roi = self.extract_collagen_features_for_one_roi(
                tile_mask_info=tile_mask_info, tidx=tidx,
            )
            if feats_this_roi is not None:
                features[self._roiname] = feats_this_roi

        # save per-roi collagen features for this slide
        features = DataFrame.from_records(features).T
        features.to_csv(where, index_label='')

        return features

    @collect_errors()
    def extract_collagen_features_for_one_roi(
        self, tile_mask_info: Dict, tidx: int
    ):
        """"""
        outs = self._isolate_stromal_components(tile_mask_info)
        if outs is None:
            return

        wsi_base_coords = self._parse_coords_relative_to_wsi_base(
            tile_mask_info
        )
        self._roiname = f"{self.slide_name}_roi-{tidx}" + ''.join([
            f"_{loc}-{wsi_base_coords[loc]}"
            for loc in ['left', 'top', 'right', 'bottom']
        ])
        feats = {'Saliency.SaliencyScore': outs['saliency_score']}
        feats.update(
            self._calculate_features_for_this_roi(
                rgb=self._fetch_roi_rgb(wsi_base_coords),
                stroma_semantic=outs['stroma_semantic'],
            )
        )
        return feats

    def _summarize_collagen_features_for_slide(self, metrics: DataFrame):
        """"""
        if (metrics.shape[0] < 1) or self._slidename_in_df_index(opj(
            self.output_dir,
            'perDatasetSlideSummaries',
            'CollagenFeatureSummary_Means.csv'
        )):
            return

        # restrict to topk salient rois
        metrics = metrics.sort_values(
            'Saliency.SaliencyScore', ascending=False
        )
        if self.topk_salient_rois is not None:
            metrics = metrics[:self.topk_salient_rois]
        # metrics are weighted by saliency (saliency itself is unweighted)
        saliency = metrics.loc[:, 'Saliency.SaliencyScore'].values
        means = {'Saliency.SaliencyScore': np.nanmean(saliency)}
        stds = {'Saliency.SaliencyScore': np.nanstd(saliency)}
        # keep relevant columns
        cols = [j for j in metrics.columns if not j.startswith('Saliency.')]
        metrics = metrics.loc[:, cols]
        max_saliency = saliency.max()
        if max_saliency > 0:
            saliency = saliency / max_saliency
        else:
            saliency = np.ones(saliency.size)
        for k in cols:
            col_vals = metrics.loc[:, k].values.copy()
            keep = np.isfinite(metrics.loc[:, k])
            if not keep.all():
                means[k], stds[k] = 0., 0.
            else:
                wmn, wst = weighted_avg_and_std(col_vals[keep], saliency[keep])
                means[k], stds[k] = float(wmn), float(wst)
        # now save
        append_row_to_df_or_create_it(
            opj(
                self.output_dir,
                'perDatasetSlideSummaries',
                'CollagenFeatureSummary_Means.csv'
            ),
            DataFrame.from_records([means], index=[self.slide_name]),
        )
        append_row_to_df_or_create_it(
            opj(
                self.output_dir,
                'perDatasetSlideSummaries',
                'CollagenFeatureSummary_Stds.csv'
            ),
            DataFrame.from_records([stds], index=[self.slide_name]),
        )

    def _slidename_in_df_index(self, where: str) -> bool:
        """Check if slidename is in the index of a dataframe."""
        if not os.path.isfile(where):
            return False

        existing = read_csv(where, usecols=[0]).iloc[:, 0].tolist()
        return self.slide_name in existing

    @collect_errors()
    def _calculate_features_for_this_roi(
        self, rgb: Image.Image, stroma_semantic: np.ndarray,
    ) -> Dict[str, float]:
        """"""
        all_features = {}
        collagen_edges, collagen_grayscale = self._isolate_collagen_fibers(
            rgb, stromal_matrix=stroma_semantic == self._ncd['BACKGROUND']
        )
        eprops = self._calculate_collagen_edge_regionprops(collagen_edges)
        # stromal cellularity
        total = np.sum(stroma_semantic > 0)
        all_features.update(
            self._calculate_stromal_cellularity_features(
                stroma_semantic, total=total,
            )
        )
        # calculate intensity features from stromal matrix
        all_features.update(
            self._calculate_collagen_intensity_features(collagen_grayscale)
        )
        # Calculate features for collagen fibril edge (separation of fibres)
        all_features.update(
            self._summarize_per_fibril_edge_regionprops(eprops, total=total)
        )
        entfeats, cooc = self._calculate_collagen_fibril_entropy(eprops=eprops)
        all_features.update(entfeats)
        # for debugging, visualize
        if self._debug:
            self._visualize_collagen_detection_for_debugging(
                rgb=rgb,
                collagen_grayscale=collagen_grayscale,
                collagen_edges=collagen_edges,
                eprops=eprops,
                orient_cooc=cooc,
                entfeats=entfeats,
            )
        return all_features

    @staticmethod
    def _calculate_collagen_fibril_entropy(eprops: DataFrame, n_bins=18):
        """
        Calculate entropy in collagen fiber length & direction. See:

        Li H, et al. Collagen fiber orientation disorder from H&E images is
        prognostic for early stage breast cancer: clinical trial validation.
        NPJ Breast Cancer. 2021 Aug 6;7(1):1-0.

        """
        feats = {}
        # This is restricted to straight fibers
        props = eprops.loc[
            eprops.loc[:, 'CollagenFiberEdges.Special.IsStraightFiber'], :
        ].copy()
        props.loc[:, 'AbsAngle'] = np.abs(
            props.loc[:, 'CollagenFiberEdges.Special.Orientation']
        )
        # * Histogram * entropy in length fibers
        hist, _ = np.histogram(
            props.loc[:, 'CollagenFiberEdges.Morphology.MajorAxisLength'].values,
            bins=n_bins,
        )
        feats['CollagenFiberEdges.HistEntropyOfLength'] = entropy(
            hist / np.sum(hist, dtype=np.float32)
        )
        # * Histogram * entropy in orientation of fibers (unweighted)
        hist, _ = np.histogram(
            np.abs(props.loc[:, 'AbsAngle'].values),
            bins=n_bins,
            range=(0, np.pi / 2),
        )
        feats['CollagenFiberEdges.HistEntropyOfOrientation'] = (
            entropy(hist / np.sum(hist, dtype=np.float32))
        )
        # Length-weighted co-occurence matrix of orientations (n_bins, n_bins)
        # This is used to calculate the Collagen Fiber Orientation Disorder of
        # Tumor Associates Stroma (CFOD-TS), as described in Supplementary
        # Equation 1 of Anant's group
        props.loc[:, 'AbsAngleBin'] = np.digitize(
            props.loc[:, 'AbsAngle'].values,
            bins=np.arange(0, np.pi / 2, np.pi / (2 * n_bins))
        ) - 1
        cooc = np.zeros((n_bins, n_bins), dtype=np.float32)
        for _, row in props.iterrows():
            # All other collagen fibers in this ROI "co-occured" with this one.
            # Consistent with Anant's paper, we weigh by this fiber's length
            weighted_hist = np.float32(hist.copy())
            weighted_hist[row['AbsAngleBin']] -= 1  # exclude the fiber itself
            weighted_hist *= row['CollagenFiberEdges.Morphology.MajorAxisLength']
            cooc[row['AbsAngleBin']] += weighted_hist
        if cooc.sum() > 0:
            cooc /= cooc.sum()
            # Now calculate entropy.
            # https://github.com/DigitalSlideArchive/HistomicsTK/blob/master/
            # histomicstk/features/compute_haralick_features.py#L346
            rcooc = cooc.ravel()
            feats['CollagenFiberEdges.EntropyOfOrientationAnant'] = -np.dot(
                rcooc, np.log2(rcooc + 1e-6)
            )
        else:
            feats['CollagenFiberEdges.EntropyOfOrientationAnant'] = np.nan

        return feats, cooc

    @staticmethod
    def _summarize_per_fibril_edge_regionprops(eprops: DataFrame, total: int):
        """summarize per-fibril edge morphometric features"""
        summary = {
            "CollagenFiberEdges.Counts.NormNoOfEdges":
                eprops.shape[0] / total,
            "CollagenFiberEdges.Counts.NormNoOfStraightEdges":
                np.sum(eprops.loc[:, 'CollagenFiberEdges.Special.IsStraightFiber'])
                / total,
        }
        summary["CollagenFiberEdges.Counts.Straight2AllEdges"] = (
            summary["CollagenFiberEdges.Counts.NormNoOfStraightEdges"]
            / summary["CollagenFiberEdges.Counts.NormNoOfEdges"]
        )
        morph = [
            c for c in eprops.columns
            if c.startswith('CollagenFiberEdges.Morphology.')
        ]
        summary.update({
            f"{c}.Mean": value for c, value in
            zip(morph, np.nanmean(eprops.loc[:, morph], axis=0))
        })
        summary.update({
            f"{c}.Std": value for c, value in
            zip(morph, np.nanstd(eprops.loc[:, morph], axis=0))
        })
        return summary

    def _calculate_collagen_edge_regionprops(self, collagen_edges: DataFrame):
        """"""
        edge_feats = DataFrame(
            regionprops_table(
                collagen_edges, properties=self._region_prop_names,
            )
        )
        edge_feats.index = edge_feats.loc[:, 'label']
        edge_feats.drop(columns=['label'], inplace=True)
        edge_feats.loc[:, 'Morphology.MinorMajorAxisRatio'] = (
                edge_feats.loc[:, 'minor_axis_length'] /
                np.maximum(1e-8, edge_feats.loc[:, 'major_axis_length'])
        )
        edge_feats = edge_feats.rename(columns={
            'area': 'Morphology.Area',
            'major_axis_length': 'Morphology.MajorAxisLength',
            'minor_axis_length': 'Morphology.MinorAxisLength',
            'orientation': 'Special.Orientation',
        })
        edge_feats.loc[:, 'Special.IsStraightFiber'] = (
            edge_feats.loc[:, 'Morphology.MinorMajorAxisRatio']
            < self.max_axis_ratio_for_fiber
        )
        edge_feats = edge_feats.rename(
            columns={c: f"CollagenFiberEdges.{c}" for c in edge_feats.columns},
        )

        return edge_feats

    @staticmethod
    def _calculate_collagen_intensity_features(collagen_grayscale: np.ndarray):
        """"""
        feats = compute_intensity_features(
            im_label=0 + (collagen_grayscale > 0),
            im_intensity=np.uint8(255 * collagen_grayscale),
            num_hist_bins=18,
        )
        return {
            f"CollagenStaining{k}": v
            for k, v in feats.iloc[0, :].to_dict().items()
        }

    def _calculate_stromal_cellularity_features(
        self, stroma_semantic: np.ndarray, total: int = None,
    ):
        """"""
        total = total or np.sum(stroma_semantic > 0)
        feats = {
            f"StromalCellularity.AcellularStroma":
                np.sum(stroma_semantic == self._ncd['BACKGROUND']),
            f"StromalCellularity.FibroblastDensity":
                np.sum(stroma_semantic == self._ncd['StromalCellNOS']),
            f"StromalCellularity.TILsDensity":
                np.sum(stroma_semantic == self._ncd['TILsCell']),
        }
        return {k: v / total for k, v in feats.items()}

    @staticmethod
    def _isolate_collagen_fibers(rgb: Image.Image, stromal_matrix: np.ndarray):
        """"""
        mask_out = ~stromal_matrix
        collagen_grayscale = 1 - rgb2gray(np.uint8(rgb))
        collagen_grayscale[mask_out] = 0
        # Detect edges (canny edge already smoothes)
        collagen_edges = canny(collagen_grayscale, sigma=1.5)
        # exclude nuclear and tissue edges (we only want collagen fibres)
        other_edges = binary_dilation(mask_out, selem=np.ones((5, 5)))
        eroded = binary_erosion(mask_out, selem=np.ones((5, 5)))
        other_edges[eroded] = False
        collagen_edges[other_edges] = False
        collagen_edges = sklabel(collagen_edges, connectivity=2)
        collagen_edges = remove_small_objects(
            collagen_edges, min_size=16, connectivity=2,
        )
        return collagen_edges, collagen_grayscale

    def _visualize_collagen_detection_for_debugging(
        self,
        rgb,
        collagen_grayscale,
        collagen_edges,
        eprops,
        orient_cooc,
        entfeats,
    ):
        import matplotlib.ticker as ticker
        from matplotlib.colors import ListedColormap
        # from ctme.GeneralUtils import save_json

        savename = opj(
            self.output_dir,
            "perSlideCollagenFeatures",
            f"DEBUG.{self.slide_name}",
            f"CFOD=%.2f_{self._roiname}"
            % entfeats['CollagenFiberEdges.EntropyOfOrientationAnant']
        )
        # thinken edges for aesthetics & differentiate straight from curved
        straight_labels = list(eprops.index[
            eprops.loc[:, 'CollagenFiberEdges.Special.IsStraightFiber']
        ])
        straight_edges = np.in1d(
            collagen_edges, straight_labels).reshape(collagen_edges.shape)
        straight_edges = binary_dilation(straight_edges, selem=np.ones((3, 3)))
        coll_edges = 0 + (collagen_edges > 0)
        coll_edges[straight_edges] = 2

        _, ax = plt.subplots(2, 2, figsize=(10, 10))
        plt.tick_params(
            left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False,
        )
        ax[0, 0].imshow(rgb)
        offwhite = VisConfigs.NUCLEUS_CMAP.colors[-1]
        for axis in [ax[0, 1], ax[1, 0]]:
            axis.imshow(
                np.zeros(collagen_grayscale.shape),
                cmap=ListedColormap([offwhite])
            )
        ax[0, 1].imshow(
            np.ma.masked_array(collagen_grayscale, collagen_grayscale == 0),
            cmap='RdPu',
        )
        ax[1, 0].imshow(
            coll_edges,
            cmap=ListedColormap([offwhite, 'hotpink', 'deeppink']),
            interpolation='nearest'
        )
        ax[1, 1].imshow(orient_cooc, cmap='plasma', interpolation='nearest')
        for axis in ax.ravel():
            axis.xaxis.set_major_locator(ticker.NullLocator())
            axis.yaxis.set_major_locator(ticker.NullLocator())
        plt.tight_layout()
        plt.savefig(f"{savename}.png")
        plt.close()
        # save_json(
        #     {k: float(v) for k, v in entfeats.items()}, f"{savename}.json"
        # )

    @collect_errors()
    def _fetch_roi_rgb(self, wsi_base_coords: Dict) -> Image.Image:
        """"""
        mm = self.mpp / 1000
        rgb, _ = self._wsi_tilesource.getRegion(
            region=dict(
                left=wsi_base_coords['left'],
                top=wsi_base_coords['top'],
                right=wsi_base_coords['right'],
                bottom=wsi_base_coords['bottom'],
                units="base_pixels",
            ),
            scale=dict(mm_x=mm, mm_y=mm),
            format=large_image.tilesource.TILE_FORMAT_PIL,
            jpegQuality=100,
        )
        rgb = rgb.convert("RGB")

        return rgb.resize((self.roi_size, self.roi_size))

    def _parse_coords_relative_to_wsi_base(self, tile_mask_info):
        """"""
        xmin = int(tile_mask_info['x'] * self._mask2wsi_sf)
        ymin = int(tile_mask_info['y'] * self._mask2wsi_sf)
        side = int(self.roi_size / self._wsi2target_sf)
        return {
            'left': xmin,
            'top': ymin,
            'right': xmin + side,
            'bottom': ymin + side,
        }

    @collect_errors()
    def _isolate_stromal_components(self, tile_info: Dict) -> Image.Image:
        """"""
        tile_mask = [
            arr[..., 0] for ch, arr in enumerate(
                np.dsplit(np.array(tile_info['tile']), 3)
            )
            if ch < 2
        ]

        # most tiles are empty/irrelevant
        npixels = self._mask_tile_size ** 2
        min_stroma = self.min_stroma_ratio * npixels
        min_tumor = self.min_tumor_ratio * npixels
        if np.count_nonzero(tile_mask) < min_stroma:
            return

        # map some classes as needed
        for region1, region2 in self._region_class_map.items():
            tile_mask[0][tile_mask[0] == self._rcd[region1]] = (
                self._rcd[region2]
            )
        for nucl1, nucl2 in self._nucleus_class_map.items():
            tile_mask[1][tile_mask[1] == self._ncd[nucl1]] = (
                self._ncd[nucl2]
            )

        # if not enough tumor/stroma, return None
        tumor_region = tile_mask[0] == self._rcd['TUMOR']
        stromal_region = tile_mask[0] == self._rcd['STROMA']
        if any([
            np.sum(stromal_region) < min_stroma,
            np.sum(tumor_region) < min_tumor,
        ]):
            return

        # semantic segment. of stroma - collagen is self._ncd['BACKGROUND']
        stroma_semantic = tile_mask[1].copy()
        stroma_semantic[~stromal_region] = 0

        return {
            'saliency_score': np.mean(tumor_region) * np.mean(stromal_region),
            'stroma_semantic': np.array(
                Image.fromarray(stroma_semantic).resize(
                    (self.roi_size, self.roi_size), resample=Image.NEAREST
                ),
                dtype=np.uint8,
            ),
        }

    @cached_property
    def _mask2wsi_sf(self) -> float:
        return self._mask2target_sf / self._wsi2target_sf

    @cached_property
    def _wsi2target_sf(self) -> float:
        """Scale factor from WSI MPP to desired mpp"""
        return 1e3 * self._wsi_tilesource.getMetadata()['mm_x'] / self.mpp

    @cached_property
    def _mask_tile_size(self) -> int:
        return int(self._mask2target_sf * self.roi_size)

    @cached_property
    def _mask2target_sf(self) -> float:
        """Scale factor from mask MPP to desired mpp"""
        return 1e3 * self._mask_tilesource.getMetadata()['mm_x'] / self.mpp

    @cached_property
    def _mask_tilesource(self):
        """"""
        return large_image.getTileSource(
            opj(self.slide_dir, self.slide_name + '.tif')
        )

    @cached_property
    def _wsi_tilesource(self):
        """"""
        return large_image.getTileSource(self.wsi_file)


class SlideRegionFeatureExtractor(object):
    """
    Extracts region features from MuTIsWSIRunner ouput.
    """
    def __init__(
        self,
        slide_dir: str,
        output_dir: str,
        *,
        feature_sets: Set[str] = None,
        region_prop_names: Iterable[str] = None,
        neighborhood_distances: Iterable[int] = (64, 128),
        slide_name: str = None,
        mpp: float = 2.0,
        monitor: str = "",
        logger: Any = None,
        _discrete_regions: Iterable[str] = ('TUMOR', 'TILS'),
        _relevant_regions: Iterable[str] = None,
        _relevant_nuclei: Iterable[str] = None,
        _min_discrete_region_area: int = 64 ** 2,
        _min_region_hole_area: Union[int, None] = 24 ** 2,
        _region_class_map: Dict[str, str] = None,
        _nucleus_class_map: Dict[str, str] = None,
    ):
        """
        This represents the slide as a feature matrix summarizing
        its semantic segmantation mask.
        """
        self.slide_dir = slide_dir
        self.output_dir = output_dir
        self.slide_name = slide_name or os.path.basename(slide_dir)
        self.mpp = mpp
        self.monitor = monitor
        self.logger = logger or logging.getLogger(__name__)

        self.features_aware_of_center = {
            "nuclear_composition",
        }
        self.features_aware_of_surround = {
            "neighborhood_composition_regions",
            "neighborhood_composition_nuclei",
        }
        self.features_aware_of_cs = self.features_aware_of_center.union(
            self.features_aware_of_surround
        )
        available_feature_sets = {
            "morphology",
        }.union(self.features_aware_of_cs)
        self.feature_sets = feature_sets or available_feature_sets
        not_recognized = self.feature_sets.difference(available_feature_sets)
        assert len(not_recognized) < 1, (
            "These feature sets you requested are not recognized: "
            f"{not_recognized}"
        )
        self._region_prop_names = region_prop_names or (
            'label',
            'area',
            'convex_area',
            'eccentricity',
            'equivalent_diameter',
            'euler_number',
            'extent',
            'feret_diameter_max',
            'filled_area',
            'major_axis_length',
            'minor_axis_length',
            'perimeter',
            'solidity',
        )
        self.neighborhood_distances = neighborhood_distances

        self._discrete_regions = _discrete_regions
        self._relevant_regions = _relevant_regions or (
            'TUMOR', 'STROMA', 'TILS', 'JUNK', 'WHITE'
        )
        self._relevant_nuclei = _relevant_nuclei or (
            'CancerEpithelium',
            'NormalEpithelium',
            'StromalCellNOS',
            'TILsCell',
            'ActiveTILsCell',
            'UnknownOrAmbiguousCell',
            'BACKGROUND',
        )
        self._min_discrete_region_area = _min_discrete_region_area
        self._min_region_hole_area = _min_region_hole_area

        # codes and class map for most relevant classes
        self._region_class_map = _region_class_map or {'NORMAL': 'TUMOR'}
        self._rcd = RegionCellCombination.REGION_CODES
        self._rcd2 = {reg: self._rcd[reg] for reg in self._relevant_regions}
        self._nucleus_class_map = _nucleus_class_map or {
            'ActiveStromalCellNOS': 'StromalCellNOS',
        }
        self._ncd = RegionCellCombination.NUCLEUS_CODES
        self._ncd2 = {nuc: self._ncd[nuc] for nuc in self._relevant_nuclei}

        # tile size used for loading the semantic mask
        self._tile_size = 2048

    def run(self):
        """"""
        object_feats = self.load_or_extract_features_per_region_object()
        self.summarize_region_features_for_slide(object_feats)

    @collect_errors()
    def summarize_region_features_for_slide(self, object_feats: DataFrame):
        """
        Summarize slide regions as a single row in the dataset.
        """
        if self._slidename_not_in_df_index(opj(
            self.output_dir,
            'perDatasetSlideSummaries',
            'RegionFeatureSummary_Means.csv'
        )):
            summary = [
                self._summarize_features_per_region_class(object_feats, region)
                for region in self._discrete_regions
            ]
            append_row_to_df_or_create_it(
                opj(
                    self.output_dir,
                    'perDatasetSlideSummaries',
                    'RegionFeatureSummary_Means.csv'
                ),
                concat([j[0] for j in summary], axis=1),
            )
            append_row_to_df_or_create_it(
                opj(
                    self.output_dir,
                    'perDatasetSlideSummaries',
                    'RegionFeatureSummary_Stds.csv'
                ),
                concat([j[1] for j in summary], axis=1),
            )

    def load_or_extract_features_per_region_object(self) -> DataFrame:
        """
        DataFrame representation of features (one row per tissue region).
        """
        where = opj(
            self.output_dir, 'perSlideRegionFeatures', f"{self.slide_name}.csv"
        )
        if os.path.isfile(where):
            return read_csv(where, index_col=0)

        return self._extract_features_per_region_object()

    # HELPERS -----------------------------------------------------------------

    @collect_errors()
    def _summarize_features_per_region_class(
        self, object_feats: DataFrame, region: str
    ) -> Tuple[DataFrame]:
        """"""
        feature_cols = [
            c for c in object_feats.columns if not c.startswith('Identifier.')
        ]
        feats = object_feats.loc[
            object_feats.loc[:, 'Identifier.RegionClass'] == region,
            feature_cols
        ]
        feats = feats.rename(
            columns={c: f"{c}.{region}" for c in feats.columns},
        )
        means = DataFrame(
            np.nanmean(feats, axis=0)[None, ...],
            columns=feats.columns,
            index=[self.slide_name],
        )
        means.loc[:, f"Global.NoOfRegions.{region}"] = feats.shape[0]
        stds = DataFrame(
            np.nanstd(feats, axis=0)[None, ...],
            columns=feats.columns,
            index=[self.slide_name],
        )

        return means, stds

    @collect_errors()
    def _extract_features_per_region_object(self) -> DataFrame:
        """"""
        features = []

        if "morphology" in self.feature_sets:
            features.append(self._extract_region_morphology())

        if len(self.feature_sets.intersection(self.features_aware_of_cs)) > 0:
            features.append(
                self._extract_features_aware_of_center_or_surround()
            )

        features = concat(features, axis=1)
        features = self._fix_and_move_identifier_columns(features)
        features.to_csv(
            opj(
                self.output_dir,
                'perSlideRegionFeatures',
                f"{self.slide_name}.csv"
            ),
            index_label='',
        )

        return features

    @collect_errors()
    def _extract_features_aware_of_center_or_surround(self) -> DataFrame:
        """
        Extract features aware of center (self) and surrounding neighborhood
        for all distinct tissue region objects.
        """
        self.logger.info(
            f"{self.monitor}: extract features aware of center or surround"
        )

        feats = {}
        for region in self._discrete_regions:
            self.logger.info(f"{self.monitor}: center/surround for {region}")
            objmask = self._region_object_masks[region].tocoo(copy=False)
            feats.update({
                f"{region}.{objcode}":
                    self._extract_features_aware_of_cs_for_region_object(
                        objmask, objcode=objcode, region=region,
                    )
                for objcode in np.unique(objmask.data)
            })

        return DataFrame.from_records(feats).T

    @collect_errors()
    def _extract_features_aware_of_cs_for_region_object(
        self, objmask: csr_matrix, objcode: int, region: str
    ) -> Dict[str, Union[float, int]]:
        """
        Extract features aware of center (self) and surrounding neighborhood
        for a single distinct tissue region object.

        Parameters
        ----------
        objmask: csr_matrix
            self._object_masks[region].tocoo(copy=False)
        objcode: int
            object code in the region object mask (`objmask` parameter).
        region: str
            region type of object (eg TUMOR).

        Returns
        -------
        Dict[str, Union[float, int]]
            keys are features names, values are fature values.

        """
        keep = objmask.data == objcode

        # coordinates relative to big mask
        rows = objmask.row[keep]
        cols = objmask.col[keep]

        # this will hold center and/or surround features
        feats = {}

        # extent of object itself (tight, no surrounding)
        rmin = rows.min()
        cmin = cols.min()
        rmax = rows.max()
        cmax = cols.max()

        # features for object itself (eg nuclei pixels within tumor region)
        if len(
            self.feature_sets.intersection(self.features_aware_of_center)
        ) > 0:
            one_obj = self._region_object_masks[region][rmin:rmax, cmin:cmax]
            one_obj = one_obj.toarray() == objcode
            if "nuclear_composition" in self.features_aware_of_center:
                obj_semantic = self._nucleus_semantic[rmin:rmax, cmin:cmax]
                obj_semantic = obj_semantic.toarray().copy()
                obj_semantic[~one_obj] = 0
                feats.update(
                    self._summarize_nuclear_composition_for_region_object(
                        obj_semantic
                    )
                )

        # features for object surrounding
        if len(
            self.feature_sets.intersection(self.features_aware_of_surround)
        ) > 0:
            feats.update(
                self._summarize_region_object_neighborhood(
                    objcode=objcode,
                    region=region,
                    objmask_shape=objmask.shape,
                    rmin=rmin,
                    cmin=cmin,
                    rmax=rmax,
                    cmax=cmax,
                )
            )

        return feats

    @collect_errors()
    def _summarize_nuclear_composition_for_region_object(
        self, obj_semantic: np.ndarray,
    ) -> Dict[str, Union[float, int]]:
        """Summarize nuclei enclosed within region object.

        Parameters
        ----------
        obj_semantic: np.ndarray
            semantic segmentation of the nuclei enclosed within the tissue
            region object, where zeros are pixels outside the region object,
            and pixel values encode the nuclear classification.

        Returns
        -------
        Dict[str, Union[float, int]]
            keys are features names, values are fature values.

        """
        total = np.sum(obj_semantic > 0)

        return {
            f"NuclearComposition.NormalizedPixelCount.{category}": (
                _pixsum(obj_semantic, cd) / total
            )
            for category, cd in self._ncd2.items()
        }

    def _summarize_region_object_neighborhood(
        self,
        objcode: int,
        *,
        region: str,
        objmask_shape: Tuple[int],
        rmin: int,
        cmin: int,
        rmax: int,
        cmax: int,
    ) -> Dict[str, Union[float, int]]:
        """
        Summarize the neighborhood of a single tissue region object.

        Parameters
        ----------
        objcode: int
            object code in the region object mask.
        region: str
            region type of object (eg TUMOR).
        objmask_shape: Tuple[int]
            row, col shape of object mask.
        rmin: int
            minimum row coordinate relative to self._object_masks[region]
        cmin: int
            minimum col coordinate relative to self._object_masks[region]
        rmax: int
            maximum row coordinate relative to self._object_masks[region]
        cmax: int
            maximum col coordinate relative to self._object_masks[region]

        Returns
        -------
        Dict[str, Union[float, int]]
            keys are features names, values are fature values.

        """
        summary = {}
        for distance in self.neighborhood_distances:
            summary.update(
                self._summarize_object_neighborhood_for_distance(
                    objcode=objcode,
                    distance=distance,
                    region=region,
                    objmask_shape=objmask_shape,
                    rmin=rmin,
                    cmin=cmin,
                    rmax=rmax,
                    cmax=cmax,
                )
            )

        return summary

    @collect_errors()
    def _summarize_object_neighborhood_for_distance(
        self,
        objcode: int,
        distance: int,
        *,
        region: str,
        objmask_shape: Tuple[int],
        rmin: int,
        cmin: int,
        rmax: int,
        cmax: int,
    ) -> Dict[str, Union[float, int]]:
        """
        Summarize the neighborhood of a single object within x pixels.

        Parameters
        ----------
        objcode: int
            object code in the region object mask.
        distance: int
            neighborhood distance around object
        region: str
            region type of object (eg TUMOR).
        objmask_shape: Tuple[int]
            row, col shape of object mask.
        rmin: int
            minimum row coordinate relative to self._object_masks[region]
        cmin: int
            minimum col coordinate relative to self._object_masks[region]
        rmax: int
            maximum row coordinate relative to self._object_masks[region]
        cmax: int
            maximum col coordinate relative to self._object_masks[region]

        Returns
        -------
        Dict[str, Union[float, int]]
            keys are features names, values are fature values.

        """
        # expanded coordinates
        rmin = max(0, rmin - distance)
        cmin = max(0, cmin - distance)
        rmax = min(objmask_shape[0], rmax + distance)
        cmax = min(objmask_shape[1], cmax + distance)
        # isolate center and surround mask for object
        single_obj = self._region_object_masks[region][rmin:rmax, cmin:cmax]
        single_obj = single_obj.toarray() == objcode
        neighborhood_mask = get_region_within_x_pixels(
            center_mask=single_obj,
            surround_mask=~single_obj,
            max_dist=distance,
        )
        # summarize semantic mask for neighborhood
        total = np.count_nonzero(neighborhood_mask)
        summary = {
            f"Neighborhood.PixelCount.Surround-ALL.Distance-{distance}": total
        }
        for isregion in [True, False]:
            semantic = (
                self._region_semantic if isregion else self._nucleus_semantic
            )
            summary.update(
                self._summarize_neighborhood_pixels_per_category(
                    neighborhood_mask=neighborhood_mask,
                    semantic=semantic[rmin:rmax, cmin:cmax],
                    distance=distance,
                    isregion=isregion,
                    total=total,
                )
            )

        return summary

    @collect_errors()
    def _summarize_neighborhood_pixels_per_category(
        self,
        neighborhood_mask: np.ndarray,
        semantic: csr_matrix,
        distance: Union[int, float],
        isregion: bool = True,
        total: int = None,
    ) -> Dict[str, Union[float, int]]:
        """"""
        codemap = self._rcd2 if isregion else self._ncd2
        total = (
            total if total is not None else np.count_nonzero(neighborhood_mask)
        )
        neighborhood_semantic = semantic.toarray().copy()
        neighborhood_semantic[~neighborhood_mask] = 0

        return {
            f"Neighborhood.{'Regions' if isregion else 'Nuclei'}."
            f"NormalizedPixelCount.Surround-{category}."
            f"Distance-{distance}": _pixsum(neighborhood_semantic, cd) / total
            for category, cd in codemap.items()
        }

    def _extract_region_morphology(self) -> DataFrame:
        """
        Extract morhology of each distinct tissue region in the slide.
        """
        self.logger.info(f"{self.monitor}: extract region morphologies")
        features = [
            self._extract_morphology_for_region_class(region)
            for region in self._discrete_regions
        ]
        return concat(features, axis=0)

    @collect_errors()
    def _extract_morphology_for_region_class(self, region: str) -> DataFrame:
        """
        Extract morphology of tissue regions belonging to a class.
        """
        self.logger.info(f"{self.monitor}: morphology for {region}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feats = DataFrame(
                regionprops_table(
                    self._region_object_masks[region].toarray(),
                    properties=self._region_prop_names,
                    extra_properties=[self.fractal_dimension],
                )
            )
        feats = feats.rename(columns={
            'label': 'Identifier.label',
            'area': 'Size.Area',
            'convex_area': 'Size.ConvexHullArea',
            'filled_area': 'Size.FilledArea',
            'major_axis_length': 'Size.MajorAxisLength',
            'minor_axis_length': 'Size.MinorAxisLength',
            'perimeter': 'Size.Perimeter',
            'eccentricity': 'Shape.Eccentricity',
            'equivalent_diameter': 'Shape.EquivalentDiameter',
            'euler_number': 'Shape.EulerNumber',
            'extent': 'Shape.Extent',
            'feret_diameter_max': 'Shape.MaxFeretDiameter',
            'solidity': 'Shape.Solidity',
            'fractal_dimension': 'Shape.FractalDimension',
        })
        feats.loc[:, 'Shape.MinorMajorAxisRatio'] = (
            feats.loc[:, 'Size.MinorAxisLength']
            / feats.loc[:, 'Size.MajorAxisLength']
        )
        feats.loc[:, 'Shape.Circularity'] = (
            4 * np.pi * feats.loc[:, 'Size.Area']
            / (feats.loc[:, 'Size.Perimeter'] ** 2)
        )
        feats.loc[:, 'Identifier.RegionClass'] = (
            region if feats.shape[0] > 0 else []
        )
        feats.index = (
            feats.loc[:, 'Identifier.label'].map(lambda x: f"{region}.{x}")
        )
        feats = self._fix_and_move_identifier_columns(feats)

        return feats

    @staticmethod
    def _fix_and_move_identifier_columns(df: DataFrame) -> DataFrame:
        """"""
        if 'Identifier.RegionClass' not in df.columns:
            df.loc[:, 'Identifier.RegionClass'] = [
                j.split('.')[0] for j in df.index
            ]
        first = [c for c in df.columns if c.startswith('Identifier.')]
        second = [c for c in df.columns if c not in first]

        return df.loc[:, first + second]

    @staticmethod
    def fractal_dimension(x):
        try:
            return _fractal_dimension(x)
        except:  # noqa
            return np.nan

    def _get_objects_from_binmask(self, binmask: np.ndarray) -> np.ndarray:
        """
        Get cleaned up connected components from a binary mask.
        """
        obj_mask = binary_dilation(binmask, selem=np.ones((5, 5)))
        obj_mask = remove_small_holes(
            obj_mask,
            area_threshold=self._min_region_hole_area,
            connectivity=2,
        )
        obj_mask = sklabel(obj_mask, connectivity=2)
        obj_mask = remove_small_objects(
            obj_mask,
            min_size=self._min_discrete_region_area,
            connectivity=2,
        )

        return obj_mask

    def _slidename_not_in_df_index(self, where: str) -> bool:
        """Check if slidename is not in the index of a dataframe."""
        if not os.path.isfile(where):
            return True

        existing = read_csv(where, usecols=[0]).iloc[:, 0].tolist()
        return self.slide_name not in existing

    @cached_property
    def _region_object_masks(self) -> Dict[str, csr_matrix]:
        """
        Labeled masks for each of the region classes with discrete boundaries.
        Note that we store the arrays to csr_matrix which is a sparse
        matrix representation to take less memory
        """
        return {
            region: csr_matrix(
                self._get_objects_from_binmask(
                    (self._region_semantic == self._rcd[region]).toarray()
                )
            )
            for region in self._discrete_regions
        }

    @property
    def _nucleus_semantic(self) -> csr_matrix:
        return self._semantic_masks[1]

    @property
    def _region_semantic(self) -> csr_matrix:
        return self._semantic_masks[0]

    @cached_property
    def _semantic_masks(self) -> List[csr_matrix]:
        """
        Semantic segmentation mask for slide at desired MPP.

        This is list of (m, n) masks, where first element is semantic region
        segmentation, and second is nuclei semantic segmentation.

        IMPORTANT NOTE: We can NOT trust large_image itself to do the
        resizing for us here because it uses LANCZOS but we we want NEAREST!
        https://github.com/girder/large_image/blob/ ...
        4e1bb5b91d4db8ac5ee56916d0ae190dedfd2259/large_image/tilesource/ ...
        base.py#L1606-L1610
        """
        # init empty semantic mask
        meta = self._tilesource.getMetadata()
        semantic = [
            Image.new(
                mode='L',
                size=[
                    1 + int(meta[f"size{j}"] * self._sf) for j in ('X', 'Y')
                ],
                color=0,
            )
            for _ in range(2)
        ]
        # iterate through tiles, resize and paste
        tidx = -1
        for tile_info in self._tilesource.tileIterator(
            tile_size=dict(width=self._tile_size, height=self._tile_size),
            tile_overlap=dict(x=0, y=0),
            format=large_image.tilesource.TILE_FORMAT_PIL
        ):
            tidx += 1
            if tidx % 100 == 0:
                self.logger.info(f"{self.monitor}: loading tile {tidx}")

            tile_mask = self._process_wsi_mask_tile(tile_info)
            if tile_mask is not None:
                self._paste_wsi_mask_tile(
                    tile_info=tile_info,
                    tile_mask=tile_mask,
                    semantic=semantic,
                )

        semantic = [
            csr_matrix(np.uint8(channel), dtype=np.uint8)
            for channel in semantic
        ]
        self.logger.info(f"{self.monitor}: Done loading semantic mask.")

        return semantic

    @collect_errors()
    def _paste_wsi_mask_tile(
        self,
        tile_info: Dict,
        tile_mask: List[Image.Image],
        semantic: List[Image.Image],
    ) -> None:
        """"""
        for ch, channel in enumerate(semantic):
            channel.paste(
                tile_mask[ch],
                box=(
                    int(tile_info['x'] * self._sf),
                    int(tile_info['y'] * self._sf)
                ),
            )

    @collect_errors()
    def _process_wsi_mask_tile(self, tile_info: Dict) -> List[Image.Image]:
        """
        Process the semantic masks for one tile from the semantic WSI mask.
        The returned list has two elements, the first being the region semantic
        mask, while the second being the nucleus semantic mask.
        """
        tile_mask = [
            arr[..., 0] for ch, arr in enumerate(
                np.dsplit(np.array(tile_info['tile']), 3)
            )
            if ch < 2
        ]

        if tile_mask[0].sum() < 2:
            return

        # map some classes as needed
        for region1, region2 in self._region_class_map.items():
            tile_mask[0][tile_mask[0] == self._rcd[region1]] = (
                self._rcd[region2]
            )
        for nucl1, nucl2 in self._nucleus_class_map.items():
            tile_mask[1][tile_mask[1] == self._ncd[nucl1]] = (
                self._ncd[nucl2]
            )

        return [
            Image.fromarray(arr).resize(
                (self._new_tile_size, self._new_tile_size),
                resample=Image.NEAREST
            )
            for arr in tile_mask
        ]

    @cached_property
    def _new_tile_size(self) -> int:
        return int(self._sf * self._tile_size)

    @cached_property
    def _sf(self) -> float:
        """Scale factor from mask MPP to desired mpp"""
        return 1e3 * self._tilesource.getMetadata()['mm_x'] / self.mpp

    @cached_property
    def _tilesource(self):
        """"""
        return large_image.getTileSource(
            opj(self.slide_dir, self.slide_name + '.tif')
        )


class SlideNucleiFeatureExtractor(object):
    """
    Extracts nuclear features from MuTIsWSIRunner ouput.
    """
    def __init__(
        self,
        slide_dir: str,
        output_dir: str,
        *,
        feature_sets: Set[str] = None,
        slide_name: str = None,
        monitor: str = "",
        logger: Any = None,
        kdtree_kwargs: dict = None,
        ripleyk_radii: Iterable[Union[int, float]] = None,
        additional_ripleyk_kwargs: dict = None,
        topk_salient_rois: int = 128,
        _roi_side: int = 1024,
        _min_nucl_size: int = 5,
        _nucleus_superclasses: Iterable[str] = None,
    ):
        """
        This summarizes the slide as a nuclear feature matrix, as well
        as some basic region pixel counts.
        """
        self.slide_dir = slide_dir
        self.output_dir = output_dir
        self.slide_name = slide_name or os.path.basename(slide_dir)
        self.monitor = monitor
        self.logger = logger or logging.getLogger(__name__)
        self.kdtree_kwargs = kdtree_kwargs
        self.ripleyk_radii = ripleyk_radii or (32, 64, 128)
        self.additional_ripleyk_kwargs = additional_ripleyk_kwargs or {}
        assert max(self.ripleyk_radii) <= _roi_side / 2, (
            "Maximum Ripley's K radius must be less than half the ROI side."
        )
        self.topk_salient_rois = topk_salient_rois

        self._roi_side = _roi_side
        self._nucleus_superclasses = _nucleus_superclasses or [
            'EpithelialSuperclass',
            'StromalSuperclass',
            'TILsSuperclass',
        ]

        # pipeline of methods, each gets a different set of features
        self.features_using_nuclei_graph = {
            "nuclear_ripleyk_self_vs_self",
            "nuclear_ripleyk_self_vs_other",
            "orientation_entropies",
        }
        available_feature_sets = {
            "saliency",
            "region_areas",
            "nuclei_counts",
            "tils_score_variants",
            "fibroblast_activation",
            "tils_activation",
            "epithelial_atypia",
            "nuclear_size",
            "simple_nuclear_shape",
            "complex_nuclear_shape",
            "nuclear_staining",
            "nuclear_texture",
            "cytoplasmic_staining",
            "cytoplasmic_texture",
        }.union(self.features_using_nuclei_graph)
        self.feature_sets = feature_sets or available_feature_sets
        self.feature_sets = set(self.feature_sets)
        not_recognized = self.feature_sets.difference(available_feature_sets)
        assert len(not_recognized) < 1, (
            "These feature sets you requested are not recognized: "
            f"{not_recognized}"
        )
        self.get_nuclear_graph = len(
            self.feature_sets.intersection(self.features_using_nuclei_graph)
        ) > 0

        self._roiname = None
        self._sf = None  # scale factor from ROI MPP to base MPP

        # custom dtypes
        self.roi_dtype = namedtuple(
            'roi_data', ('roi_meta', 'nuclei_metas', 'nuclei_props')
        )

    def run(self):
        """"""
        self.maybe_extract_roi_feature_summaries()
        self.maybe_extract_global_nuclei_features()

    @collect_errors()
    def maybe_extract_global_nuclei_features(self):
        """
        Get summary of features from one ROI.
        """
        self._roiname = 'GLOBAL'
        monitor = f"{self.monitor}: {self._roiname} (buckle up!)"
        collect_errors.monitor = monitor
        where = opj(
            self.output_dir,
            'perDatasetSlideSummaries',
            'GlobalRoiBasedFeatures.csv'
        )
        if self._slidename_not_in_df_index(where):
            self.logger.info(monitor)
            feats = self._get_features_from_roi(
                *self._concat_data_from_all_rois()
            )
            feats.index = [self.slide_name]
            append_row_to_df_or_create_it(where, df=feats)

    @collect_errors()
    def maybe_extract_roi_feature_summaries(self):
        """"""
        # summarize features for each roi in the slide (one row per roi)
        metrics = self._load_or_extract_roi_feature_summaries()

        # summary of roi summaries .. whole slide is one row now
        if self._slidename_not_in_df_index(opj(
            self.output_dir,
            'perDatasetSlideSummaries',
            'RoiFeatureSummary_Means.csv'
        )):
            self._summarize_roi_feature_summaries(metrics)

    def get_features_from_one_roi(self, roiname):
        """
        Get summary of features from one ROI.
        """
        self._roiname = roiname.split('_')[1]
        self._sf = None
        rmonitor = f"{self.monitor}: {self._roiname}"
        self.logger.info(rmonitor)
        collect_errors.monitor = rmonitor

        return self._get_features_from_roi(*self._read_roi_data(roiname))

    # HELPERS -----------------------------------------------------------------

    def _concat_data_from_all_rois(self):
        """"""
        roinames = [
            os.path.splitext(roiname)[0]
            for roiname in os.listdir(opj(self.slide_dir, 'roiMeta'))
        ]
        roinames.sort()

        # base (scan) MPP resolution
        base_mpp = load_json(
            opj(self.slide_dir, f"{self.slide_name}.json")
        )['meta']['base_mpp']

        slide_data = [[], [], []]
        for roiname in roinames:
            roi_data = self._read_roi_data(roiname, base_mpp=base_mpp)
            for idx, field in enumerate(roi_data):
                slide_data[idx].append(field)

        slide_data = [concat(data) for data in slide_data]
        slide_data[0] = self._aggregate_roi_metas(slide_data[0])

        return self.roi_dtype(*slide_data)

    @staticmethod
    def _aggregate_roi_metas(roi_metas):
        """"""
        to_mean = [
            col for col in roi_metas.columns if col.startswith('metrics.')
        ]
        to_sum = [
            col for col in roi_metas.columns if any([
                col.startswith('region_summary.'),
                col.startswith('nuclei_summary.'),
            ])
        ]
        meta = concat([
            roi_metas.loc[:, to_mean].mean(),
            roi_metas.loc[:, to_sum].sum()
        ])
        return DataFrame(meta).T

    @collect_errors()
    def _summarize_roi_feature_summaries(self, metrics: DataFrame):
        """"""
        # restrict to topk salient rois
        metrics = metrics.sort_values(
            'Saliency.SaliencyScore', ascending=False
        )
        if self.topk_salient_rois is not None:
            metrics = metrics[:self.topk_salient_rois]
        # metrics are weighted by saliency (saliency itself is unweighted)
        saliency = metrics.loc[:, 'Saliency.SaliencyScore'].values
        means = {
            'Saliency.SaliencyScore': np.nanmean(saliency),
            'Saliency.TissueRatio': np.nanmean(
                metrics.loc[:, 'Saliency.TissueRatio']
            ),
        }
        stds = {
            'Saliency.SaliencyScore': np.nanstd(saliency),
            'Saliency.TissueRatio': np.nanstd(
                metrics.loc[:, 'Saliency.TissueRatio']
            ),
        }
        # keep relevant columns
        cols = [j for j in metrics.columns if not j.startswith('Saliency.')]
        metrics = metrics.loc[:, cols]
        max_saliency = saliency.max()
        if max_saliency > 0:
            saliency = saliency / max_saliency
        else:
            saliency = np.ones(saliency.size)
        for k in cols:
            col_vals = metrics.loc[:, k].values.copy()
            keep = np.isfinite(metrics.loc[:, k])
            if not keep.all():
                means[k], stds[k] = 0., 0.
            else:
                wmn, wst = weighted_avg_and_std(col_vals[keep], saliency[keep])
                means[k], stds[k] = float(wmn), float(wst)
        # now save
        append_row_to_df_or_create_it(
            opj(
                self.output_dir,
                'perDatasetSlideSummaries',
                'RoiFeatureSummary_Means.csv'
            ),
            DataFrame.from_records([means], index=[self.slide_name]),
        )
        append_row_to_df_or_create_it(
            opj(
                self.output_dir,
                'perDatasetSlideSummaries',
                'RoiFeatureSummary_Stds.csv'
            ),
            DataFrame.from_records([stds], index=[self.slide_name]),
        )

    def _slidename_not_in_df_index(self, where):
        """Check if slidename is not in the index of a dataframe."""
        if not os.path.isfile(where):
            return True

        existing = read_csv(where, usecols=[0]).iloc[:, 0].tolist()
        return self.slide_name not in existing

    def _load_or_extract_roi_feature_summaries(self):
        """
        Returns
        -------
        DataFrame
            Dataframe of features where rows are the roi names and columns are
            the column names. Each method ex tracts a set of features with a
            common "theme", as specified by the user
        """
        where = opj(
            self.output_dir, 'perSlideROISummaries', self.slide_name + '.csv'
        )

        # maybe this was already done
        if os.path.isfile(where):
            try:
                return read_csv(where, index_col=0)
            except Exception as whoknows:
                pass

        # summarize roi-level features
        roinames = os.listdir(opj(self.slide_dir, 'roiMeta'))
        roinames.sort()
        roi_summaries = [
            self.get_features_from_one_roi(os.path.splitext(roiname)[0])
            for roiname in roinames
        ]
        roi_summaries = concat(roi_summaries, axis=0)
        roi_summaries.index = roinames

        # save
        roi_summaries.to_csv(where)

        return roi_summaries

    @collect_errors()
    def _get_features_from_roi(self, roi_meta, nuclei_metas, nuclei_props):
        """"""
        nuclei_graphs = self._maybe_construct_nuclei_graph(nuclei_metas)
        features = []

        if "saliency" in self.feature_sets:
            features.append(self._get_saliency(roi_meta))

        if "region_areas" in self.feature_sets:
            features.append(self._get_region_areas(roi_meta))

        if "nuclei_counts" in self.feature_sets:
            features.append(self._get_nuclei_counts(roi_meta))

        if "tils_score_variants" in self.feature_sets:
            features.append(self._get_tils_score_variants(roi_meta))

        if "fibroblast_activation" in self.feature_sets:
            features.append(self._get_fibroblast_activation(nuclei_metas))

        if "tils_activation" in self.feature_sets:
            features.append(self._get_tils_activation(nuclei_metas))

        if "epithelial_atypia" in self.feature_sets:
            features.append(self._get_epithelial_atypia(nuclei_metas))

        if "nuclear_size" in self.feature_sets:
            features.append(
                self._get_nuclei_props_subset_for_superclasses(
                    nuclei_metas,
                    nuclei_props,
                    prefix="Size.",
                    rename_rule=lambda x: f"Nuclear{x}",
                )
            )

        if "simple_nuclear_shape" in self.feature_sets:
            features.append(
                self._get_nuclei_props_subset_for_superclasses(
                    nuclei_metas,
                    nuclei_props,
                    prefix="Shape.",
                    rename_rule=lambda x: f"SimpleNuclear{x}",
                    exclude_substrings=('HuMoments', 'FSD'),
                )
            )

        if "complex_nuclear_shape" in self.feature_sets:
            features.append(
                self._get_nuclei_props_subset_for_superclasses(
                    nuclei_metas,
                    nuclei_props,
                    prefix="Shape.",
                    rename_rule=lambda x: f"ComplexNuclear{x}",
                    include_substrings=('HuMoments', 'FSD'),
                )
            )

        if "nuclear_staining" in self.feature_sets:
            features.append(
                self._get_nuclei_props_subset_for_superclasses(
                    nuclei_metas,
                    nuclei_props,
                    prefix="Nucleus.Intensity.",
                    rename_rule=lambda x: x.replace(
                        "Nucleus.Intensity.", "NuclearStaining."
                    ),
                )
            )

        if "nuclear_texture" in self.feature_sets:
            features.append(
                self._get_nuclei_props_subset_for_superclasses(
                    nuclei_metas,
                    nuclei_props,
                    prefix="Nucleus.",
                    rename_rule=lambda x: x.replace(
                        "Nucleus.Gradient.", "NuclearTexture."
                    ).replace(
                        "Nucleus.Haralick.", "NuclearTexture."
                    ),
                    include_substrings=('Gradient.', 'Haralick.'),
                )
            )

        if "cytoplasmic_staining" in self.feature_sets:
            features.append(
                self._get_nuclei_props_subset_for_superclasses(
                    nuclei_metas,
                    nuclei_props,
                    prefix="Cytoplasm.Intensity.",
                    rename_rule=lambda x: x.replace(
                        "Cytoplasm.Intensity.", "CytoplasmicStaining."
                    ),
                )
            )

        if "cytoplasmic_texture" in self.feature_sets:
            features.append(
                self._get_nuclei_props_subset_for_superclasses(
                    nuclei_metas,
                    nuclei_props,
                    prefix="Cytoplasm.",
                    rename_rule=lambda x: x.replace(
                        "Cytoplasm.Gradient.", "CytoplasmicTexture."
                    ).replace(
                        "Cytoplasm.Haralick.", "CytoplasmicTexture."
                    ),
                    include_substrings=('Gradient.', 'Haralick.'),
                )
            )

        if "nuclear_ripleyk_self_vs_self" in self.feature_sets:
            features.append(
                self._get_nuclear_ripleyk_self_vs_self(nuclei_graphs)
            )

        if "nuclear_ripleyk_self_vs_other" in self.feature_sets:
            features.append(
                self._get_nuclear_ripleyk_self_vs_other(nuclei_graphs)
            )

        if "orientation_entropies" in self.feature_sets:
            features.extend(
                self._get_orientation_entropies(
                    nuclei_metas=nuclei_metas,
                    orientations=nuclei_props.loc[:, 'Orientation.Orientation'],
                    nuclei_graphs=nuclei_graphs,
                )
            )

        return concat(features, axis=1)

    def _get_orientation_entropies(
        self, nuclei_metas, orientations, nuclei_graphs
    ):
        """"""
        return [
            self._get_superclass_orientation(
                nuclei_metas=nuclei_metas,
                orientations=orientations,
                nuclei_graphs=nuclei_graphs,
                superclass=superclass,
            )
            for superclass in ['StromalSuperclass', 'EpithelialSuperclass']
        ]

    def _get_superclass_orientation(
        self, nuclei_metas, orientations, nuclei_graphs, superclass
    ):
        """"""
        radius = max(self.ripleyk_radii)  # standard radius name
        _radius = max(self._ripleyk_radii)  # scaled for this slide
        orients = orientations[
            nuclei_metas.loc[:, 'Classif.SuperClass'] == superclass
        ].values

        if orients.shape[0] == 0:
            return DataFrame.from_records([{
                f"Orientation.{superclass}.Std": 0.,
                f"Orientation.{superclass}.Entropy": 0.,
                f"Orientation.Radius-{radius}.{superclass}.MeanStd": 0.,
                f"Orientation.Radius-{radius}.{superclass}.MeanEntropy": 0.,
            }])

        orients = normalize_to_zero_one_range(orients)

        # orientation of cells in local neighborhood of each cell
        # NOTE: this is done at the max radius of self._ripleyk_radii
        ntree = nuclei_graphs[superclass]._tree
        local_orients = [
            orients[idxs]
            for idxs in ntree.query_ball_point(ntree.data, _radius)
        ]

        feats = {
            f"Orientation.{superclass}.Std": np.nanstd(orients),
            f"Orientation.{superclass}.Entropy": entropy(orients),
            f"Orientation.Radius-{radius}.{superclass}.MeanStd": (
                np.nanmean([np.nanstd(lo) for lo in local_orients])
            ),
            f"Orientation.Radius-{radius}.{superclass}.MeanEntropy": (
                np.nanmean([entropy(lo) for lo in local_orients])
            ),
        }

        return DataFrame.from_records([feats])

    def _get_nuclear_ripleyk_self_vs_other(self, nuclei_graphs):
        """"""
        feats = {}

        for sc_surround, ngraph_surround in nuclei_graphs.items():
            for sc_center, ngraph_center in nuclei_graphs.items():

                if sc_center == sc_surround:
                    continue

                raw_ks = ngraph_surround.calculate_unnormalized(
                    ngraph_center._tree.data
                )
                sc_str = f"Center-{sc_center}-Surround-{sc_surround}"
                feats.update({
                    f"RipleysK.Raw.{sc_str}.Radius-{r}": raw_ks[idx]
                    for idx, r in enumerate(self.ripleyk_radii)
                })

                # area & perimeter are expensive to calculate globally!
                if self._roiname == 'GLOBAL':
                    continue

                ks = ngraph_surround.maybe_normalize(
                    raw_ks,
                    normalize_to_csr=True,
                    center_to_zero=True,
                    unit_variance=True,
                )
                feats.update({
                    f"RipleysK.Normalized.{sc_str}.Radius-{r}": ks[idx]
                    for idx, r in enumerate(self.ripleyk_radii)
                })

        return DataFrame.from_records([feats])

    def _get_nuclear_ripleyk_self_vs_self(self, nuclei_graphs):
        """"""
        feats = {}

        for sc, ngraph in nuclei_graphs.items():

            raw_ks = ngraph.calculate_unnormalized()
            feats.update({
                f"RipleysK.Raw.{sc}.Radius-{r}": raw_ks[idx]
                for idx, r in enumerate(self.ripleyk_radii)
            })

            # area & perimeter are expensive to calculate globally!
            if self._roiname == 'GLOBAL':
                continue

            ks = ngraph.maybe_normalize(
                raw_ks,
                normalize_to_csr=True,
                center_to_zero=True,
                unit_variance=True,
            )
            feats.update({
                f"RipleysK.Normalized.{sc}.Radius-{r}": ks[idx]
                for idx, r in enumerate(self.ripleyk_radii)
            })

        return DataFrame.from_records([feats])

    def _get_nuclei_props_subset_for_superclasses(
        self,
        nuclei_metas,
        nuclei_props,
        prefix="",
        include_substrings=("", ),
        exclude_substrings=(),
        rename_rule: Callable = None,
    ):
        """"""
        rule = (lambda x: x) if rename_rule is None else rename_rule
        feats = []
        for sc in self._nucleus_superclasses:
            df = nuclei_props.loc[
                nuclei_metas.loc[:, 'Classif.SuperClass'] == sc, :
            ]
            colmap = {
                c: f"{rule(c)}.{sc}" for c in nuclei_props.columns
                if all([
                    c.startswith(prefix),
                    any(ss in c for ss in include_substrings),
                    not any(ss in c for ss in exclude_substrings),
                ])
            }
            df = self._keep_and_rename_columns_from_df(df, colmap=colmap)
            feats.append(self._get_df_mean_and_std(df))

        return concat(feats, axis=1)

    def _get_epithelial_atypia(self, nuclei_metas):
        """"""
        esc = 'EpithelialSuperclass'
        df = nuclei_metas.loc[
            nuclei_metas.loc[:, 'Classif.SuperClass'] == esc, :
        ]
        df = self._keep_and_rename_columns_from_df(
            df,
            colmap={
                col: f"EpithelialAtypia.{col.split('.')[1]}"
                for col in ['ClassifProbab.CancerEpithelium']
            },
        )

        return self._get_df_mean_and_std(df)

    def _get_tils_activation(self, nuclei_metas):
        """"""
        df = nuclei_metas.loc[
            nuclei_metas.loc[:, 'Classif.SuperClass'] == 'TILsSuperclass', :
        ]
        df = self._keep_and_rename_columns_from_df(
            df,
            colmap={
                col: f"Activation.TILs.{col.split('.')[1]}"
                for col in [
                    'ClassifProbab.ActiveTILsCell',
                    'ClassifProbab.CancerEpithelium',
                    'SuperClassifProbab.EpithelialSuperclass',
                ]
            },
        )
        ta = "Activation.TILs"
        df.loc[:, f"{ta}.CancerOrActiveTILs"] = (
            df.loc[:, f"{ta}.ActiveTILsCell"]
            + df.loc[:, f"{ta}.CancerEpithelium"]
        )
        df.loc[:, f"{ta}.EpithelialOrActiveTILs"] = (
            df.loc[:, f"{ta}.ActiveTILsCell"]
            + df.loc[:, f"{ta}.EpithelialSuperclass"]
        )

        return self._get_df_mean_and_std(df)

    def _get_fibroblast_activation(self, nuclei_metas):
        """"""
        df = nuclei_metas.loc[
            nuclei_metas.loc[:, 'Classif.SuperClass'] == 'StromalSuperclass', :
        ]
        df = self._keep_and_rename_columns_from_df(
            df,
            colmap={
                col: f"Activation.Stromal.{col.split('.')[1]}"
                for col in [
                    'ClassifProbab.ActiveStromalCellNOS',
                    'ClassifProbab.CancerEpithelium',
                    'SuperClassifProbab.EpithelialSuperclass',
                ]
            },
        )
        sa = "Activation.Stromal"
        df.loc[:, f"{sa}.CancerOrActiveStromal"] = (
            df.loc[:, f"{sa}.ActiveStromalCellNOS"]
            + df.loc[:, f"{sa}.CancerEpithelium"]
        )
        df.loc[:, f"{sa}.EpithelialOrActiveStromal"] = (
            df.loc[:, f"{sa}.ActiveStromalCellNOS"]
            + df.loc[:, f"{sa}.EpithelialSuperclass"]
        )

        return self._get_df_mean_and_std(df)

    def _get_tils_score_variants(self, roi_meta):
        return self._keep_and_rename_columns_from_df(
            roi_meta,
            colmap={
                col: col.replace('metrics', 'TILsScore')
                for col in roi_meta.columns
                if col.startswith('metrics') and 'TILs' in col
            },
        )

    def _get_nuclei_counts(self, roi_meta):
        df = self._keep_and_rename_columns_from_df(
            roi_meta,
            colmap={
                col: col.replace('nuclei_summary.nNuclei_', 'NoOfNuclei.')
                for col in roi_meta.columns
                if col.startswith('nuclei_summary') and 'BACKGROUND' not in col
            },
        )
        return df

    def _get_region_areas(self, roi_meta):
        """"""
        df = self._keep_and_rename_columns_from_df(
            roi_meta,
            colmap={
                col: col.replace(
                    'region_summary.pixelCount_', 'RegionPixelCount.'
                )
                for col in roi_meta.columns
                if col.startswith('region_summary') and not col.endswith('EXCLUDE')
            },
        )
        return df

    def _get_saliency(self, roi_meta):
        """"""
        return self._keep_and_rename_columns_from_df(
            roi_meta,
            colmap={
                'metrics.TissueRatio': 'Saliency.TissueRatio',
                'metrics.SaliencyScore': 'Saliency.SaliencyScore',
            },
        )

    @staticmethod
    def _get_df_mean_and_std(df):
        """"""
        means = DataFrame(df.mean(0, skipna=True)).T
        means.columns = [f"{c}.Mean" for c in means.columns]
        stds = DataFrame(df.std(0, skipna=True)).T
        stds.columns = [f"{c}.Std" for c in stds.columns]

        return concat([means, stds], axis=1)

    @staticmethod
    def _keep_and_rename_columns_from_df(df, colmap):
        """"""
        df = df.loc[:, list(colmap.keys())]
        df = df.rename(columns=colmap)

        return df

    def _maybe_construct_nuclei_graph(self, nuclei_metas):
        """
        If certain features are reuqested, construct a K-d tree that allows
        Ripley's K type of calculation like no of nuclei within a certain
        neighborhood radius.
        """
        if not self.get_nuclear_graph:
            return

        return {
            sc: self._get_kdtree_for_nucleus_superclass(nuclei_metas, sc)
            for sc in self._nucleus_superclasses
        }

    def _get_kdtree_for_nucleus_superclass(self, metas, superclass):
        """"""
        sliced = metas.loc[metas.loc[:, 'Classif.SuperClass'] == superclass, :]
        sliced = sliced.loc[:, [f"Identifier.Centroid{d}" for d in ('X', 'Y')]]

        additional_ripleyk_kwargs = deepcopy(self.additional_ripleyk_kwargs)
        if self._roiname == 'GLOBAL':
            additional_ripleyk_kwargs.update({'do_boundary_correction': False})

        tree = FastRipleyK(
            radii=self._ripleyk_radii,
            region_side_or_radius=self._roi_side,
            kdtree_kwargs=self.kdtree_kwargs,
            **additional_ripleyk_kwargs,
        )
        tree.fit(sliced.values)

        return tree

    def _visualize_kdtree(self, tree, xs, ys):
        """Just for debugging etc."""
        pairs = tree.query_pairs(self._ripleyk_radii[0])
        plt.figure(figsize=(5, 5))
        plt.scatter(xs, ys, c='gray')
        for (i, j) in pairs:
            plt.plot([xs[i], xs[j]], [ys[i], ys[j]], "-r")
        plt.xlim(0, 1024)
        plt.ylim(0, 1024)
        plt.show()

    @collect_errors()
    def _read_roi_data(self, roiname, base_mpp=None):
        """"""
        roi_meta = flatten_dict(
            load_json(opj(self.slide_dir, 'roiMeta', f"{roiname}.json"))
        )
        nuclei_metas = read_csv(
            opj(self.slide_dir, 'nucleiMeta', f"{roiname}.csv"),
            index_col='Identifier.ObjectCode',
        )
        keep_cols = [
            c for c in nuclei_metas.columns
            if not any([c.startswith('Unconstrained.'), '.' not in c])
        ]
        nuclei_metas = nuclei_metas.loc[:, keep_cols]

        # maybe adjust coords & radii to be relative to wsi base level
        if base_mpp is not None:
            nuclei_metas = self._adjust_nuclei_metas_coords(
                nuclei_metas, roi_meta=roi_meta, base_mpp=base_mpp,
            )

        nuclei_props = read_csv(
            opj(self.slide_dir, 'nucleiProps', f"{roiname}.csv"),
            index_col='Identifier.ObjectCode',
        )
        keep_cols = [
            c for c in nuclei_props.columns
            if not any([c.startswith('Identifier.'), '.' not in c])
        ]
        nuclei_props = nuclei_props.loc[:, keep_cols]
        nuclei_props.index = np.int32(nuclei_props.index)

        return self.roi_dtype(
            roi_meta=DataFrame.from_records([roi_meta]),
            nuclei_metas=nuclei_metas,
            nuclei_props=nuclei_props,
        )

    def _adjust_nuclei_metas_coords(self, nuclei_metas, roi_meta, base_mpp):
        """"""
        self._sf = roi_meta['mpp'] / base_mpp
        xcols = ['Identifier.Xmin', 'Identifier.Xmax', 'Identifier.CentroidX']
        ycols = ['Identifier.Ymin', 'Identifier.Ymax', 'Identifier.CentroidY']
        nuclei_metas.loc[:, xcols + ycols] *= self._sf
        nuclei_metas.loc[:, xcols] += roi_meta['wsi_left']
        nuclei_metas.loc[:, ycols] += roi_meta['wsi_top']

        return nuclei_metas

    @property
    def _ripleyk_radii(self) -> Tuple[int]:
        if self._sf is None:
            return self.ripleyk_radii
        return tuple([r * self._sf for r in self.ripleyk_radii])

    @cached_property
    def _roi_area(self) -> float:
        """Area of the region of interest, assuming square ROI."""
        return self._roi_side ** 2


class DatasetFeatureExtractor(object):
    """
    Extracts features from MuTIsWSIRunner ouput.
    """
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        *,
        wsi_dir: str = None,
        region_extractor_kwargs: dict = None,
        nuclei_extractor_kwargs: dict = None,
        extras_extractor_kwargs: dict = None,
        monitor: str = "",
        logger: Any = None,
        _debug: bool = False,
        _reverse: bool = False,
    ):
        """"""
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.wsi_dir = wsi_dir
        self.region_extractor_kwargs = region_extractor_kwargs or {}
        self.nuclei_extractor_kwargs = nuclei_extractor_kwargs or {}
        self.collagen_extractor_kwargs = extras_extractor_kwargs or {}
        self.slide_names = os.listdir(input_dir)
        self.monitor = monitor
        self.logger = logger or logging.getLogger(__name__)
        self._debug = _debug
        self._reverse = _reverse
        self._slmonitor = self.monitor

        self.slide_names.sort(reverse=self._reverse)

        collect_errors.logger = self.logger
        collect_errors._debug = self._debug

        self._create_directory_structure()

    def run(self):
        """"""
        self.extract_region_features()
        self.extract_nuclear_features(emode='roi')
        self.extract_nuclear_features(emode='global')
        self.extract_collagen_features()

    def extract_collagen_features(self):
        """"""
        for slidx, slide_name in enumerate(self.slide_names):
            collect_errors.reset()
            self._slmonitor = (
                f"{self.monitor}: "
                f"Slide {slidx + 1} of {len(self.slide_names)} "
                f"({slide_name} - CollagenFeats)"
            )
            self._extract_collagen_features_for_slide(slide_name)

    def extract_region_features(self):
        """"""
        for slidx, slide_name in enumerate(self.slide_names):
            collect_errors.reset()
            self._slmonitor = (
                f"{self.monitor}: "
                f"Slide {slidx + 1} of {len(self.slide_names)} "
                f"({slide_name} - regions)"
            )
            self._extract_region_features_for_slide(slide_name)

    def extract_nuclear_features(self, emode: str):
        """"""
        assert emode in ['roi', 'global']
        for slidx, slide_name in enumerate(self.slide_names):
            collect_errors.reset()
            self._slmonitor = (
                f"{self.monitor}: Slide {slidx + 1} of {len(self.slide_names)}"
                f" ({slide_name} - {emode})"
            )
            self._extract_nuclear_features_for_slide(slide_name, emode=emode)

    # HELPERS -----------------------------------------------------------------

    @collect_errors()
    def _extract_collagen_features_for_slide(self, slide_name):
        """
        Run extra feature extraction (eg collagen entropy) for a single slide.
        """
        self.logger.info(self._slmonitor)
        slide_ext = 'svs'  # if slide_name.startswith('TCGA') else 'ndpi'
        wsi_file = opj(self.wsi_dir, f"{slide_name}.{slide_ext}")
        if not os.path.isfile(wsi_file):
            self.logger.error(f"{self._slmonitor}: {wsi_file} nor found!")
            return
        extractor = SlideCollagenFeatureExtractor(
            slide_dir=opj(self.input_dir, slide_name),
            output_dir=self.output_dir,
            wsi_file=wsi_file,
            monitor=self._slmonitor,
            logger=self.logger,
            _debug=self._debug,
            **self.collagen_extractor_kwargs
        )
        extractor.run()

    @collect_errors()
    def _extract_region_features_for_slide(self, slide_name):
        """
        Run nuclear feature extraction for a single slide.
        """
        self.logger.info(self._slmonitor)
        extractor = SlideRegionFeatureExtractor(
            slide_dir=opj(self.input_dir, slide_name),
            output_dir=self.output_dir,
            monitor=self._slmonitor,
            logger=self.logger,
            **self.region_extractor_kwargs
        )
        extractor.run()

    @collect_errors()
    def _extract_nuclear_features_for_slide(self, slide_name, emode='all'):
        """
        Run nuclear feature extraction for a single slide.
        """
        self.logger.info(self._slmonitor)
        extractor = SlideNucleiFeatureExtractor(
            slide_dir=opj(self.input_dir, slide_name),
            output_dir=self.output_dir,
            monitor=self._slmonitor,
            logger=self.logger,
            **self.nuclei_extractor_kwargs
        )
        if emode == 'all':
            extractor.run()
        elif emode == 'roi':
            extractor.maybe_extract_roi_feature_summaries()
        elif emode == 'global':
            extractor.maybe_extract_global_nuclei_features()
        else:
            self.logger.error(
                f"{self.monitor}: Unknown feature extraction mode: {emode}"
            )

    def _create_directory_structure(self):
        """"""
        for subdir in (
            'perSlideROISummaries',
            'perSlideRegionFeatures',
            'perSlideCollagenFeatures',
            'perDatasetSlideSummaries',
        ):
            os.makedirs(opj(self.output_dir, subdir), exist_ok=True)


# =============================================================================

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Extract WSI features.')
    parser.add_argument(
        '-c',
        '--cohort',
        type=str,
        help='CPSII_40X, CPS3_40X, TCGA_BRCA, or plco_breast',
        required=True,
        default='plco_breast',
    )
    parser.add_argument('-r', '--reverse', type=int, default=0)
    parser.add_argument('-d', '--debug', type=int, default=0)
    ARGS = parser.parse_args()
    ARGS.debug = bool(ARGS.debug)
    ARGS.reverse = bool(ARGS.reverse)

    WSI_DIR = opj('/input', ARGS.cohort)
    INPUT_DIR = opj('/output', ARGS.cohort, 'perSlideResults')
    OUTPUT_DIR = opj('/output', ARGS.cohort, 'cTMEfeats')

    # configure logger
    from MuTILs_Panoptic.utils.MiscRegionUtils import get_configured_logger
    LOGDIR = opj(OUTPUT_DIR, 'LOGS')
    os.makedirs(LOGDIR, exist_ok=True)

    ssme = DatasetFeatureExtractor(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        wsi_dir=WSI_DIR,
        monitor=(
            f"{'(DEBUG) ' if ARGS.debug else ''}"
            f"{ARGS.cohort} "
            f"{'(reverse)' if ARGS.reverse else ''}"
        ),
        logger=get_configured_logger(
            logdir=LOGDIR, prefix='SlideFeatureExtractor', tofile=True
        ),
        _debug=ARGS.debug,
        _reverse=ARGS.reverse,
    )
    ssme.run()
