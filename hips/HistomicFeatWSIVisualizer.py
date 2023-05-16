import os
from os.path import join as opj
import warnings
from typing import Tuple, List
from pandas import read_csv, Series
import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle
# from histomicstk.preprocessing.color_normalization import \
#     deconvolution_based_normalization
from sklearn.preprocessing import MinMaxScaler

from MuTILs_Panoptic.histolab.src.histolab.types import CoordinatePair
from MuTILs_Panoptic.histolab.src.histolab.slide import Slide
from MuTILs_Panoptic.histolab.src.histolab.util import np_to_pil


class HistomicFeatWSIVisualizer(object):
    """
    Visualize WSI heatmap of histomic features.
    """
    def __init__(
        self,
        perslide_feats_dir: str,
        wsi_dir: str,
        featname_list: List[Tuple[str,str]],
        *,
        savedir: str = None,
        topk: int = 64,
        tile_size: Tuple = (512, 512),
        slide_names: List[str] = None,  # names of slides, no file extension
        color_normalize: bool = False,
        wsi_ext: str = 'svs',
        normalize_features: bool = True,
        _debug=False,
    ):
        if _debug:
            warnings.warn("Running in DEBUG mode!!!")
            raise NotImplementedError("Didn't implement debug mode yet.")

        self._debug = _debug
        self.perslide_feats_dir = perslide_feats_dir
        self.wsi_dir = wsi_dir
        self.featname_list = featname_list
        self.topk = topk
        self.tile_size = tile_size
        self.slide_names = slide_names if slide_names is not None else [
            dc.replace('.csv', '') for dc in os.listdir(perslide_feats_dir)
            if dc.endswith('.csv')
        ]
        self.savedir = savedir if savedir is not None else perslide_feats_dir
        os.makedirs(self.savedir, exist_ok=True)
        self.color_normalize = color_normalize
        self.wsi_ext = wsi_ext
        self.normalize_features = normalize_features

        # variables to carry over. This is not the best way to do this but
        # I'm in a rush now so whatever.
        self._slide = None
        self._slidename = None
        self._featname = None
        self._short_featname = None
        self._thumb = None
        self._sf = None

    @staticmethod
    def _get_coords_from_tilename(name: str):
        """"""
        return CoordinatePair(*[
            int(name.split(f"_{loc}-")[-1].split('_')[0].replace('.json', ''))
            for loc in ['left', 'top', 'right', 'bottom']
        ])

    def _save_tile(self, tidx, tilename, feat_df):
        """"""
        if np.isnan(feat_df[tilename]):
            return

        where = opj(
            self.savedir, self._slidename, f"{self._short_featname}_tiles"
        )
        os.makedirs(where, exist_ok=True)

        coords = self._get_coords_from_tilename(tilename)
        tile = self._slide.extract_tile(
            coords, tile_size=self.tile_size, mpp=0.5
        )
        rgb = tile.image

        # color normalize for comparability
        if self.color_normalize:
            rgb = deconvolution_based_normalization(  # noqa
                np.array(rgb), mask_out=~tile._tissue_mask
            )
            rgb = np_to_pil(rgb)

        # now plot and save
        description = f"rank={tidx}_{self._short_featname}={feat_df[tilename]:.3E}"
        print(f"{self._slidename}: {description}")
        fig, ax = plt.subplots(1, 2, figsize=(2 * 7, 7))
        ax[0].imshow(rgb)
        ax[0].set_title(description)
        ax[1].imshow(self._thumb)
        xmin, ymin, xmax, ymax = [int(j * self._sf) for j in coords]
        ax[1].add_patch(Rectangle(
            xy=(xmin + ((xmax - xmin) // 2), ymin + ((ymax - ymin) // 2)),
            width=xmax - xmin,
            height=ymax - ymin,
            linewidth=2,
            color='yellow',
            fill=False,
        ))
        plt.tight_layout()
        plt.savefig(opj(
            where, f"rank={tidx}__{tilename.replace('.json', '.png')}",
        ))
        plt.close()

    def visualize_top_and_bottom_tiles(self, top_salient_feats_df):
        """"""
        feat_df = top_salient_feats_df.loc[:, self._featname].sort_values(
            ascending=False
        )
        feat_df = feat_df.dropna()
        k = min(feat_df.shape[0], self.topk) // 2
        for tidx in (list(range(k)) + list(range(-1, -k-1, -1))):
            self._save_tile(
                tidx=tidx, tilename=feat_df.index[tidx], feat_df=feat_df
            )

    def save_heatmap_for_feat(self, all_feats_df):
        """"""
        savedir = opj(self.savedir, self._slidename)
        os.makedirs(savedir, exist_ok=True)

        # init heatmap masks
        feat_heatmap = np.zeros(
            (self._thumb.size[1], self._thumb.size[0]), dtype=np.float32
        )
        saliency_heatmap = feat_heatmap.copy()

        # heatmap for feature
        finite_feat = all_feats_df.loc[:, self._featname]
        finite_feat = finite_feat[np.isfinite(finite_feat.values)]
        normalized_feat = (
            MinMaxScaler().fit_transform(finite_feat.values.reshape(-1, 1))
            if self.normalize_features else finite_feat.values.reshape(-1, 1)
        )
        normalized_feat = Series(normalized_feat[:, 0], index=finite_feat.index)
        for tilename, feat_value in normalized_feat.items():
            coords = self._get_coords_from_tilename(tilename)
            xmin, ymin, xmax, ymax = [int(j * self._sf) for j in coords]
            feat_heatmap[ymin:ymax, xmin:xmax] = feat_value

        # heatmap for saliency
        top_salient_feats_df = all_feats_df.iloc[:self.topk, :]
        normalized_saliency = MinMaxScaler().fit_transform(
            top_salient_feats_df.loc[:, "Saliency.SaliencyScore"].values.reshape(-1, 1)
        )[:, 0]
        normalized_saliency = Series(
            normalized_saliency, index=top_salient_feats_df.index,
        )
        for tilename, saliency_value in normalized_saliency.items():
            coords = self._get_coords_from_tilename(tilename)
            xmin, ymin, xmax, ymax = [int(j * self._sf) for j in coords]
            saliency_heatmap[ymin:ymax, xmin:xmax] = saliency_value

        fig, ax = plt.subplots(
            1,
            4,
            figsize=(5 + 3 * 12, 10),
            gridspec_kw={'width_ratios': [25, 25, 1, 25]},
            # sharey='row',
        )

        # rgb for comparison
        ax[0].imshow(self._thumb)

        # heatmap for feature
        ax[1].imshow(self._thumb, alpha=0.4)
        im = ax[1].imshow(
            np.ma.masked_array(feat_heatmap, feat_heatmap == 0),
            cmap='plasma',
            alpha=0.95,
            vmin=np.min(feat_heatmap),
            vmax=np.max(feat_heatmap),
        )
        ax[1].set_title(self._short_featname)

        fig.colorbar(im, cax=ax[2], orientation='vertical')

        # top k salient tiles included in weighted averaging
        ax[3].imshow(self._thumb, alpha=0.4)
        ax[3].imshow(
            np.ma.masked_array(saliency_heatmap, saliency_heatmap == 0),
            cmap='plasma',
            alpha=0.7,
        )
        ax[3].set_title(f"Tile saliency (Top {self.topk})")

        plt.tight_layout()
        plt.savefig(opj(
            savedir, f"{self._short_featname}_HEATMAP_{self._slidename}.png",
        ))
        plt.close()


    def run(self):
        """"""
        for self._slidename in self.slide_names:

            all_feats_df = read_csv(
                opj(self.perslide_feats_dir, f"{self._slidename}.csv"),
                index_col=0
            )
            self._slide = Slide(
                opj(self.wsi_dir, f"{self._slidename}.{self.wsi_ext}"),
                opj(self.wsi_dir, "out", f"{self._slidename}.{self.wsi_ext}"),
                use_largeimage=True,
            )
            # scale factor from base to thumbnail
            self._thumb = self._slide.thumbnail
            self._sf = self._thumb.size[0] / self._slide.dimensions[0]
            # topk salient rois used for feature analysis
            all_feats_df.sort_values(
                "Saliency.SaliencyScore", axis=0, ascending=False, inplace=True
            )
            # visualize features one by one
            for _, (self._featname, self._short_featname) in enumerate(self.featname_list):

                self.save_heatmap_for_feat(all_feats_df=all_feats_df)
                self.visualize_top_and_bottom_tiles(
                    top_salient_feats_df=all_feats_df.iloc[:self.topk, :]
                )


# =============================================================================

if __name__ == "__main__":

    import argparse

    HOME = os.path.expanduser('~')

    parser = argparse.ArgumentParser(
        description='Visualize histomic feature heatmaps using ROI-level data'
    )
    parser.add_argument(
        '--perslidedir', type=str,
        default=opj(
            HOME, 'Desktop', 'STROMAL_IMPACT_ANALYSIS', 'plco_breast',
            'perSlideROISummaries',
        ),
    )
    parser.add_argument(
        '--wsidir', type=str,
        default=opj(
            HOME, 'Desktop', 'STROMAL_IMPACT_ANALYSIS', 'plco_breast', 'wsi',
        ),
    )
    parser.add_argument(
        '--savedir', type=str,
        default=opj(
            HOME, 'Desktop', 'STROMAL_IMPACT_ANALYSIS', 'plco_breast', 'HFVis',
        ),
    )
    parser.add_argument('--wsiext', type=str, default='svs')
    ARGS = parser.parse_args()

    vizer = HistomicFeatWSIVisualizer(
        perslide_feats_dir=ARGS.perslidedir,
        wsi_dir=ARGS.wsidir,
        savedir=ARGS.savedir,
        featname_list=[
            ("NuclearTexture.Canny.Sum.EpithelialSuperclass.Mean", "ChromatinClumpingOfEpithNuclei"),
            ("NuclearTexture.Canny.Sum.TILsSuperclass.Mean", "ChromatinClumpingOfTILsNuclei"),
            ("SimpleNuclearShape.FractalDimension.StromalSuperclass.Mean", "ComplexityOfCAFNuclBoundary"),
            ("RipleysK.Normalized.Center-EpithelialSuperclass-Surround-StromalSuperclass.Radius-128", "CAFClusteringAroundEpithCell64uM"),
            ("CytoplasmicStaining.Mean.StromalSuperclass.Mean", "PeriCAFMatrixHeteroIn512uMROI"),
        ],
        wsi_ext=ARGS.wsiext,
    )
    vizer.run()
