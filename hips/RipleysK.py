from scipy.spatial import cKDTree
from typing import Iterable
from functools import cached_property
import numpy as np


class FastRipleyK(object):
    """"""

    def __init__(
            self,
            radii: Iterable[int],
            *,
            region_side_or_radius: int = None,
            region_shape="square",
            kdtree_kwargs: dict = None,
            do_boundary_correction=True,
            fast_correction_by_iou=True,
            normalize_to_csr=True,
            center_to_zero=True,
            unit_variance=True,
            _return_nans_as_zeros=False,
    ):
        """
        Fast implementation of Ripley's K using KD-trees.

        Parameters
        ----------
        radii: Iterable[int]
            radii at which to calculate K value
        kdtree_kwargs: dict
            kwargs, passed on as-is to KDTree.

        """
        self.radii = np.array(radii)
        self.region_side_or_radius = region_side_or_radius
        self.region_shape = region_shape
        self.kdtree_kwargs = kdtree_kwargs or {
            'leafsize': 16,
            'compact_nodes': True,
            'copy_data': False,
            'balanced_tree': True,
            'boxsize': None,
        }
        self.do_boundary_correction = do_boundary_correction
        self.correction_by_bounding_box = fast_correction_by_iou
        self.normalize_to_csr = normalize_to_csr
        self.center_to_zero = center_to_zero
        self.unit_variance = unit_variance
        self._return_nans_as_zeros = _return_nans_as_zeros

        # sanity checks
        if do_boundary_correction or normalize_to_csr:
            assert region_side_or_radius is not None, (
                "Some requested functionality requires region_side_or_radius!"
            )

        self._tree = None

    # noinspection PyArgumentList,PyUnresolvedReferences
    def fit(self, data):
        """Fit KD tree.

        Parameters
        ----------
        data : array_like, shape (x, y)
            The n data points of dimension m to be indexed. This array is
            not copied unless except in some situations. See documentation
            for KDTree.

        """
        self._tree = cKDTree(data, **self.kdtree_kwargs)

    def calculate(
            self, other_data: np.ndarray = None
    ) -> Iterable[float]:
        """Calculate Ripley's K function.

        Parameters
        ----------
        other_data: np.ndarray, optional
            x, y coordinates of points act as our reference (CENTER). For eg,
            if self is fitted to all lymphocytes in a region of interest (ROI),
            other could be the tumor cells within the same ROI. Ripleys K
            would then be a normalized version of the average number of
            lymphocytes surrounding tumor cells.

        Notes
        -----
        [1] Center vs Surround ...
            other_data is CENTER, self._tree.data is SURROUND.
        [2] Symmetry ...
            This is NOT a symmetric value; the number of lymphocytes
            surrounding tumor is NOT necessarily the same as the number of
            tumor surrounding lymphocytes. If this argument is not provided,
            Ripley's K is calculated with respect to self.

        Returns
        -------
        Iterable[float]
            Calculated Ripley's K values at self.radii. This is an array of
            the same length as self.radii.

        """
        return self.maybe_normalize(self.calculate_unnormalized(other_data))

    def calculate_unnormalized(
            self, other_data: np.ndarray = None
    ) -> Iterable[float]:
        """Calculate unnormalized Ripley's K function.

        Notes
        -----
        [1] Center vs Surround ...
            other_data is CENTER, self._tree.data is SURROUND.
        [2] What do we mean by unnormalized?
            Since this is unnormalized, this can be interpreted as the average
            number of neighbors per event/cell.

            See docstring for self.calculate. This version does NOT normalize
            for CSR, center around zero, or normalize to unit variance.

        """
        assert self._tree is not None, "You must fit the KD tree first!"

        if other_data is None:
            data = self._tree.data
            # search space excluded central event when looking for neighbors
            n_events = self.n_events - 1
        else:
            data = other_data
            # all of self._tree.data included in search space
            n_events = self.n_events

        # no center
        if data.shape[0] == 0:
            k_values = np.zeros((len(self.radii),), dtype=np.float32)
            if not self._return_nans_as_zeros:
                k_values[:] = np.nan
            return k_values

        # no surround
        if self._tree.data.shape[0] == 0:
            return np.zeros((len(self.radii),), dtype=np.float32)

        k_values = [
            np.sum(self._get_n_neighbors(data, radius=r)) / n_events
            for r in self.radii
        ]

        return np.array(k_values)

    def maybe_normalize(
            self,
            k_values: np.ndarray,
            normalize_to_csr: bool = None,
            center_to_zero: bool = None,
            unit_variance: bool = None,
    ) -> np.ndarray:
        """
        Maybe normalize K values to Complete Spatial Randomness (CSR),
        center to zero, and/or make them have a unit variance.

        If we normalize to CSR, this becomes a statistical metric that
        normalizes to the density of events. For example, if we are using
        this to estimate clustering of cells, there's a higher chance that
        more lymphocytes will surround another lymphocyte by random chance,
        just because there's so many of them. On the other hand, just a few
        fibroblasts surrounding another may be enough since there's so few
        of them so the chance of this happening by chance is lower.

        Parameters
        ----------
        k_values: np.ndarray
            K values from self.calculate_unnormalized
        normalize_to_csr: bool, optional
            if None, use self attribute
        center_to_zero: bool, optional
            if None, use self attribute
        unit_variance: bool, optional
            if None, use self attribute

        Returns
        -------
        np.ndarray
            Normalized Ripley's K values at self.radii. This is an array of
            the same length as self.radii.

        """
        if np.all(np.isnan(k_values)) or np.sum(k_values) == 0:
            return k_values

        # Note: this is done by design to allow the user to reuse unnormalized
        # values when calculating normalized ones if they want both
        if normalize_to_csr is None:
            normalize_to_csr = self.normalize_to_csr
        if center_to_zero is None:
            center_to_zero = self.center_to_zero
        if unit_variance is None:
            unit_variance = self.unit_variance

        normalized_ks = k_values.copy()

        # order of operations is important!

        if normalize_to_csr:
            normalized_ks /= self._gamma

        if center_to_zero:
            normalized_ks -= self._search_areas

        if unit_variance:
            normalized_ks /= self._k_value_stdevs

        return normalized_ks

    @cached_property
    def n_events(self) -> int:
        return self._tree.data.shape[0]

    @cached_property
    def region_area(self) -> float:
        """Area of the entire region of interest from which events exist."""
        if self.region_shape == 'square':
            return self.region_side_or_radius ** 2

        raise NotImplementedError(
            f"{self.region_shape} regions of interest not implemented!"
        )

    # HELPERS -----------------------------------------------------------------

    def _get_n_neighbors(self, data, radius):
        """"""
        n_neighbors = np.array([
            len(j) for j in self._tree.query_ball_point(data, radius)
        ])
        corrections = self._get_besag_correction_terms(data, radius=radius)

        return n_neighbors * corrections

    def _get_besag_correction_terms(
            self, data: np.ndarray, radius: int
    ) -> np.ndarray:
        """
        Get boundary correction terms for data.
        """
        if not self.do_boundary_correction:
            return np.ones((data.shape[0],))

        if not self.correction_by_bounding_box:
            raise NotImplementedError(
                "My implementation assumes that the ROI is square while "
                "search space around each point is circular. Calculating the "
                "overlap in this case is not trivial without looping though "
                "each point and doing some arithmetic, which is slower."
            )

        region_bounds = np.array(
            [0, 0, self.region_side_or_radius, self.region_side_or_radius]
        )[None, ...]
        areas = self.iou_components(
            self._centroids_to_bounds(data, radius),
            region_bounds,
        )

        return areas['bboxes1_areas'] / areas['intersection']

    # noinspection PyPep8Naming
    @staticmethod
    def iou_components(bboxes1, bboxes2):
        """Fast, vectorized IOU components.

        Modified from:
            https://medium.com/@venuktan/vectorized-intersection-over-union ...
            -iou-in-numpy-and-tensor-flow-4fa16231b63d

        Parameters
        -----------
        bboxes1 : np array
            columns encode bounding box corners xmin, ymin, xmax, ymax
        bboxes2 : np array
            same as bboxes 1

        Returns
        --------
        dict

        """
        x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))

        interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
        boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

        return {
            'intersection': interArea[:, 0],
            'bboxes1_areas': boxAArea[:, 0],
            'bboxes2_areas': boxBArea[:, 0],
        }

    @staticmethod
    def _centroids_to_bounds(data: np.ndarray, r: int) -> np.ndarray:
        """Convert from x, y centroids to xmin, ymin, xmax, ymax."""
        cx, cy = np.hsplit(data, 2)
        return np.concatenate([(cx - r), (cy - r), (cx + r), (cy + r)], axis=1)

    @cached_property
    def _k_value_stdevs(self) -> np.ndarray:
        """
        Standard deviation of Ripley's K at CSR for various radii at CSR.

        References:
        -----------
            Lagache T, Lang G, Sauvonnet N, Olivo-Marin J-C. Analysis of the
            Spatial Organization of Molecules with Robust Statistics.
            Rappoport JZ, editor. PLoS One. 2013;8: e80914. pmid:24349021

        """
        beta = self._search_areas / self.region_area
        gamma = (self._region_perimeter * self.radii) / self.region_area

        part1 = (2 * (self.region_area ** 2) * beta) / (self.n_events ** 2)
        part2 = (
                1 + 0.305 * gamma + beta * (
                    -1 + 0.0132 * self.n_events * gamma)
        )
        variance = part1 * part2

        return np.sqrt(variance)

    @cached_property
    def _search_areas(self) -> np.ndarray:
        """
        Circular search areas at various radii.
        This is also the expected value of K at CSR.
        """
        return np.pi * (self.radii ** 2)

    @cached_property
    def _region_perimeter(self) -> float:
        """Perimeter of the region of interest from which events exist."""
        if self.region_shape == 'square':
            return self.region_side_or_radius * 4

        raise NotImplementedError(
            f"{self.region_shape} regions of interest not implemented!"
        )

    @cached_property
    def _gamma(self) -> float:
        """Event density."""
        return self.n_events / self.region_area
