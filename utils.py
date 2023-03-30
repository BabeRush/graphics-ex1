import numpy as np
from PIL import Image
from numba import jit
from tqdm import tqdm
from abc import abstractmethod, abstractstaticmethod


class SeamImage:
    def __init__(self, img_path, vis_seams=True):
        """ SeamImage initialization.

        Parameters:
            img_path (str): image local path
            method (str) (a or b): a for Hard Vertical and b for the known Seam Carving algorithm
            vis_seams (bool): if true, another version of the original image shall be store, and removed seams should be marked on it
        """
        #################
        # Do not change #
        #################
        self.path = img_path

        self.gs_weights = np.array([[0.299, 0.587, 0.114]]).T

        self.rgb = self.load_image(img_path)
        self.resized_rgb = self.rgb.copy()

        self.vis_seams = vis_seams
        if vis_seams:
            self.seams_rgb = self.rgb.copy()

        self.h, self.w = self.rgb.shape[:2]

        try:
            self.gs = self.rgb_to_grayscale(self.rgb)
            self.resized_gs = self.gs.copy()
            self.cumm_mask = np.ones_like(self.gs, dtype=bool)
        except NotImplementedError as e:
            print(e)

        try:
            self.E = self.calc_gradient_magnitude()
        except NotImplementedError as e:
            print(e)
        #################

        # additional attributes you might find useful
        self.seam_history = []
        self.seam_balance = 0

        # This might serve you to keep tracking original pixel indices
        self.idx_map_h, self.idx_map_v = np.meshgrid(range(self.w), range(self.h))
        self.tracking_matrix = np.zeros((self.h, self.w, 2))
        for i in range(self.tracking_matrix.shape[0]):
            for j in range(self.tracking_matrix.shape[1]):
                self.tracking_matrix[i, j] = [i, j]

    def rgb_to_grayscale(self, np_img):
        """ Converts a np RGB image into grayscale (using self.gs_weights).
        Parameters
            np_img : ndarray (float32) of shape (h, w, 3)
        Returns:
            grayscale image (float32) of shape (h, w, 1)

        Guidelines & hints:
            Use NumpyPy vectorized matrix multiplication for high performance.
            To prevent outlier values in the boundaries, we recommend to pad them with 0.5
        """
        padded_img = np.pad(np_img, [(1, 1), (1, 1), (0, 0)], mode='constant', constant_values=0.5)
        gray_scaled_img = np.dot(padded_img, self.gs_weights)
        h, w, c = gray_scaled_img.shape
        return gray_scaled_img.reshape(h, w)[2:, 2:]

    def calc_gradient_magnitude(self):
        """ Calculate gradient magnitude of a grayscale image

        Returns:
            A gradient magnitude image (float32) of shape (h, w)

        Guidelines & hints:
            In order to calculate a gradient of a pixel, only its neighborhood is required.
        """
        gs_img = self.resized_gs.copy()
        e_vertical = np.zeros_like(gs_img)
        e_vertical[:, :-1] = np.abs(np.diff(gs_img, axis=1))
        e_vertical[:, -1] = np.abs(gs_img[:, -1] - gs_img[:, -2])
        e_horizontal = np.zeros_like(gs_img)
        e_horizontal[:-1, :] = np.abs(np.diff(gs_img, axis=0))
        e_horizontal[-1, :] = np.abs(gs_img[-1, :] - gs_img[-2, :])

        return np.sqrt(e_vertical ** 2 + e_horizontal ** 2)

    def calc_M(self):
        pass

    def seams_removal(self, num_remove):
        pass

    def seams_removal_horizontal(self, num_remove):
        pass

    def seams_removal_vertical(self, num_remove):
        pass

    def rotate_mats(self, clockwise):
        pass

    def init_mats(self):
        pass

    def update_ref_mat(self):
        pass

    def backtrack_seam(self):
        pass

    def remove_seam(self):
        pass

    def reinit(self):
        """ re-initiates instance
        """
        self.__init__(self.path)

    @staticmethod
    def load_image(img_path):
        return np.asarray(Image.open(img_path)).astype('float32') / 255.0


class ColumnSeamImage(SeamImage):
    """ Column SeamImage.
    This class stores and implements all required data and algorithmics from implementing the "column" version of the seam carving algorithm.
    """

    def __init__(self, *args, **kwargs):
        """ ColumnSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        try:
            self.M = self.calc_M()
        except NotImplementedError as e:
            print(e)

    def calc_M(self, column=True):
        """ Calculates the matrix M discussed in lecture, but with the additional constraint:
            - A seam must be a column. That is, the set of seams S is simply columns of M.
            - implement forward-looking cost

        Returns:
            A "column" energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            The formula of calculation M is as taught, but with certain terms omitted.
            You might find the function 'np.roll' useful.
        """
        if column:
            gs_img = self.resized_gs.copy()
            cost_matrix = self.E.copy()
        else:
            gs_img = self.resized_gs.copy().T
            cost_matrix = self.E.copy().T
        c_v = gs_img.copy()
        for i in range(1, c_v.shape[1] - 1):
            c_v[:, i] = np.abs(gs_img[:, i + 1] - gs_img[:, i - 1])
        # Storing values in the edges, according a cyclic logic
        c_v[:, 0] = np.abs(gs_img[:, 1] - gs_img[:, c_v.shape[1] - 1])
        c_v[:, c_v.shape[1] - 1] = np.abs(gs_img[:, 0] - gs_img[:, c_v.shape[1] - 2])
        cost_matrix += c_v
        for i in range(1, cost_matrix.shape[0]):
            cost_matrix[i] += cost_matrix[i - 1]
        if column:
            return cost_matrix
        else:
            return cost_matrix.T

    # def seams_removal(self, num_remove: int):
    #     """ Iterates num_remove times and removes num_remove vertical seams
    #
    #     Parameters:
    #         num_remove (int): number of vertical seam to be removed
    #
    #     This step can be divided into a couple of steps:
    #         i) init/update matrices (E, M, backtracking matrix, seam mask) where:
    #             - E is the gradient magnitude matrix
    #             - M is the cost matric
    #             - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
    #             - mask is a boolean matrix for removed seams
    #         ii) seam backtracking: calculates the actual indices of the seam
    #         iii) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
    #         iv) seam removal: create the carved image with the reduced (and update seam visualization if desired)
    #         Note: the flow described below is a recommendation. You may implement seams_removal as you with, but it needs to support:
    #         - removing seams couple of times (call the function more than once)
    #         - visualize the original image with removed seams marked (for comparison)
    #     """

    # def update_E(self):
    #     self.E = self.calc_gradient_magnitude()
    #
    # def update_M(self):
    #     self.M = self.calc_M()

    # def calc_row_M(self):
    #     # Calculate row cost matrix
    #     gs_img = self.resized_gs.copy()
    #     cost_row_matrix = self.E.copy()
    #     c_h = gs_img.copy()
    #     for i in range(1, c_h.shape[0] - 1):
    #         c_h[i] = np.abs(gs_img[i + 1] - gs_img[i - 1])
    #     # Storing values in the edges, according a cyclic logic
    #     c_h[0] = np.abs(gs_img[1] - gs_img[c_h.shape[0] - 1])
    #     c_h[c_h.shape[0] - 1] = np.abs(gs_img[0] - gs_img[c_h.shape[0] - 2])
    #     cost_row_matrix += c_h
    #     for i in range(1, cost_row_matrix.shape[1]):
    #         cost_row_matrix[:, i] += cost_row_matrix[:, i - 1]
    #     return cost_row_matrix

    def seams_removal_horizontal(self, num_remove):
        """ Removes num_remove horizontal seams

        Parameters:
            num_remove (int): number of horizontal seam to be removed

        Guidelines & hints:
            You may find np.rot90 function useful

        """

        # Remove horizontal seams
        cost_row_matrix = self.calc_M(column=False)
        temp_M = cost_row_matrix.copy()
        for i in range(num_remove):
            min_val = np.min(cost_row_matrix[:, -1])
            seam_idx = np.argwhere(cost_row_matrix[:, -1] == min_val)
            original_idx = np.argwhere(temp_M[:, -1] == min_val)
            self.resized_gs = np.delete(self.resized_gs, seam_idx, axis=0)
            self.E = self.calc_gradient_magnitude()
            cost_row_matrix = self.calc_M(column=False)
            self.cumm_mask[original_idx] = False
            self.resized_rgb = np.delete(self.resized_rgb, seam_idx, axis=0)
        self.seams_rgb[~self.cumm_mask] = [255, 0, 0]

    def seams_removal_vertical(self, num_remove):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): number of vertical seam to be removed
        """
        temp_M = self.M
        for i in range(num_remove):
            min_val = np.min(self.M[-1])
            seam_idx = np.argwhere(self.M[-1] == min_val)
            original_idx = np.argwhere(temp_M[-1] == min_val)
            self.resized_gs = np.delete(self.resized_gs, seam_idx, axis=1)
            self.E = self.calc_gradient_magnitude()
            self.M = self.calc_M()
            self.cumm_mask[:, original_idx] = False
            self.resized_rgb = np.delete(self.resized_rgb, seam_idx, axis=1)
        self.seams_rgb[~self.cumm_mask] = [255, 0, 0]

    def backtrack_seam(self):
        """ Backtracks a seam for Column Seam Carving method
        """
        raise NotImplementedError("TODO: Implement SeamImage.backtrack_seam")

    def remove_seam(self):
        """ Removes a seam for self.rgb (you may create a resized version, like self.resized_rgb)

        Guidelines & hints:
        In order to apply the removal, you might want to extend the seam mask to support 3 channels (rgb) using: 3d_mak = np.stack([1d_mask] * 3, axis=2), and then use it to create a resized version.
        """
        raise NotImplementedError("TODO: Implement SeamImage.remove_seam")


class VerticalSeamImage(SeamImage):
    def __init__(self, *args, **kwargs):
        """ VerticalSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        try:
            self.M = self.calc_M()
        except NotImplementedError as e:
            print(e)

    def calc_M(self, column=True):
        """ Calculates the matrix M discussed in lecture (with forward-looking cost)

        Returns:
            An energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            You might find the function 'np.roll' useful.
        """
        if column:
            gs_img = self.resized_gs.copy()
            cost_matrix = np.zeros_like(gs_img)
            cost_matrix[0] = gs_img[0]
            energy_matrix = self.E.copy()
        else:
            gs_img = self.resized_gs.copy().T
            cost_matrix = np.zeros_like(gs_img).T
            cost_matrix[0] = gs_img[0]
            energy_matrix = self.E.copy().T
        for i in range(1, gs_img.shape[0]):
            for j in range(gs_img.shape[1]):
                c_v = np.abs(np.roll(gs_img[i], -1)[j] - np.roll(gs_img[i], 1)[j])
                c_r = c_v + np.abs(np.roll(gs_img[i - 1], -1)[j] - np.roll(gs_img[i], -1)[j])
                c_l = c_v + np.abs(np.roll(gs_img[i - 1], 1)[j] - np.roll(gs_img[i], 1)[j])
                m_r = np.roll(cost_matrix[i - 1], -1)[j]
                m_l = np.roll(cost_matrix[i - 1], 1)[j]
                m_v = cost_matrix[i - 1, j]
                cost_matrix[i][j] = np.min([m_r + c_r, m_l + c_l, m_v + c_v])
        if column:
            return energy_matrix + cost_matrix
        else:
            return energy_matrix.T + cost_matrix.T

    def seams_removal(self, num_remove: int):
        """ Iterates num_remove times and removes num_remove vertical seams

        Parameters:
            num_remove (int): number of vertical seam to be removed

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (E, M, backtracking matrix, saem mask) where:
                - E is the gradient magnitude matrix
                - M is the cost matrix
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
                - mask is a boolean matrix for removed seams
            ii) fill in the backtrack matrix corresponding to M
            iii) seam backtracking: calculates the actual indices of the seam
            iv) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            v) seam removal: create the carved image with the reduced (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you with, but it needs to supprt:
            - removing seams couple of times (call the function more than once)
            - visualize the original image with removed seams marked (for comparison)
        """
        raise NotImplementedError("TODO: Implement SeamImage.seams_removal")

    def seams_removal_horizontal(self, num_remove):
        """ Removes num_remove horizontal seams

        Parameters:
            num_remove (int): number of horizontal seam to be removed

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        raise NotImplementedError("TODO: Implement SeamImage.seams_removal_horizontal")

    def seams_removal_vertical(self, num_remove):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): umber of vertical seam to be removed
        """

        raise NotImplementedError("TODO: Implement SeamImage.seams_removal_vertical")

    def backtrack_seam(self):
        """ Backtracks a seam for Seam Carving as taught in lecture
        """
        raise NotImplementedError("TODO: Implement SeamImage.backtrack_seam_b")

    def remove_seam(self):
        """ Removes a seam from self.rgb (you may create a resized version, like self.resized_rgb)

        Guidelines & hints:
        In order to apply the removal, you might want to extend the seam mask to support 3 channels (rgb) using: 3d_mak = np.stack([1d_mask] * 3, axis=2), and then use it to create a resized version.
        """
        raise NotImplementedError("TODO: Implement SeamImage.remove_seam")

    def seams_addition(self, num_add: int):
        """ BONUS: adds num_add seamn to the image

            Parameters:
                num_add (int): number of horizontal seam to be removed

            Guidelines & hints:
            - This method should be similar to removal
            - You may use the wrapper functions below (to support both vertical and horizontal addition of seams)
            - Visualization: paint the added seams in green (0,255,0)

        """
        raise NotImplementedError("TODO: Implement SeamImage.seams_addition")

    def seams_addition_horizontal(self, num_add):
        """ A wrapper for removing num_add horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): number of horizontal seam to be added

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_horizontal")

    def seams_addition_vertical(self, num_add):
        """ A wrapper for removing num_add vertical seams (just a recommendation)

        Parameters:
            num_add (int): number of vertical seam to be added
        """

        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_vertical")

    @staticmethod
    # @jit(nopython=True)
    def calc_bt_mat(M, backtrack_mat):
        """ Fills the BT back-tracking index matrix. This function is static in order to support Numba. To use it, uncomment the decorator above.

        Recommnded parameters (member of the class, to be filled):
            M: np.ndarray (float32) of shape (h,w)
            backtrack_mat: np.ndarray (int32) of shape (h,w): to be filled here

        Guidelines & hints:
            np.ndarray is a rederence type. changing it here may affected outsde.
        """
        raise NotImplementedError("TODO: Implement SeamImage.calc_bt_mat")


def scale_to_shape(orig_shape: np.ndarray, scale_factors: list):
    """ Converts scale into shape

    Parameters:
        orig_shape (np.ndarray): original shape [y,x]
        scale_factors (list): scale factors for y,x respectively

    Returns
        the new shape
    """
    raise NotImplementedError("TODO: Implement SeamImage.scale_to_shape")


def resize_seam_carving(seam_img: SeamImage, shapes: np.ndarray):
    """ Resizes an image using Seam Carving algorithm

    Parameters:
        seam_img (SeamImage) The SeamImage instance to resize
        shapes (np.ndarray): desired shape (y,x)

    Returns
        the resized rgb image
    """
    raise NotImplementedError("TODO: Implement SeamImage.resize_seam_carving")


def bilinear(image, new_shape):
    """
    Resizes an image to new shape using bilinear interpolation method
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :returns: the image resized to new_shape
    """
    in_height, in_width, _ = image.shape
    out_height, out_width = new_shape
    new_image = np.zeros(new_shape)

    ###Your code here###
    def get_scaled_param(org, size_in, size_out):
        scaled_org = (org * size_in) / size_out
        scaled_org = min(scaled_org, size_in - 1)
        return scaled_org

    scaled_x_grid = [get_scaled_param(x, in_width, out_width) for x in range(out_width)]
    scaled_y_grid = [get_scaled_param(y, in_height, out_height) for y in range(out_height)]
    x1s = np.array(scaled_x_grid, dtype=int)
    y1s = np.array(scaled_y_grid, dtype=int)
    x2s = np.array(scaled_x_grid, dtype=int) + 1
    x2s[x2s > in_width - 1] = in_width - 1
    y2s = np.array(scaled_y_grid, dtype=int) + 1
    y2s[y2s > in_height - 1] = in_height - 1
    dx = np.reshape(scaled_x_grid - x1s, (out_width, 1))
    dy = np.reshape(scaled_y_grid - y1s, (out_height, 1))
    c1 = np.reshape(image[y1s][:, x1s] * dx + (1 - dx) * image[y1s][:, x2s], (out_width, out_height, 3))
    c2 = np.reshape(image[y2s][:, x1s] * dx + (1 - dx) * image[y2s][:, x2s], (out_width, out_height, 3))
    new_image = np.reshape(c1 * dy + (1 - dy) * c2, (out_height, out_width, 3)).astype(int)
    return new_image
