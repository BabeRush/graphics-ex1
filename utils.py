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
        # self.idx_map_h, self.idx_map_v = np.meshgrid(range(self.w), range(self.h))
        self.idx_map_h, self.idx_map_v = np.meshgrid(range(self.w), range(self.h))
        self.c_v = None
        self.c_l = None
        self.c_r = None
        self.m_v = None
        self.m_l = None
        self.m_r = None

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
        # Convert to grayscale using matrix multiplication
        gs_img = np.dot(np_img, self.gs_weights)
        gs_img = gs_img.reshape(gs_img.shape[0], gs_img.shape[1])
        gs_img[0] = 0.5
        gs_img[-1] = 0.5
        gs_img[:, -1] = 0.5
        gs_img[:, 0] = 0.5
        return gs_img

    def calc_gradient_magnitude(self):
        """ Calculate gradient magnitude of a grayscale image

        Returns:
            A gradient magnitude image (float32) of shape (h, w)

        Guidelines & hints:
            In order to calculate a gradient of a pixel, only its neighborhood is required.
        """
        gs_img = self.resized_gs.copy()
        e_vertical = np.abs(gs_img - np.roll(gs_img, -1, axis=1))
        e_vertical[:, -1] = np.abs(gs_img[:, -1] - gs_img[:, -2])  # handle the last column separately
        e_horizontal = np.abs(gs_img - np.roll(gs_img, -1, axis=0))
        e_horizontal[-1, :] = np.abs(gs_img[-1, :] - gs_img[-2, :])  # handle the last row separately
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
        gs_img = self.resized_gs.copy()
        cost_matrix = np.zeros_like(gs_img)
        energy_matrix = self.E.copy()
        c_v = np.zeros_like(gs_img)
        c_v[0] = np.abs(np.roll(gs_img[0], -1, axis=0) - np.roll(gs_img[0], 1, axis=0))
        cost_matrix[0] = gs_img[0] + c_v[0]
        for i in range(1, cost_matrix.shape[0]):
            c_v[i] = np.abs(np.roll(gs_img[i], -1, axis=0) - np.roll(gs_img[i], 1, axis=0))
            cost_matrix[i] = cost_matrix[i - 1] + c_v[i] + energy_matrix[i]
        return cost_matrix

    def seams_removal(self, num_remove: int, column=True):
        """ Iterates num_remove times and removes num_remove vertical seams

        Parameters:
            num_remove (int): number of vertical seam to be removed

        This step can be divided into a couple of steps:
            i) init/update matrices (E, M, backtracking matrix, seam mask) where:
                - E is the gradient magnitude matrix
                - M is the cost matric
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
                - mask is a boolean matrix for removed seams
            ii) seam backtracking: calculates the actual indices of the seam
            iii) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            iv) seam removal: create the carved image with the reduced (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you with, but it needs to support:
            - removing seams couple of times (call the function more than once)
            - visualize the original image with removed seams marked (for comparison)
        """
        for i in range(num_remove):
            self.backtrack_seam()
            self.remove_seam()
            self.update_E(self.seam_history[-1])
            self.update_M(self.seam_history[-1])
            self.seam_balance += 1

    def update_E(self, seam_idx):
        self.E = np.delete(self.E, seam_idx, axis=1)
        s_minus_one = seam_idx - 1
        if s_minus_one >= 0:
            self.E[:, s_minus_one] = self.calculate_energy_local(s_minus_one)
        else:
            self.E[:, self.E.shape[1] - 1] = self.calculate_energy_local(self.E.shape[1] - 1)
        self.E[:, seam_idx] = self.calculate_energy_local(seam_idx)

    def calculate_energy_local(self, seam_idx):
        dest_horiz = self.E[:, seam_idx].copy()
        for i in range(len(dest_horiz) - 1):
            dest_horiz[i] = np.abs(self.resized_gs[i] - self.resized_gs[i + 1])
        dest_horiz[len(dest_horiz) - 1] = np.abs(
            self.resized_gs[len(dest_horiz) - 1] - self.resized_gs[len(dest_horiz) - 2])
        if seam_idx < self.E.shape[1] - 1:
            dest_vert = np.abs(self.E[:, seam_idx] - self.E[:, seam_idx + 1])
        else:
            dest_vert = np.abs(self.E[:, seam_idx] - self.E[:, seam_idx - 2])
        dest_horiz[len(dest_horiz) - 1] = np.abs(
            self.resized_gs[len(dest_horiz) - 1] - self.resized_gs[len(dest_horiz) - 2])
        pixel_energy = np.sqrt((dest_vert ** 2) + (dest_horiz ** 2))
        return pixel_energy

    def update_M(self, seam_idx):
        self.M = np.delete(self.M, seam_idx, axis=1)
        s_minus_one = seam_idx - 1
        if s_minus_one == -1:
            s_minus_one = self.M.shape[1] - 1
        for j in [s_minus_one, seam_idx]:
            cv = np.abs(np.roll(self.resized_gs[:, j], -1, axis=0) - np.roll(self.resized_gs[:, j], 1, axis=0))
            temp_col = self.M[:, j]
            temp_col[0] = self.resized_gs[0]
            for i in range(1, len(temp_col)):
                temp_col[i] = temp_col[i - 1] + cv[i] + self.E[i]
            self.M[:, j] = temp_col

    def seams_removal_horizontal(self, num_remove):
        """ Removes num_remove horizontal seams

        Parameters:
            num_remove (int): number of horizontal seam to be removed

        Guidelines & hints:
            You may find np.rot90 function useful

        """

        # Preprocessing:
        self.resized_rgb = np.transpose(self.resized_rgb, (1, 0, 2))
        self.resized_gs = self.resized_gs.T
        self.E = self.E.T
        self.M = self.M.T
        self.M = self.cumm_mask.T
        self.tracking_matrix = self.tracking_matrix.T
        self.seams_rgb = np.transpose(self.seams_rgb, (1, 0, 2))
        # Call seams_removal
        temp = self.seams_removal(num_remove, column=False)
        # Postprocessing:
        self.resized_rgb = np.transpose(self.resized_rgb, (0, 1, 2))
        self.resized_gs = self.resized_gs.T
        self.E = self.E.T
        self.M = self.M.T
        self.M = self.cumm_mask.T
        self.tracking_matrix = self.tracking_matrix.T
        self.seams_rgb = np.transpose(self.seams_rgb, (0, 1, 2))

    def seams_removal_vertical(self, num_remove):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): number of vertical seam to be removed
        """
        # Preprocessing:
        # Call seams_removal
        self.seams_removal(num_remove)

    def backtrack_seam(self):
        """ Backtracks a seam for Column Seam Carving method
        """
        self.seam_history.append(np.argmin(self.M[-1]))

    def remove_seam(self):
        """ Removes a seam for self.rgb (you may create a resized version, like self.resized_rgb)

        Guidelines & hints:
        In order to apply the removal, you might want to extend the seam mask to support 3 channels (rgb) using: 3d_mak = np.stack([1d_mask] * 3, axis=2), and then use it to create a resized version.
        """
        # deleting the column in both matrices
        seam_index = self.seam_history[-1]
        self.resized_gs = np.delete(self.resized_gs, seam_index, axis=1)
        self.resized_rgb = np.delete(self.resized_rgb, seam_index, axis=1)
        original_pic_seam_index = self.idx_map_h[0, seam_index]

        # setting the cumm_mask indices false where we removed pixels
        self.cumm_mask[:, original_pic_seam_index] = False

        # cumm_3d_mask = np.stack([self.cumm_mask, self.cumm_mask, self.cumm_mask], axis=2)
        # # coloring the removed seams red
        # red_3d_mask = np.invert(cumm_3d_mask) * (1, 0, 0)
        self.seams_rgb[~self.cumm_mask] = [1, 0, 0]
        self.idx_map_h = np.delete(self.idx_map_h, seam_index, axis=1)
        self.idx_map_v = np.delete(self.idx_map_v, seam_index, axis=1)


class VerticalSeamImage(SeamImage):
    def __init__(self, *args, **kwargs):
        """ VerticalSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        try:
            self.M = self.calc_M()
        except NotImplementedError as e:
            print(e)

    def calc_M(self):
        """ Calculates the matrix M discussed in lecture (with forward-looking cost)

        Returns:
            An energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            You might find the function 'np.roll' useful.
        """
        gs_img = self.resized_gs.copy()
        cost_matrix = np.zeros_like(gs_img)
        cost_matrix[0] = gs_img[0]
        energy_matrix = self.E.copy()
        self.c_v = np.zeros_like(gs_img)
        self.c_l = np.zeros_like(gs_img)
        self.c_r = np.zeros_like(gs_img)
        self.m_v = np.zeros_like(gs_img)
        self.m_l = np.zeros_like(gs_img)
        self.m_r = np.zeros_like(gs_img)
        for i in range(1, cost_matrix.shape[0]):
            c_v = np.abs(np.roll(gs_img[i], -1, axis=0) - np.roll(gs_img[i], 1, axis=0))
            self.c_v[i] = c_v
            c_r = c_v + np.abs(gs_img[i - 1] - np.roll(gs_img[i], -1, axis=0))
            self.c_r[i] = c_r
            c_l = c_v + np.abs(gs_img[i - 1] - np.roll(gs_img[i], 1, axis=0))
            self.c_l[i] = c_l
            m_r = np.roll(cost_matrix[i - 1], -1, axis=0)
            self.m_r[i] = m_r
            m_l = np.roll(cost_matrix[i - 1], 1, axis=0)
            self.m_l[i] = m_l
            m_v = cost_matrix[i - 1]
            self.m_v[i] = m_v
            cost_matrix[i] = np.min([m_r + c_r, m_l + c_l, m_v + c_v], axis=0) + energy_matrix[i]
        return cost_matrix

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

    def seams_removal_vertical(self, num_remove, vertical=True):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): umber of vertical seam to be removed
        """
        for k in range(num_remove):
            path = self.backtrack_seam()
            # Creates new gs
            temp_gs = self.resized_gs
            updated_gs = np.empty((0, self.resized_gs.shape[1] - 1))
            for i in range(self.resized_gs.shape[0]):
                row = np.delete(temp_gs[i], path[i])
                updated_gs = np.vstack((updated_gs, row))
            self.resized_gs = updated_gs
            # Creates new E
            self.E = self.calc_gradient_magnitude()
            # Creates new M
            self.M = self.calc_M()
            # Removes from rgb
            updated_rgb = np.empty((0, self.resized_rgb.shape[1] - 1, 3))
            for i in range(self.resized_rgb.shape[0]):
                row = np.delete(self.resized_rgb[i], path[i], axis=0).reshape(1, self.resized_rgb.shape[1] - 1, 3)
                updated_rgb = np.vstack((updated_rgb, row))
            self.resized_rgb = updated_rgb

    def backtrack_seam(self):
        """ Backtracks a seam for Seam Carving as taught in lecture
        """
        path = []
        min_arg = np.argmin(self.M[-1])
        path.append(min_arg)
        h = self.M.shape[0]
        for i in range(h - 2, -1, -1):
            min_arg_minus_one = min_arg - 1
            min_arg_plus_one = min_arg + 1
            if min_arg_minus_one == -1:
                min_arg_minus_one = self.M.shape[1] - 1
            elif min_arg_plus_one == self.M.shape[1]:
                min_arg_plus_one = 0
            if self.M[i, min_arg_minus_one] + self.c_l[i + 1, min_arg] + self.E[i + 1, min_arg] == \
                    self.M[i + 1, min_arg]:
                min_arg = min_arg_minus_one
            elif self.M[i, min_arg_plus_one] + self.c_r[i + 1, min_arg] + self.E[i + 1, min_arg] == self.M[
                i + 1, min_arg]:
                min_arg = min_arg_plus_one
            elif self.M[i, min_arg] + self.c_v[i + 1, min_arg] + self.E[i + 1, min_arg] == self.M[
                i + 1, min_arg]:
                min_arg = min_arg
            path.append(min_arg)
        path.reverse()

        for i in range(self.cumm_mask.shape[0]):
            self.cumm_mask[i, path[i]] = False
        self.seams_rgb[~self.cumm_mask] = [1, 0, 0]

        return path

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
