
# Helper functions for image warping

# Code based on https://github.com/seasonSH/WarpGAN

import torch
import numpy as np
import time

import utils_misc


################################################################################
# _image_warp(): Main functionfor image warping
################################################################################
def _image_warp(img, dst_pts, flow_pts, sharp=False, img2=None):
    """Warps the giving image based on displacement and destination points

    Args:
        img: [b, c, h, w] float `Tensor`
        dst_pts: [n, 2] float `Tensor`
        flow_pts: [n, 2] float `Tensor`

    Returns:
        interpolated: [b, c, h, w] float `Tensor`
    """

    start_time = time.time()

    w, v = _solve_interpolation(train_points=dst_pts, train_values=flow_pts)

    solve_time = time.time()

    _, _, height, width = img.size()
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_locations = np.stack([grid_y, grid_x], axis=2) # (H, W, 2)
    flattened_grid_locations = grid_locations.reshape([height*width, 2]) # (H*W, 2)
    flattened_grid_locations = utils_misc.to_device(torch.from_numpy(flattened_grid_locations).float())

    flattened_flow = _apply_interpolation(flattened_grid_locations, dst_pts, w, v)
    flow = flattened_flow.reshape([height, width, 2])

    interp_time = time.time()


    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    stacked_grid = np.stack([grid_y, grid_x], axis=2) # (H, W, 2)
    stacked_grid = utils_misc.to_device(torch.from_numpy(stacked_grid).float())

    query_points_on_grid = stacked_grid - flow # (H, W, 2)
    query_points_flattened = query_points_on_grid.reshape([height*width, 2]) # (H*W, 2)

    img = img.reshape((3, height, width)).permute((1, 2, 0))
    interpolated = _interpolate_bilinear(img, query_points_flattened, indexing='ij', sharp=sharp) # (H*W, C)
    interpolated = interpolated.reshape((height, width, 3)).permute((2, 0, 1))
    interpolated = interpolated.reshape((1, 3, height, width))
    warp_field = flow.clone().reshape((height, width, 2)).permute((2, 0, 1))
    warp_field = warp_field.reshape((1, 2, height, width))

    if img2 is not None:
      img2 = img2.reshape((3, height, width)).permute((1, 2, 0))
      interpolated2 = _interpolate_bilinear(img2, query_points_flattened, indexing='ij', sharp=sharp) # (H*W, C)
      interpolated2 = interpolated2.reshape((height, width, 3)).permute((2, 0, 1))
      interpolated2 = interpolated2.reshape((1, 3, height, width))

    finish_time = time.time()

    #print('solve fraction', (solve_time-start_time)/(time.time()-start_time), (interp_time-solve_time)/(time.time()-start_time), (finish_time-interp_time)/(time.time()-start_time))
    if img2 is None:
      return interpolated, warp_field
    else:
      return interpolated, interpolated2, warp_field


################################################################################
# Helper functions needed for _image_warp()
################################################################################

def _phi(r): return 0.5 * r * torch.log(torch.clamp(r, min=1e-10))


def _cross_squared_distance_matrix(x, y):
    """Pairwise squared distance between two matrices' rows.

    Computes the pairwise distances between rows of x and rows of y
    Args:
      x: [n, d] float `Tensor`
      y: [m, d] float `Tensor`

    Returns:
      squared_dists: [n, m] float `Tensor`, where
      squared_dists[i,j] = ||x[i,:] - y[j,:]||^2
    """
    x_norm_squared = torch.sum(torch.mul(x, x), 1)
    y_norm_squared = torch.sum(torch.mul(y, y), 1)

    # Expand so that we can broadcast.
    x_norm_squared_tile = x_norm_squared.unsqueeze(1)
    y_norm_squared_tile = y_norm_squared.unsqueeze(0)

    x_y_transpose = torch.matmul(utils_misc.to_device(x), torch.transpose(y, 0, 1))

    # squared_dists[i,j] = ||x_i - y_j||^2 = x_i'x_i- 2x_i'x_j + x_j'x_j
    squared_dists = utils_misc.to_device(x_norm_squared_tile) - 2 * x_y_transpose + y_norm_squared_tile

    return squared_dists


def _pairwise_squared_distance_matrix(x):
    """Pairwise squared distance among a matrix's rows.

    This saves a bit of computation vs. using _cross_squared_distance_matrix(x,x)

    Args:
        x: `[n, d]` float `Tensor`

    Returns:
        squared_dists: `[n, n]` float `Tensor`, where
        squared_dists[i,j] = ||x[i,:] - x[j,:]||^2
    """

    x_x_transpose = torch.matmul(x, torch.transpose(x, 0, 1)) # (n, n)
    x_norm_squared = torch.diagonal(x_x_transpose).reshape((-1, 1)) # (n, 1)

    # squared_dists[i,j] = ||x_i - x_j||^2 = x_i'x_i- 2x_i'x_j + x_j'x_j
    squared_dists = x_norm_squared - 2 * x_x_transpose + torch.transpose(x_norm_squared, 0, 1)

    return squared_dists


def _solve_interpolation(train_points, train_values):
    """Solve for interpolation coefficients.

    Computes the coefficients of the polyharmonic interpolant for the 'training'
    data defined by (train_points, train_values) using the kernel phi.

    Args:
        train_points: `[n, d]` interpolation centers
        train_values: `[n, k]` function values

    Returns:
        w: `[n, k]` weights on each interpolation center
        v: `[d, k]` weights on each input dimension
    """

    n, d = train_points.size()
    _, k = train_values.size()

    # First, rename variables so that the notation (c, f, w, v, A, B, etc.)
    # follows https://en.wikipedia.org/wiki/Polyharmonic_spline.
    c = train_points
    f = train_values

    # Next, construct the linear system.
    matrix_a = _phi(_pairwise_squared_distance_matrix(c))  # [n, n]
    #matrix_a += torch.randn_like(matrix_a)*1e-3

    # Append ones to the feature values for the bias term in the linear model.
    ones = utils_misc.to_device(torch.ones([n, 1]))
    matrix_b = torch.cat([c, ones], 1)  # [n, d+1]
    #matrix_b += torch.randn_like(matrix_b)*1e-3

    left_block = torch.cat([matrix_a, torch.transpose(matrix_b, 0, 1)], 0) # [n+d+1, n]
    num_b_cols = matrix_b.size(1) # d+1

    lhs_zeros = utils_misc.to_device(torch.zeros([num_b_cols, num_b_cols]))

    right_block = torch.cat([matrix_b, lhs_zeros], 0)  # [n+d+1, d+1]
    lhs = torch.cat([left_block, right_block], 1)  # [n+d+1, n+d+1]

    rhs_zeros = utils_misc.to_device(torch.zeros([d+1, k]))
    rhs = torch.cat([f, rhs_zeros], 0)  # [n+d+1, k]

    # Then, solve the linear system and unpack the results.
    w_v, _ = torch.solve(rhs, lhs)
    w = w_v[:n]
    v = w_v[n:]

    return w, v


def _apply_interpolation(query_points, train_points, w, v):
    """Apply polyharmonic interpolation model to data.

    Given coefficients w and v for the interpolation model, we evaluate
    interpolated function values at query_points.

    Args:
      query_points: `[m, d]` x values to evaluate the interpolation at
      train_points: `[n, d]` x values that act as the interpolation centers
                      ( the c variables in the wikipedia article)
      w: `[n, k]` weights on each interpolation center
      v: `[d, k]` weights on each input dimension

    Returns:
      Polyharmonic interpolation evaluated at points defined in query_points.
    """

    num_query_points = query_points.size(0)

    # First, compute the contribution from the rbf term.
    pairwise_dists = _cross_squared_distance_matrix(query_points, train_points)
    phi_pairwise_dists = _phi(pairwise_dists)

    rbf_term = torch.matmul(phi_pairwise_dists, w)

    # Then, compute the contribution from the linear term.
    # Pad query_points with ones, for the bias term in the linear model.
    query_points_pad = torch.cat([query_points, utils_misc.to_device(torch.ones([num_query_points, 1]))], 1)
    linear_term = torch.matmul(query_points_pad, v)

    return rbf_term + linear_term

def _interpolate_bilinear(grid, query_points, indexing='ij', sharp=False):
    """Similar to Matlab's interp2 function.
    Finds values for query points on a grid using bilinear interpolation.

    Args:
      grid: a 3-D float `Tensor` of shape `[height, width, channels]`.
      query_points: a 2-D float `Tensor` of N points with shape `[N, 2]`.
      name: a name for the operation (optional).
      indexing: whether the query points are specified as row and column (ij),
        or Cartesian coordinates (xy).

    Returns:
      values: a 2-D `Tensor` with shape `[N, channels]`

    Raises:
      ValueError: if the indexing mode is invalid, or if the shape of the inputs
        invalid.
    """

    height, width, channels = grid.size()
    query_type = query_points.dtype
    grid_type = grid.dtype
    num_queries, _ = query_points.shape

    alphas = []
    floors = []
    ceils = []

    index_order = [0, 1] if indexing == 'ij' else [1, 0]
    unstacked_query_points = torch.unbind(query_points, 1)

    for dim in index_order:
        queries = unstacked_query_points[dim]
        size_in_indexing_dimension = grid.size(dim)

        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # is still a valid index into the grid.
        max_floor = size_in_indexing_dimension - 2 # query_type
        min_floor = 0.0 # query_type
        floor = torch.clamp(torch.clamp(torch.floor(queries), min=min_floor), max=max_floor)
        int_floor = floor.int()
        floors.append(int_floor)
        ceil = int_floor + 1
        ceils.append(ceil)

        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        alpha = queries - floor # grid_type
        min_alpha = 0.0 # grid_type
        max_alpha = 1.0 # grid_type
        alpha = torch.clamp(torch.clamp(alpha, min=min_alpha), max=max_alpha)

        # Expand alpha to [b, n, 1] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = alpha.unsqueeze(1)
        alphas.append(alpha)

    flattened_grid = grid.reshape([height*width, channels])

    # This wraps array_ops.gather. We reshape the image data such that the
    # batch, y, and x coordinates are pulled into the first dimension.
    # Then we gather. Finally, we reshape the output back. It's possible this
    # code would be made simpler by using array_ops.gather_nd.
    def gather(y_coords, x_coords, name):
        linear_coordinates = y_coords*width + x_coords # (H*W)
        linear_coordinates = linear_coordinates.long()#torch.LongTensor(linear_coordinates.long())
        gathered_values= flattened_grid[linear_coordinates]
        return gathered_values

    if not sharp:
        # Grab the pixel values in the 4 corners around each query point
        top_left = gather(floors[0], floors[1], 'top_left') # (H*W, 3)
        top_right = gather(floors[0], ceils[1], 'top_right') # (H*W, 3)
        bottom_left = gather(ceils[0], floors[1], 'bottom_left') # (H*W, 3)
        bottom_right = gather(ceils[0], ceils[1], 'bottom_right') # (H*W, 3)

        # Now do the actual interpolation
        try:
            interp_top = alphas[1].cuda() * (top_right - top_left) + top_left
            interp_bottom = alphas[1].cuda() * (bottom_right - bottom_left) + bottom_left
            interp = alphas[0].cuda() * (interp_bottom - interp_top) + interp_top

        except:
            interp_top = alphas[1] * (top_right - top_left) + top_left
            interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
            interp = alphas[0] * (interp_bottom - interp_top) + interp_top

    else:

        # Grab the pixel values in the 4 corners around each query point
        top_left = gather(floors[0], floors[1], 'top_left') # (H*W, 3)
        top_right = gather(floors[0], ceils[1], 'top_right') # (H*W, 3)
        bottom_left = gather(ceils[0], floors[1], 'bottom_left') # (H*W, 3)
        bottom_right = gather(ceils[0], ceils[1], 'bottom_right') # (H*W, 3)


        alpha_sharp = sharpWarp.apply(alphas[0], alphas[1])
        alpha_sharp = [utils_misc.to_device(ai) for ai in alpha_sharp]


        interp_top = alpha_sharp[1] * (top_right - top_left) + top_left
        interp_bottom = alpha_sharp[1] * (bottom_right - bottom_left) + bottom_left
        interp = alpha_sharp[0] * (interp_bottom - interp_top) + interp_top


    return interp


################################################################################
# Function to run _image_warp()
################################################################################
def apply_warp(im, src_list, dst_list, sharp=False, im2=None):

    new_im = im.clone()
    if im2 is None:
      new_im2 = None
    else:
      new_im2 = im2.clone()

    for i in list(range(len(src_list)))[::-1]:
        src = src_list[i].clone()#.detach()

        src[:,0] = src[:,0]*im.size(2)
        src[:,1] = src[:,1]*im.size(3)

        dst = dst_list[i].clone()#.detach()

        dst[:,0] = dst[:,0]*im.size(2)
        dst[:,1] = dst[:,1]*im.size(3)

        if im2 is None:
          new_im = _image_warp(new_im, dst, dst-src, sharp=sharp)
        else:
          new_im, new_im2, warp_field = _image_warp(new_im, dst, dst-src, sharp=sharp, img2=new_im2)

    if im2 is None:
      return new_im
    else:
      return new_im, new_im2, warp_field
