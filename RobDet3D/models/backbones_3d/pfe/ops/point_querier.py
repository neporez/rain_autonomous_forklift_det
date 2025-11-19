import torch
from .builder import querier
from .....ops.pointnet2.pointnet2_batch import pointnet2_batch_cuda as cuda_ops_batch
from .....ops.pointnet2.pointnet2_stack import pointnet2_stack_cuda as cuda_ops_stack


@torch.no_grad()
@querier.register_module('ball_old')
def basic_ball_query(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor):
    """
    :param ctx:
    :param radius: float, radius of the balls
    :param nsample: int, maximum number of features in the balls
    :param xyz: (B, N, 3) xyz coordinates of the features
    :param new_xyz: (B, npoint, 3) centers of the ball query
    :return:
        idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
    """
    assert new_xyz.is_contiguous()
    assert xyz.is_contiguous()

    B, N, _ = xyz.size()
    npoint = new_xyz.size(1)
    idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()

    cuda_ops_batch.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
    return idx.long()

@torch.no_grad()
@querier.register_module('neighbour')
def neighbour_idx_query(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor, sampled_ind: torch.Tensor):
    # 이웃 인덱스 쿼리 설계해야됌
    B, N, _ = xyz.shape
    M = sampled_ind.shape[1]
    K = nsample
    half_k = K // 2

    device = xyz.device
    base_idx = sampled_ind  # (B, M)
    offsets = torch.arange(-half_k, half_k, device=device).view(1, 1, -1)  # (1, 1, K)
    neighbor_idx = base_idx.unsqueeze(-1) + offsets  # (B, M, K)

    neighbor_idx = neighbor_idx.clamp(0, N - 1)  # (B, M, K)

    B, M, K = neighbor_idx.shape
    batch_idx = torch.arange(B).view(B, 1, 1).expand(B, M, K) 
    neighbor_xyz = xyz[batch_idx, neighbor_idx]

    center_xyz = new_xyz.unsqueeze(2)  # (B, M, 1, 3)
    dists = torch.norm(neighbor_xyz - center_xyz, dim=-1)  # (B, M, K)

    valid_mask = dists < radius
    valid_cnt = torch.full((B, M), fill_value=nsample, device=xyz.device, dtype=torch.long)

    pad_mask = ~valid_mask  # (B, M, K)
    padded_idx = neighbor_idx.clone()  # (B, M, K)

    # fallback: use sampled_ind as padding
    fallback = base_idx.unsqueeze(-1).expand(-1, -1, K)  # (B, M, K)
    padded_idx[pad_mask] = fallback[pad_mask]

    return valid_cnt, padded_idx

class BallQuery(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor):
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()
        idx_cnt = torch.cuda.IntTensor(B, npoint).zero_()

        cuda_ops_batch.ball_query_cnt_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx_cnt, idx)
        return idx_cnt, idx.long()

    @staticmethod
    def symbolic(g: torch._C.Graph, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor):
        return g.op(
            "rd3d::BallQuery", xyz, new_xyz,
            nsample_i=nsample,
            radius_f=radius,
            outputs=2
        )


class GridQuery(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor, voxel: torch.Tensor,
                voxel_hash: torch.Tensor, hash2query: torch.Tensor):
        from .....ops.hvcs import hvcs_cuda
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        idx_cnt, idx = hvcs_cuda.query_from_hash_table(xyz, new_xyz, voxel_hash, hash2query, voxel, radius, nsample)
        return idx_cnt, idx

    @staticmethod
    def symbolic(g: torch._C.Graph, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor,
                 voxel: torch.Tensor, voxel_hash: torch.Tensor, hash2query: torch.Tensor):
        return g.op(
            "rd3d::GridBallQuery", xyz, new_xyz, voxel_hash, hash2query, voxel,
            nsample_i=nsample,
            radius_f=radius,
            outputs=2
        )


ball_query_cnt = querier.register_module('ball')(BallQuery.apply)
grid_ball_query_cnt = querier.register_module('grid_ball')(GridQuery.apply)


# @torch.no_grad()
# @querier.register_module('ball')
# def ball_query_cnt(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor):
#     """
#     :param radius: float, radius of the balls
#     :param nsample: int, maximum number of features in the balls
#     :param xyz: (B, N, 3) xyz coordinates of the features
#     :param new_xyz: (B, npoint, 3) centers of the ball query
#     :return:
#         idx_cnt: (B, npoint) tensor with the number of grouped points for each ball query
#         idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
#     """
#     assert new_xyz.is_contiguous()
#     assert xyz.is_contiguous()
#
#     B, N, _ = xyz.size()
#     npoint = new_xyz.size(1)
#     idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()
#     idx_cnt = torch.cuda.IntTensor(B, npoint).zero_()
#
#     cuda_ops_batch.ball_query_cnt_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx_cnt, idx)
#     return idx_cnt, idx.long()


@torch.no_grad()
@querier.register_module('shell', 'ball_dilated')
def ball_query_dilated(radius_in: float, radius_out: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor):
    """
    :param radius_in: float, radius of the inner balls
    :param radius_out: float, radius of the outer balls
    :param nsample: int, maximum number of features in the balls
    :param xyz: (B, N, 3) xyz coordinates of the features
    :param new_xyz: (B, npoint, 3) centers of the ball query
    :return:
        idx_cnt: (B, npoint) tensor with the number of grouped points for each ball query
        idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
    """
    assert radius_in < radius_out
    assert new_xyz.is_contiguous()
    assert xyz.is_contiguous()
    B, N, _ = xyz.size()
    npoint = new_xyz.size(1)
    idx_cnt = torch.cuda.IntTensor(B, npoint).zero_()
    idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()

    cuda_ops_batch.ball_query_dilated_wrapper(B, N, npoint, radius_in, radius_out, nsample, new_xyz, xyz, idx_cnt, idx)
    return idx_cnt, idx.long()


@torch.no_grad()
@querier.register_module('ball_stack')
def ball_query_stack(radius: float, nsample: int, xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor,
                     new_xyz: torch.Tensor, new_xyz_batch_cnt):
    """
    Args:
        ctx:
        radius: float, radius of the balls
        nsample: int, maximum number of features in the balls
        xyz: (N1 + N2 ..., 3) xyz coordinates of the features
        xyz_batch_cnt: (batch_size), [N1, N2, ...]
        new_xyz: (M1 + M2 ..., 3) centers of the ball query
        new_xyz_batch_cnt: (batch_size), [M1, M2, ...]

    Returns:
        idx: (M1 + M2, nsample) tensor with the indicies of the features that form the query balls
    """
    assert new_xyz.is_contiguous()
    assert new_xyz_batch_cnt.is_contiguous()
    assert xyz.is_contiguous()
    assert xyz_batch_cnt.is_contiguous()

    B = xyz_batch_cnt.shape[0]
    M = new_xyz.shape[0]
    idx = torch.cuda.IntTensor(M, nsample).zero_()

    cuda_ops_stack.ball_query_wrapper(B, M, radius, nsample, new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx)
    empty_ball_mask = (idx[:, 0] == -1)
    idx[empty_ball_mask] = 0
    return empty_ball_mask, idx.long()
