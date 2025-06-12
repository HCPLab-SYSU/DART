import torch
import os
import torch.nn.functional as F
import torch.nn as nn
import importlib.util
import math
import matplotlib.pyplot as plt
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        out = self.net(x)  # (n, seqlen, 1)
        return out  # (n, seqlen)

def dynamic_image_patch_sample(images, row_heights, new_edges, shape=(16,16), visualize=False, version='v2'):
    seqlen = new_edges.size(1)-1
    device = images.device
    tar_h, tar_w = shape
    
    images_reshaped = resample_image_by_heights(images, row_heights, max(tar_h, 2), visualize=visualize, version=version)
    n, c, hh, ww = images_reshaped.shape  # hh=16
    
    
    
    x_starts = new_edges[:, :-1]  # (n, seqlen)
    x_ends   = new_edges[:, 1:]   # (n, seqlen)
    if version == 'v1':
        t_lin = torch.linspace(0, 1, steps=tar_w, device=device).view(1,1,tar_w)   # =>(1,1,tar_w)
    else:
        t_lin = torch.arange(0,tar_w, device=images.device).view(1,1,tar_w)/tar_w   # =>(1,1,tar_w)
    t_lin = t_lin.expand(n, seqlen, tar_w)
    x_starts_ex = x_starts.unsqueeze(-1)  # (n,seqlen,1)
    x_ends_ex   = x_ends.unsqueeze(-1)    # (n,seqlen,1)
    
    x_coords_all = x_starts_ex + (x_ends_ex - x_starts_ex) * t_lin
    
    x_coords_all = x_coords_all.reshape(n, seqlen*tar_w)  # => (n,tar_w*seqlen)
    
    y_1d = torch.linspace(0, hh-1, steps=tar_h, device=device)  # =>(tar_h,)
    y_2d = y_1d.view(1, tar_h).expand(n, -1)  # => (n,tar_h)
    x_grid = x_coords_all.unsqueeze(1).expand(-1,tar_h,-1)  # =>(n,tar_h,tar_w*seqlen)
    y_grid = y_2d.unsqueeze(-1).expand(-1,-1,seqlen*tar_w)  # =>(n,tar_h,tar_w*seqlen)
    
    x_grid_norm = 2.0 * (x_grid / (ww - 1)) - 1.0
    y_grid_norm = 2.0 * (y_grid / (hh - 1)) - 1.0
    
    grid = torch.stack([x_grid_norm, y_grid_norm], dim=-1)
    patches_wide = F.grid_sample(
        images_reshaped,  # (n,c,16,ww)
        grid,             # (n,16,16*seqlen,2)
        mode='bilinear',
        align_corners=True
    )
    patches_5d = patches_wide.reshape(n, c, tar_h, seqlen, tar_w)
    patches = patches_5d.permute(0, 3, 1, 2, 4)  # => (n,seqlen,c,16,16)
    
    if visualize and n>0:
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=images.device).view(3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616], device=images.device).view(3, 1, 1)
        mean_images = patches * std + mean
        mean_images = mean_images.clamp(0, 1)  # [batch_size, 3, 32, 32]
        for j in range(seqlen):
            patch_j = mean_images[0, j]  # => (3,16,16)
            patch_np = patch_j.permute(1,2,0).detach().cpu().numpy()
            plt.figure(figsize=(2,2))
            plt.imshow(patch_np)
            plt.title(f"Patch#{j+1} of Image#0")
            plt.axis("off")
            plt.savefig(f"patch_{j}.png")
            plt.close()
    return patches

def resample_image_by_heights(images, row_heights, final_row_height, visualize=False, version='v2'):
    N, C, H, W = images.shape
    num_rows = row_heights.size(1)
    
    cumsum_heights = row_heights.cumsum(dim=1)  # shape: (N, 14)

    row_chunks = []
    
    if version == 'v1':
        h_lin = torch.linspace(0, 1, steps=final_row_height, device=images.device)  
    else:
        h_lin = torch.arange(0, final_row_height, device=images.device)/(final_row_height)
    w_lin = torch.linspace(0, 1, steps=W, device=images.device)

    h_grid, w_grid = torch.meshgrid(h_lin, w_lin, indexing='ij')  # shape(16,W), (16,W)
    
    for i in range(num_rows):
        if i == 0:
            start_y = torch.zeros_like(cumsum_heights[:, i])           # (N,)
        else:
            start_y = cumsum_heights[:, i-1]                            # (N,)
        end_y = cumsum_heights[:, i]                                    # (N,)

        row_range = end_y - start_y  # (N,) 

        #
        start_y_ = start_y.view(N, 1, 1)
        row_range_ = row_range.view(N, 1, 1)
        
        h_grid_expanded = h_grid.unsqueeze(0).expand(N, -1, -1)
        w_grid_expanded = w_grid.unsqueeze(0).expand(N, -1, -1)
        
        source_y = (start_y_ + row_range_ * h_grid_expanded) / (H - 1) * 2.0 - 1.0
        source_x = w_grid_expanded * 2.0 - 1.0

        grid = torch.stack([source_x, source_y], dim=-1)  # (N,16,W,2)

        row_chunk = F.grid_sample(
            images, 
            grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )
        
        row_chunks.append(row_chunk)
    
    rearranged_images = torch.cat(row_chunks, dim=3)

    if visualize:
        import numpy as np
        img_np = rearranged_images[0].permute(1, 2, 0).detach().cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        
        plt.figure(figsize=(12, 4))
        plt.title("Rearranged Image with Variable Row Heights")
        plt.imshow(img_np)
        plt.axis('off')
        plt.savefig('rearranged_variable_height.png')
        plt.close()

    return rearranged_images

def resample_tokens_by_heights(x, row_heights, org_h=14):
    N, patch_count, D = x.shape
    
    old_tokens_2d = x.view(N, -1, org_h, D)  # (N, j=14, col=14, D)

    row_heights_clamped = row_heights.clone()
    row_heights_clamped[row_heights_clamped < 0] = 0

    cumsum_heights = row_heights_clamped.cumsum(dim=1)  # (N, max_row_num)
    new_starts = torch.cat([
        torch.zeros(N, 1, device=row_heights.device, dtype=row_heights.dtype),
        cumsum_heights[:, :-1]
    ], dim=1)  # (N, max_row_num)
    new_ends = cumsum_heights  # (N, max_row_num)

    old_starts = 16 * torch.arange(old_tokens_2d.size(1), device=x.device, dtype=row_heights.dtype)  # (14,)
    old_ends   = old_starts + 16  # (14,)

    new_starts_expanded = new_starts.unsqueeze(-1)       # (N, max_row_num, 1)
    new_ends_expanded   = new_ends.unsqueeze(-1)         # (N, max_row_num, 1)
    old_starts_expanded = old_starts.view(1, 1, -1)      # (1, 1, 14)
    old_ends_expanded   = old_ends.view(1, 1, -1)        # (1, 1, 14)

    # overlap_length: (N, max_row_num, 14)
    overlap_length = (
        torch.min(new_ends_expanded, old_ends_expanded)
        - torch.max(new_starts_expanded, old_starts_expanded)
    ).clamp(min=0)
    overlap_ratio = overlap_length / 16.0

    overlap_sum = overlap_ratio.sum(dim=2, keepdim=True).clamp(min=1e-6)
    overlap_ratio = overlap_ratio / overlap_sum

    # old_tokens_2d: (N, j=14, col=14, D)
    # overlap_ratio:    (N, i=max_row_num, j=14)
    new_tokens_2d = torch.einsum('b i j, b j c d -> b i c d', overlap_ratio, old_tokens_2d)

    new_tokens = new_tokens_2d.view(N, -1, D)

    return new_tokens

def find_quantiles(p_values, pdf, eps=1e-8):

    if p_values.numel() == 0:
        return torch.empty(pdf.size(0), 0, device=pdf.device, dtype=pdf.dtype)
    
    n, seqlen = pdf.shape
    num_quant = p_values.shape[0]
    cumsums = torch.cumsum(pdf, dim=1)  # (N, num_rows)
    edges = torch.linspace(
                0, seqlen, seqlen + 1, device=pdf.device, dtype=pdf.dtype
            ).unsqueeze(0).repeat(n, 1)

    p_values_expanded = p_values.view(1, 1, num_quant) 
    cumsums_expanded = cumsums.unsqueeze(-1)  # (n, seqlen, 1)
    
    mask = (cumsums_expanded >= p_values_expanded)  # (n, seqlen, num_quant)
    j_indices = torch.argmax(mask.int(), dim=1)     # (n, num_quant)

    mask_sum = mask.sum(dim=1)  # (n, num_quant)
    no_true_mask = (mask_sum == 0)
    j_indices = torch.where(no_true_mask, torch.full_like(j_indices, seqlen - 1), j_indices)

    prev_j = torch.clamp(j_indices - 1, 0, seqlen - 1)
    prev_area = torch.gather(cumsums, dim=1, index=prev_j)
    j_zero_mask = (j_indices == 0)
    prev_area = torch.where(j_zero_mask, torch.zeros_like(prev_area), prev_area)

    pdf_val = torch.gather(pdf, dim=1, index=j_indices)
    edge_val = torch.gather(edges[:, :-1], dim=1, index=j_indices)

    p_values_expanded_n = p_values.unsqueeze(0).expand(n, num_quant)
    quantiles = edge_val + (p_values_expanded_n - prev_area) / (pdf_val + eps)

    return quantiles

def pdf_to_row_heights(pdf, total_height=224, eps=1e-8, version='x', target_h=None):
    N, num_patches = pdf.shape
    
    if version == 'r':
        row_pdf = pdf
        num_rows = pdf.size(1)
    else:
        num_cols = 14
        num_rows = num_patches // num_cols  

        pdf_2d = pdf.view(N, num_rows, num_cols)  # (N, num_rows, num_cols)
        row_pdf = pdf_2d.sum(dim=-1)  # (N, num_rows)
    if not target_h:
        target_h = num_rows

    row_sum = row_pdf.sum(dim=-1, keepdim=True) + eps
    row_pdf = row_pdf / row_sum  # (N, num_rows)

    # [0, 1/num_rows, 2/num_rows,..., 1]
    quant_p = torch.linspace(0, 1, target_h + 1, device=pdf.device, dtype=pdf.dtype)[1:-1]
    edges = torch.linspace(0, num_rows, num_rows + 1, device=pdf.device, dtype=pdf.dtype)  # (num_rows + 1,)
    edges = edges.unsqueeze(0).expand(N, -1)  # (N, num_rows + 1)
    if quant_p.numel() > 0:
        quantiles = find_quantiles(quant_p, row_pdf)  # (N, num_rows)
        new_edges = torch.cat([edges[:, :1], quantiles, edges[:, -1:]], dim=1)  # (N, num_rows + 1)
    else:
        new_edges = torch.cat([edges[:, :1], edges[:, -1:]], dim=1)  # (N, 2)

    raw_row_heights = new_edges[:, 1:] - new_edges[:, :-1]  # (N, num_rows)
    
    row_heights = raw_row_heights * (total_height / num_rows)  # (N, num_rows)

    return row_heights

def get_base_edges(seqlen, x: torch.Tensor):
    edges = torch.linspace(0, seqlen, seqlen + 1, device=x.device, dtype=x.dtype)  # (num_rows + 1,)
    edges = edges.unsqueeze(0).expand(x.size(0), -1)  # (N, num_rows + 1)
    return edges

def get_edges_from_pdf(pdf, new_seqlen=None):
    seqlen = pdf.size(1)
    new_seqlen = new_seqlen or seqlen
    edges = get_base_edges(seqlen, pdf)
    quant_p = torch.linspace(0, 1, new_seqlen + 1, device=pdf.device, dtype=pdf.dtype)[1:-1]

    if quant_p.numel() > 0:
        _edges = get_base_edges(pdf.size(1), pdf)
        quantiles = find_quantiles(quant_p, pdf)
        new_edges = torch.cat([_edges[:, 0:1], quantiles, _edges[:, -1:]], dim=1)/pdf.size(1)*seqlen
    else:
        new_edges = torch.cat([edges[:, 0:1], edges[:, -1:]], dim=1)

    return new_edges

def resample_tokens_by_edges(tokens, edges):
    seqlen = tokens.size(1)
    old_edges = get_base_edges(seqlen, tokens)
    edges = edges / edges.max() * old_edges.max()

    # 计算 overlap (n, new_seqlen, seqlen)
    new_start = edges[:, :-1].clone().unsqueeze(2)  # (n, new_seqlen, 1)
    new_end   = edges[:, 1:].clone().unsqueeze(2)   # (n, new_seqlen, 1)

    old_start = old_edges[:, :-1].clone().unsqueeze(1)      # (n, 1, seqlen)
    old_end   = old_edges[:, 1:].clone().unsqueeze(1)       # (n, 1, seqlen)

    overlap = torch.clamp(
        torch.min(new_end, old_end) - torch.max(new_start, old_start),
        min=0.0
    )  # (n, new_seqlen, seqlen)

    raw_weights = overlap# * pdf.unsqueeze(1)  # (n, new_seqlen, seqlen)
    sum_weights = raw_weights.sum(dim=2, keepdim=True) + 1e-12
    weight = raw_weights / sum_weights  # (n, new_seqlen, seqlen)

    # weight: (n, new_seqlen, seqlen)
    # tokens: (n, seqlen, dim)
    # print(weight.shape, tokens.shape)
    new_tokens = torch.bmm(weight, tokens)  # (n, new_seqlen, dim)
    return new_tokens


def unpatchify(patches: torch.Tensor, patch_size: int, shape=None) -> torch.Tensor:
    N, num_patches, C, P, _ = patches.shape
    assert P == patch_size
    if shape is None:
        h = int(math.sqrt(num_patches))
        w=h
    else:
        h, w = shape
    assert h * w == num_patches
    
    # (N, H*W, C, P, P) → (N, H, W, C, P, P)
    patches = patches.view(N, h, w, C, P, P)
    patches = patches.permute(0, 3, 1, 4, 2, 5)
    imgs = patches.reshape(N, C, h * P, w * P)
    return imgs
