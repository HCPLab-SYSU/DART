from typing import Callable, Optional
import numpy as np
import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple
from .spn import ScorePredNet
from .tools import (
    MLP, get_edges_from_pdf, pdf_to_row_heights, resample_tokens_by_heights, dynamic_image_patch_sample,
    resample_tokens_by_edges, unpatchify
)
import torch.nn.functional as F

class DynamicAdaptiveRegionTokenizer(nn.Module):
    """
    A dynamic tokenizer that extracts patches from an image based on content importance.
    Instead of a fixed grid, it samples variable-sized patches from more important regions,
    determined by a Score Prediction Network (SPN).
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        spn=None,
        high_res=True,
        **kwargs
    ):
        super().__init__()
        stride = patch_size
        # Adjust image size for high-resolution input, processing it at half resolution.
        img_size = to_2tuple(img_size//2 if high_res else img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # Calculate the number of patches in a standard grid layout.
        self.grid_size = (
            (img_size[0] - patch_size[0]) // stride + 1,
            (img_size[1] - patch_size[1]) // stride + 1
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size=16 # Explicitly set patch size for convolution

        # A simple convolutional layer to project fixed-size patches into embedding space.
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

        # Optional normalization layer for the embeddings.
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.dim = embed_dim
        # Score Prediction Network (SPN) to predict region importance. Uses a default if not provided.
        self.spn = spn or ScorePredNet(nn.Identity(), 3, 64)
        self.high_res = high_res

    def forward(self, x, ret_dict=False, pos_embed=None, target_h=14,num_patches=196):
        """
        Forward pass for the tokenizer.

        Args:
            x (torch.Tensor or dict): Input image tensor (B, C, H, W) or a dictionary containing it.
            ret_dict (bool): If True, returns a dictionary of intermediate values.
            pos_embed (torch.Tensor, optional): Positional embeddings to be resampled.
            target_h (int): The target number of vertical regions (rows).
            num_patches (int): The target total number of patches to extract.

        Returns:
            torch.Tensor or dict: The processed image patch embeddings, or a dictionary if ret_dict is True.
        """
        # Unpack input if it's a dictionary
        if isinstance(x, dict):
            target_h = x.get('target_h', None)
            num_patches = x.get('num_patches', None)
            x = x['x']
        B, C, H, W = x.shape
        
        # Determine the downsampling ratio based on high_res flag.
        down_ratio = 2 if self.high_res else 1
        # Assert that the input image dimensions match the model's configuration.
        assert H == self.img_size[0]*down_ratio and W == self.img_size[1]*down_ratio, (
            f"Input image size ({H}x{W}) doesn't match model "
            f"({self.img_size[0]}*{down_ratio}x{self.img_size[1]}*{down_ratio})."
        )
        ret = {'x': x} # Dictionary to store intermediate results
        org_x = x
        n = B
        
        # Interpolate the original image to a fixed size (224x224) for the SPN.
        org_x = F.interpolate(org_x, size=(224, 224), mode='bilinear', align_corners=False)
        # Predict importance scores for image regions using the SPN.
        score = self.spn(x)
        # Normalize scores to get a probability distribution function (PDF).
        pdf = score / score.sum(dim=-1,keepdim=True) # (B, seqlen)

        ret['pdf'] = pdf
        ret['score'] = score
        
        if pos_embed is not None:
            # Repeat positional embeddings for each item in the batch.
            pos_embed = pos_embed.repeat(n, 1, 1)

        # Calculate dynamic row heights based on the importance PDF.
        row_heights = pdf_to_row_heights(pdf, H / down_ratio, target_h=target_h)
        # Resample positional embeddings and the PDF according to the new dynamic row heights.
        pos_embed = resample_tokens_by_heights(pos_embed, row_heights)
        ret['row_heights'] = row_heights
        pdf = resample_tokens_by_heights(pdf.unsqueeze(-1), row_heights).squeeze(-1)
        pdf = pdf / pdf.sum(dim=-1,keepdim=True) # Re-normalize the PDF

        ret['reshaped_pdf']=pdf
        # Determine the horizontal boundaries (edges) for new patches from the resampled PDF.
        new_edges = get_edges_from_pdf(pdf, new_seqlen=num_patches)
        # Resample positional embeddings based on the new horizontal edges.
        pos_embed = resample_tokens_by_edges(pos_embed, new_edges)
        ret['pos'] = pos_embed
        
        # Scale edges to the actual image dimensions.
        new_edges = new_edges * x.size(2)*row_heights.size(1) / new_edges[0,-1].item()
        ret['new_edges'] = new_edges
        
        # Sample patches from the image dynamically using the calculated row heights and edges.
        patches = dynamic_image_patch_sample(x, row_heights * down_ratio, new_edges, shape=(self.patch_size,self.patch_size))

        # Project the sampled patches into embedding space and reshape.
        x = self.proj(patches.reshape(n * num_patches, C, self.patch_size, self.patch_size)).view(n, num_patches, self.dim)

        x = self.norm(x)
        ret['x']=x
        
        return x if not ret_dict else ret

class DynamicAdaptiveImageReshaper(nn.Module):
    """
    A module that reshapes an image by dynamically sampling and reconstructing it,
    effectively performing a content-aware resizing. This is useful for tasks where
    the spatial grid of the image needs to be warped based on content.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        spn=None,
        high_res=True
    ):
        super().__init__()
        stride = patch_size
        img_size = to_2tuple(img_size//2 if high_res else img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # Calculate grid size based on a standard, non-overlapping patch layout.
        self.grid_size = (
            (img_size[0] - patch_size[0]) // stride + 1,
            (img_size[1] - patch_size[1]) // stride + 1
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.patch_size=16
        # Score Prediction Network (SPN) to predict region importance.
        self.spn = spn or ScorePredNet(nn.Identity(), 3, 64)
        self.high_res=high_res

    def forward(self, x, ret_dict=False, pos_embed=None, shape=(14,14)):
        """
        Forward pass for the reshaper.

        Args:
            x (torch.Tensor): Input image tensor (B, C, H, W).
            ret_dict (bool): If True, returns a dictionary of intermediate values.
            pos_embed (torch.Tensor, optional): Positional embeddings to be resampled.
            shape (tuple): The target grid shape (target_height, target_width) for reshaping.

        Returns:
            torch.Tensor or dict: The reshaped image, or a dictionary if ret_dict is True.
        """
        B, C, H, W = x.shape
        
        # Assert that the input image dimensions match the model's configuration.
        down_ratio = 2 if self.high_res else 1
        # Assert that the input image dimensions match the model's configuration.
        assert H == self.img_size[0]*down_ratio and W == self.img_size[1]*down_ratio, (
            f"Input image size ({H}x{W}) doesn't match model "
            f"({self.img_size[0]}*{down_ratio}x{self.img_size[1]}*{down_ratio})."
        )
        # Halve dimensions if using high-resolution mode.
        if self.high_res:
            H, W = H//2, W//2
            
        target_h, target_w = shape
        ret = {'x': x} # Dictionary to store intermediate results
        org_x = x
        n = B
        
        # If high-res, interpolate to a standard size for the SPN.
        if self.high_res:
            org_x = F.interpolate(org_x, size=(224, 224), mode='bilinear', align_corners=False)
            
        # Predict importance scores and normalize to get a PDF.
        score = self.spn(x)
        pdf = score / score.sum(dim=-1,keepdim=True) # (B, seqlen)

        ret['pdf'] = pdf
        ret['score'] = score

        # Calculate dynamic row heights based on the vertical importance.
        row_heights = pdf_to_row_heights(pdf, x.size(2) // (1 if not self.high_res else 2), target_h=target_h)
        ret['row_heights'] = row_heights
        # Resample the PDF according to the new row heights and re-normalize.
        pdf = resample_tokens_by_heights(pdf.unsqueeze(-1), row_heights).squeeze(-1)
        pdf = pdf / pdf.sum(dim=-1,keepdim=True) 
        ret['reshaped_pdf']=pdf

        # Calculate dynamic column heights based on horizontal importance.
        # This is done by transposing the assumed 2D layout of the PDF and reusing the same logic.
        col_heights = pdf_to_row_heights(ret['pdf'].view(n,14,14).permute(0,2,1).reshape(n,-1), x.size(2) // (1 if not self.high_res else 2), target_h=target_h)
        col_pos = torch.cumsum(col_heights,dim=1) # Cumulative sum to find column boundaries.
        
        # Construct the patch sampling grid edges from column positions.
        new_edges = torch.cat([torch.ones_like(col_pos[:,:1])]+[col_pos+224*i for i in range(target_h)],dim=1)
        # Scale edges to the image's dimensions.
        new_edges = new_edges * x.size(2)*col_heights.size(1) / new_edges[0,-1].item()
        ret['new_edges'] = new_edges

        # Sample patches dynamically using the calculated row heights and column edges.
        down_ratio = 2 if self.high_res else 1
        patches = dynamic_image_patch_sample(x, row_heights * down_ratio, new_edges, shape=(self.patch_size,self.patch_size))
        # Reconstruct the image from the dynamically sampled patches.
        x = unpatchify(patches, self.patch_size, shape)
        ret['x'] = x
        
        # If positional embeddings are provided, resample them both vertically and horizontally.
        if pos_embed is not None:
            pos_embed = pos_embed.repeat(n, 1, 1)
            # Resample vertically based on row heights.
            pos_embed = resample_tokens_by_heights(pos_embed, row_heights/row_heights.sum()*224)
            # Reshape, resample horizontally based on column heights, and reshape back.
            pos_embed = resample_tokens_by_heights(pos_embed.view(n,target_h,14,self.dim).permute(0,2,1,3).reshape(n,-1,self.dim), col_heights/col_heights.sum()*224, org_h=target_h).view(n,target_w,target_h,self.dim).permute(0,2,1,3).reshape(n,-1,self.dim)
            ret['pos'] = pos_embed
            
        return x if not ret_dict else ret