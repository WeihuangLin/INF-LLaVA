#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from ..constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from ..mm_utils import get_anyres_image_grid_shape

import torch.nn.functional as F
import torch.nn.init as init
import math
import copy



def combine_crop_features_N(tensors,n):
    """
    Combine tensors in a grid order.
    
    Args:
    tensors (list of torch.Tensor): List of tensors to combine.
    n (int): Number of tensors in each row/column.
    
    Returns:
    torch.Tensor: The combined target tensor.
    """
    rows = []
    for i in range(n):
        row = torch.cat(tensors[i*n : (i+1)*n], dim=2)
        rows.append(row)
    target_tensor = torch.cat(rows, dim=1)

    return target_tensor


def combine_stride_features_N(features_list,N):
    """
    Combine feature tensors by placing them in a grid with stride-based concatenation.
    
    Args:
    features_list (list of torch.Tensor): List of feature tensors to combine.
    N (int): Number of tensors in each row/column.
    
    Returns:
    torch.Tensor: The combined feature tensor.
    """
    # Get the shape of the feature tensors
    b, H, W, d = features_list[0].size()
    combined_features = torch.zeros(b, N*H, N*W, d).to(features_list[0].device).to(features_list[0].dtype)
    
    # Place each sub-feature in the correct grid location
    for h in range(N):
        for w in range(N):
            sub_image = features_list[h * N + w]
            combined_features[:, h::N, w::N, :] = sub_image
    
    return combined_features



def crop_pixels_N(image_tensor, N):
    """
    Split an image tensor into non-overlapping blocks.

    Args:
    image_tensor (torch.Tensor): The original image tensor of shape (B, C, H, W).
    N (int): Number of blocks per each dimension (N x N grid).

    Returns:
    list of torch.Tensor: List of cropped blocks.
    """
    B, C, H, W = image_tensor.shape
    assert H % N == 0 and W % N == 0, "Invalid dimensions for splitting."
    # Calculate the height and width of each block
    block_height = H // N
    block_width = W // N
    split_list = []
    for i in range(N):
        for j in range(N):
            # Calculate the starting and ending indices for each block
            start_h = i * block_height
            end_h = start_h + block_height
            start_w = j * block_width
            end_w = start_w + block_width
            # Get the block from the image tensor
            block = image_tensor[:, :, start_h:end_h, start_w:end_w]
            # Append the block to the split list
            split_list.append(block)
    return split_list


def extract_pixels_N(image_tensor, N):
    """
    Extract pixels from an image tensor based on stride.

    Args:
    image_tensor (torch.Tensor): The original image tensor of shape (B, C, H, W).
    N (int): Stride value for extracting pixels.

    Returns:
    list of torch.Tensor: List of extracted sub-images.
    """
    image_list = []
    _, _, height, width = image_tensor.shape
    block_size = height // N
    for h in range(0, N):
        for w in range(0, N):
            sub_image = image_tensor[:, :, h::N, w::N]
            image_list.append(sub_image)
    return image_list


def attention(query, key, value, mask=None):
    """
    Compute the Scaled Dot-Product Attention.

    Args:
    query (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_k).
    key (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_k).
    value (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_k).
    mask (torch.Tensor, optional): Mask tensor to mask out certain positions.

    Returns:
    torch.Tensor: Attention output and attention weights.
    p_attn (torch.Tensor): The attention weights.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    return torch.matmul(p_attn, value), p_attn

def clones(module, N):
    """
    Produce N identical layers (deepcopies).

    Args:
    module (nn.Module): The module to replicate.
    N (int): The number of copies.

    Returns:
    nn.ModuleList: A list containing N identical modules.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model):
        """
        Initialize the multi-headed attention layer.

        Args:
        h (int): Number of attention heads.
        d_model (int): Dimension of the model.
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0                             
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.ln = nn.LayerNorm(d_model)
        

    def forward(self, q, k, v, mask=None):
        """
        Forward pass for multi-headed attention.

        Args:
        q (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_model).
        k (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_model).
        v (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_model).
        mask (torch.Tensor, optional): Mask tensor to mask out certain positions.

        Returns:
        torch.Tensor: The result of the multi-head attention operation.
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = q.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (q, k, v))]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        x = self.ln(self.linears[-1](x)+q)
        return x

class SpatialEmbed(nn.Module):
    def __init__(self, spatial=24,channel=1024):
        super(SpatialEmbed, self).__init__()

        self.attention = MultiHeadedAttention(8,channel)

        self.ffn_crop = nn.Sequential(
            nn.Linear(channel,channel//16),
            nn.GELU(),
            nn.Linear(channel//16,channel//2),
        )

        self.ffn_stride = nn.Sequential(
            nn.Linear(channel,channel//16),
            nn.GELU(),
            nn.Linear(channel//16,channel//2),
        )
        
        self.ln_fuse = nn.LayerNorm(channel)
        

    def forward(self, img_feats,n=2):
        b,hw,d = img_feats[0].shape
        h=w=int(hw**0.5)

        # Reshape image features from (b, hw, d) to (b, h, w, d)
        for i in range(len(img_feats)):
            img_feats[i] = img_feats[i].reshape(b, h, w, d)

        # Separate the image features for crop and stride
        crop_img_feats = img_feats[:n * n]  # (b, h, w, d) x n * n
        stride_img_feats = img_feats[n * n: 2 * n * n]  # (b, h, w, d) x n * n
        
        # Combine crop and stride features to form larger scales
        scale_crop_img_feat = combine_crop_features_N(crop_img_feats, n)  # (b, nh, nw, d)
        scale_stride_img_feat = combine_stride_features_N(stride_img_feats, n)  # (b, nh, nw, d)


        # Cross attention processing
        global_enhanced_local_feats = []
        local_enhanced_global_feats = []
        
        # Global as Query -- crop_pixels_N
        local_feats_stride = crop_pixels_N(scale_crop_img_feat.permute(0, 3, 1, 2), n)  # (b, d, h, w) x n * n
        global_feats_stride = crop_pixels_N(scale_stride_img_feat.permute(0, 3, 1, 2), n)  # (b, d, h, w) x n * n

        for i in range(n * n):
            local_feats_stride[i] = local_feats_stride[i].permute(0, 2, 3, 1).reshape(b, hw, d)  # (b, h * w, d)
            global_feats_stride[i] = global_feats_stride[i].permute(0, 2, 3, 1).reshape(b, hw, d)  # (b, h * w, d)
            local_enhanced_global_feat = self.attention(global_feats_stride[i], local_feats_stride[i], local_feats_stride[i])  # (b, h * w, d)
            local_enhanced_global_feats.append(local_enhanced_global_feat.reshape(b, h, w, d))  # (b, h, w, d)
            
        # Local as Query -- extract_pixels_N
        local_feats_crop = extract_pixels_N(scale_crop_img_feat.permute(0, 3, 1, 2), n)  # (b, d, h, w) x n * n
        global_feats_crop = extract_pixels_N(scale_stride_img_feat.permute(0, 3, 1, 2), n)  # (b, d, h, w) x n * n

        for i in range(n*n):
            local_feats_crop[i] = local_feats_crop[i].permute(0,2,3,1).reshape(b,hw,d) #(b,h*w,d)
            global_feats_crop[i] = global_feats_crop[i].permute(0,2,3,1).reshape(b,hw,d) #(b,h*w,d)
            global_enhanced_local_feat = self.attention(local_feats_crop[i], global_feats_crop[i], global_feats_crop[i])  #(b,h*w,d)
            global_enhanced_local_feats.append(global_enhanced_local_feat.reshape(b, h, w, d)) #(b,h,w,d)
            
        # Combine local and global enhanced features
        scale_stride_img_feat = combine_crop_features_N(local_enhanced_global_feats, n)  # (b, 2h, 2w, d)
        scale_crop_img_feat = combine_stride_features_N(global_enhanced_local_feats, n)  # (b, 2h, 2w, d)

        # Apply the feed-forward networks
        scale_crop_img_feat_down = self.ffn_crop(scale_crop_img_feat)  # (b, 2h, 2w, d // 2)
        scale_stride_img_feat_down = self.ffn_stride(scale_stride_img_feat)  # (b, 2h, 2w, d // 2)

        # Fuse the features by concatenation and layer normalization
        scale_fuse_img_feat = torch.cat([scale_crop_img_feat_down, scale_stride_img_feat_down], dim=-1)  # (b, 2h, 2w, d)
        scale_fuse_img_feat = self.ln_fuse(scale_fuse_img_feat + scale_stride_img_feat + scale_crop_img_feat)  # (b, 2h, 2w, d)

        # Final output processing, interpolate to the target spatial size
        scale_fuse_img_feat = scale_fuse_img_feat.permute(0, 3, 1, 2)  # (b, d, 2h, 2w)
        output = F.interpolate(scale_fuse_img_feat.to(torch.float32), size=(24, 24), mode='bilinear', align_corners=False).to(img_feats[0].dtype)
        output = output.permute(0, 2, 3, 1)  # (b, h, w, d)
        output = output.reshape(b, hw, d)

        return output



class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.spatial_embed = SpatialEmbed(spatial=24,channel=1024)
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )


    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
            self.spatial_embed = SpatialEmbed(spatial=24,channel=1024)


            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            # print(mm_projector_weights.keys())
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            self.spatial_embed.load_state_dict(get_w(mm_projector_weights, 'spatial_embed'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        images_list = []
        # Resize images to 3x the original size and apply bicubic interpolation
        images_3x3 = torch.nn.functional.interpolate(images.to(torch.float32), (336*3, 336*3), mode='bicubic', align_corners=False).to(images.dtype)
        # Crop and extract patches from resized images
        crop_images_3x3 = crop_pixels_N(images_3x3, 3)  # List of cropped images [b, C, H, W] * 9
        stride_images_3x3 = extract_pixels_N(images_3x3, 3)  # List of strided images [b, C, H, W] * 9
        # Append cropped and strided images to the image list
        images_list += crop_images_3x3
        images_list += stride_images_3x3
        
        # Resize images to the original size and apply bicubic interpolation
        resize_images = [torch.nn.functional.interpolate(images.to(torch.float32), (336, 336), mode='bicubic', align_corners=False).to(images.dtype)]  # List of resized images [b, C, H, W] * 1
        images_list += resize_images

        # Initialize a list to store image features
        img_features = []
        # Extract features for each sub-image using the vision tower
        for img in images_list:
            sub_image_features = self.get_model().get_vision_tower()(img)
            img_features.append(sub_image_features)  # List of features [b, h*w, c] * 10


        # Constants for indexing into the feature list
        n3 = 2 * (3 ** 2)
        # Merge image features for 3x resized images
        img_features_3x3_list = img_features[:n3]
        image_features_3x3 = self.get_model().spatial_embed(img_features_3x3_list, n=3)  # Output features [b, h*w, c]

        # Take the last feature set for 1x resized image
        img_features_1x1 = img_features[-1]
        
        # Combine the features from different scales using mean pooling
        image_features = torch.mean(torch.stack([img_features_1x1, image_features_3x3], dim=-1), dim=-1)  # [b, h*w, c]
        # Project the combined features to a higher-dimensional space
        image_features = self.get_model().mm_projector(image_features)  # [b, h*w, 4*c]

        return image_features  # Return the final image features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
