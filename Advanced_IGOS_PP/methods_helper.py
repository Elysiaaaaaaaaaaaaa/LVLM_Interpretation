"""
Helper function for the IGOS explanation methods.
© copyright 2024 Bytedance Ltd. and/or its affiliates.
Modified from Tyler Lawson, Saeed khorram. https://github.com/saeed-khorram/IGOS
"""

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import math
from torch.nn import UpsamplingBilinear2d
from .utils import *
# from IGOS_pp import pred_probs

def cosine_decay(init, iter):
    return init * (1 + math.cos(math.pi * iter)) / 2

def exp_decay(init, iter, gamma=0.2):
    return init * math.exp(-gamma * iter)

def tv_norm(image, beta=2):
    """
    Calculates the total variation.
    :param image:
    :param beta:
    :return:
    """
    image = image[:, 0, :, :]
    a = torch.mean(torch.abs((image[:, :-1, :] - image[:, 1:, :]).view(image.shape[0], -1)).pow(beta), dim=1)
    b = torch.mean(torch.abs((image[:, :, :-1] - image[:, :, 1:]).view(image.shape[0], -1)).pow(beta), dim=1)
    return a + b


def bilateral_tv_norm(image, mask, tv_beta=2, sigma=1):
    """
        Calculates the bilateral total variation.

    :param image:
    :param mask:
    :param tv_beta:
    :param sigma:
    :return:
    """
    # tv term
    mask_ = mask[:, 0, :]
    a = torch.mean(torch.abs((mask_[:, :-1, :] - mask_[:, 1:, :]).view(mask.shape[0], -1)).pow(tv_beta), dim=1)
    b = torch.mean(torch.abs((mask_[:, :, :-1] - mask_[:, :, 1:]).view(mask.shape[0], -1)).pow(tv_beta), dim=1)
    # bilateral tv in the image space
    if isinstance(image, list):
        image = image[0]
    elif image.shape[0] > 1:
        image = image[0].unsqueeze(0)

    up_mask_ = upscale(mask, image)
    bil_a = torch.mean(torch.exp(-(image[:, :, :-1, :] - image[:, :, 1:, :]).mean(dim=1) ** 2 / sigma).view(mask.shape[0], -1)
                    * torch.abs((up_mask_[:, :, :-1, :] - up_mask_[:, :, 1:, :]).view(up_mask_.shape[0], -1)).pow(tv_beta), dim=1)
    bil_b = torch.mean(torch.exp(-(image[:, :, :, :-1] - image[:, :, :, 1:]) ** 2 / sigma).mean(dim=1).view(mask.shape[0], -1)
                    * torch.abs((up_mask_[:, :, :, :-1] - up_mask_[:, :, :, 1:]).view(up_mask_.shape[0], -1)).pow(tv_beta), dim=1)
    return 0.5 * (a + b + bil_a + bil_b)


# def upscale(masks, images, resolution):
#     if isinstance(images, list):
#         output_masks = []
#         for image in images:
#             upscale_fn = UpsamplingBilinear2d(size=image.shape[-2:]).cuda()
#             output_masks.append(upscale_fn(masks).expand((-1,1,)+image.shape[-2:]))
#     elif images.shape[0] == 1:
#         upscale_fn = UpsamplingBilinear2d(size=images.shape[-2:]).cuda()
#         output_masks = upscale_fn(masks).expand((-1,1,)+images.shape[-2:])
#     else:
#         # divide mask to patches
#         # if resolution == None:
#         #     #patch_size = int(masks.shape[-1] / math.sqrt(images.shape[0] - 1))
#         #     if resolution[0] != resolution[1]:
#         #         ratio = resolution[0] / resolution[1]
#         #         if ratio > 1:
#         #             new_height = masks.shape[-1]
#         #             new_width = int(new_height * ratio)
#         #         else:
#         #             new_width = masks.shape[-1]
#         #             new_height = int(new_width / ratio)
#         #         masks = F.interpolate(masks, size=(new_width, new_height), mode='bilinear', align_corners=False)
#         # else:
#         patch_size1 = images.shape[-1] // masks.shape[-1]
#         patch_size2 = images.shape[-2] // masks.shape[-2]
#         patches = masks.unfold(2, patch_size2, patch_size1).unfold(3, patch_size2, patch_size1)
#         patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
#         patches = patches.view(-1, masks.shape[1], patch_size2, patch_size1)
#         upscale_fn = UpsamplingBilinear2d(size=images.shape[-2:]).cuda()
#         up_patches = upscale_fn(patches)
#         up_mask_origin = upscale_fn(masks)
#         output_masks = torch.cat((up_mask_origin, up_patches))
#         assert output_masks.shape[0] == images.shape[0]

#     return output_masks

def upscale(masks: torch.Tensor, images, mode: str = 'bilinear', align_corners: bool = False):
    """
    将 masks 上采样到与 images 的空间尺寸一致。
    支持 images 为:
      - 单图 3D: [C, H, W]
      - 批量 4D: [B, C, H, W]
      - list[Tensor]: 每个为 [C,H,W] 或 [H,W]
    masks 形状可为 [h,w] / [B,h,w] / [B,1,h,w] / [1,1,h,w]
    """
    # 规范 masks -> [B,1,h,w]
    if masks.dim() == 2:
        masks = masks.unsqueeze(0).unsqueeze(0)           # [1,1,h,w]
    elif masks.dim() == 3:
        masks = masks.unsqueeze(1)                         # [B,1,h,w]
    elif masks.dim() == 4 and masks.size(1) != 1:         # [B,C,h,w] 但 C!=1
        masks = masks[:, :1, ...]                          # 取第1通道

    def _interp(msk, size_hw, dev):
        H, W = size_hw
        msk = msk.to(dev).float()
        return F.interpolate(msk, size=(H, W), mode=mode, align_corners=align_corners)

    # --- images 为单图 3D: [C,H,W]
    if isinstance(images, torch.Tensor) and images.dim() == 3:
        _, H, W = images.shape
        # 若 masks 的 batch 是 1，就保持 1；也可按需 expand 到 1
        return _interp(masks, (H, W), images.device)       # -> [B,1,H,W]

    # --- images 为批量 4D: [B,C,H,W]
    if isinstance(images, torch.Tensor) and images.dim() == 4:
        B, _, H, W = images.shape
        if masks.size(0) == 1 and B > 1:
            masks = masks.expand(B, -1, -1, -1).contiguous()
        elif masks.size(0) != B:
            raise ValueError(f"Batch mismatch: masks B={masks.size(0)}, images B={B}")
        return _interp(masks, (H, W), images.device)       # -> [B,1,H,W]

    # --- images 为 list
    if isinstance(images, list):
        outs = []
        for img in images:
            if img.dim() == 2:     # [H,W]
                H, W = img.shape[-2], img.shape[-1]
                dev = img.device
            elif img.dim() == 3:   # [C,H,W]
                _, H, W = img.shape
                dev = img.device
            else:
                raise ValueError("Each image in list must be 2D or 3D tensor.")
            outs.append(_interp(masks, (H, W), dev))       # -> [B,1,H,W]
        return outs

    raise ValueError("images must be a 3D/4D Tensor or a list of images.")

def interval_score(args, model, model_name, images, baseline, label, up_masks, num_iter, noise=True, 
                   prompt=None, image_size=None, positions=None):
    if model_name == 'llava' or model_name == 'llava_next' or model_name == 'mgm':
        # The intervals to approximate the integral over
        intervals = torch.linspace(1/num_iter, 1, num_iter, requires_grad=False).cuda().view(-1, 1, 1, 1)
        interval_masks = up_masks.unsqueeze(1) * intervals
        local_images = phi(images.unsqueeze(1), baseline.unsqueeze(1), interval_masks)

        if noise:
            local_images = local_images + torch.randn_like(local_images) * .2

        local_images = local_images.transpose(0, 1)
        input_ids = torch.cat((prompt, label), dim=1)
        positions = torch.tensor(positions).to(input_ids.device)

        losses = torch.tensor(0.).to(input_ids.device)
        for single_img in local_images:
            single_img = single_img.half()
            probs = pred_probs(args, model, input_ids, label, single_img, image_size)
            #losses += probs[positions].mean()
            losses += torch.log(probs)[positions].sum()
    
    elif model_name == 'cambrian':
        # The intervals to approximate the integral over
        intervals = torch.linspace(1/num_iter, 1, num_iter, requires_grad=False).cuda().view(-1, 1, 1, 1)
        interval_masks = [up_mask.unsqueeze(1) * intervals for up_mask in up_masks]
        local_images = [phi(image.unsqueeze(1), base.unsqueeze(1), interval_mask) for image, base, interval_mask in zip(images, baseline, interval_masks)]

        if noise:
            local_images = [local_image + torch.randn_like(local_image) * .2 for local_image in local_images]

        local_images = [local_image.squeeze(0) for local_image in local_images]
        local_images = [[local_images[i][j] for i in range(len(local_images))] for j in range(num_iter)]
        input_ids = torch.cat((prompt, label), dim=1)
        positions = torch.tensor(positions).to(input_ids.device)
        losses = torch.tensor(0.).to(input_ids.device)

        for single_img in local_images:
            single_img = [item.unsqueeze(0).half() for item in single_img]
            probs = pred_probs(args, model, input_ids, label, single_img, image_size)
            #losses += probs[positions].mean()
            losses += torch.log(probs)[positions].sum()

    return losses / num_iter


def integrated_gradient(args, model, model_name, image, baseline, label, up_masks, num_iter,
                        noise=True, prompt=None, image_size=None, positions=None):
    loss = interval_score(
                args,
                model,
                model_name, 
                image,
                baseline,
                label,
                up_masks,
                num_iter,
                noise,
                prompt,
                image_size,
                positions
                )
    loss.sum().backward(retain_graph=True)
    return loss.sum().item()

def line_search(masks, total_grads, loss_func, alpha=8, beta=0.0001, decay=0.2,):
    # Speed up computations, reduce memory usage, and ensure no autograd
    # graphs are created
    with torch.no_grad():
        i = 0
        mod = len(masks.shape) - 3
        num_inputs = masks.shape[0]
        # The indices of masks that still need their alphas updated
        indices = torch.ones(num_inputs, dtype=torch.bool).cuda()
        # Create initial alpha values for each mask
        alphas = torch.ones(num_inputs).cuda() * alpha

        up_masks = upscale(masks.view(-1,*masks.shape[mod:])).view(-1, *masks.shape[1:mod], 1, upscale.out_size, upscale.out_size)

        # Compute the base loss used in the condition
        base_losses = loss_func(up_masks, masks, indices).view(-1)
        t = -beta * (total_grads ** 2).view(num_inputs, -1).sum(dim=1).view(num_inputs)

        while True:
            # Create a new mask with the updated alpha value to
            # see if it meets condition
            new_masks = torch.clamp(masks[indices] - alphas[indices].view(-1,*(1,) * mod,1,1) * total_grads[indices], 0, 1)
            up_masks = upscale(new_masks.view(-1,*masks.shape[mod:])).view(-1,*masks.shape[1:mod], 1, upscale.out_size, upscale.out_size)
            # Calculate new losses
            losses = loss_func(up_masks, new_masks, indices).view(-1)
            # Get indices for each alpha that meets the condition for
            # their corresponding mask
            indices[indices] = losses > base_losses[indices] + alphas[indices] * t[indices]
            # Same for this, but for if the alpha values are too low (\alpha_l)
            indices[indices] *= (alphas[indices] >= 0.00001)
            # Break out of the loop if all alpha values satisfy the condition
            # or are too low
            if not indices.sum():
                break
            # Otherwise update alphas
            alphas[indices] *= decay
            i += 1
    return alphas.view(-1,1,1,1)


def phi(img, baseline, mask):
    """
        Composes an image from img and baseline according to the mask values.

    :param img:
    :param baseline:
    :param mask:
    :return:
    """
    return img.mul(mask) + baseline.mul(1-mask)


def metric(args, image, baseline, mask, model, model_name, label, label_i, pred_data, size=28, prompt=None, image_size=None, positions=None, resolution=None):
    with torch.no_grad():
        # The dimensions for the image
        #img_size = image.shape[-1]
        # Compute the total number of pixels in a mask
        mask_pixels = torch.prod(torch.tensor(mask.shape[1:])).item()
        num_pixels = torch.prod(torch.tensor(mask.shape[1:])).item()
        # Compute the step size
        step=max(1, num_pixels // 50)
        # Used for indexing with batch sizes
        l = torch.arange(1)
        # The unmasked score
        og_scores = score_output(args, image, image_size, model, model_name, l, label, prompt, positions)
        # The baseline score
        blur_scores = score_output(args, baseline, image_size, model, model_name, l, label, prompt, positions)
        # Initial values for the curves
        del_curve = [og_scores]
        ins_curve = [blur_scores]
        index = [0.]

        # True_mask is used to hold 1 or 0. Either show that pixel or blur it.
        true_mask = torch.ones((mask.shape[0], mask_pixels)).cuda()
        del_scores = torch.zeros(mask.shape[0])
        ins_scores = torch.zeros(mask.shape[0])
        # Sort each mask by values and store the indices.
        elements = torch.argsort(mask.view(mask.shape[0], -1), dim=1)
        for pixels in range(0, num_pixels, step):
            # Get the indices used in this iteration
            indices = elements[l,pixels:pixels+step].squeeze().view(1, -1)
            # Set those indices to 0
            true_mask[l, indices.permute(1,0)] = 0
            up_mask = upscale(true_mask.view(-1, 1, size,size), image, resolution)
            # Mask the image for deletion
            if isinstance(image, list):
                del_image = [phi(x, y, z).half() for x, y, z in zip(image, baseline, up_mask)]
            else:
                del_image = phi(image, baseline, up_mask).half()
            # Calculate new scores
            outputs = score_output(args, del_image, image_size, model, model_name, l, label, prompt, positions)
            del_curve.append(outputs)
            index.append((pixels+step)/num_pixels)
            outputs = (outputs-blur_scores) / (og_scores-blur_scores)
            del_scores += outputs.cpu() * step if pixels + step < num_pixels else num_pixels - pixels

            # Mask the image for insertion
            if isinstance(image, list):
                ins_image = [phi(x, y, z).half() for x, y, z in zip(baseline, image, up_mask)]
            else:
                ins_image = phi(baseline, image, up_mask).half()

            # Calculate the new scores
            outputs = score_output(args, ins_image, image_size, model, model_name, l, label, prompt, positions)

            ins_curve.append(outputs)
            outputs = (outputs-blur_scores) / (og_scores-blur_scores)
            ins_scores += outputs.cpu() * step if pixels + step < num_pixels else num_pixels - pixels

        # Force scores between 0 and 1.
        del_scores /= num_pixels
        ins_scores /= num_pixels

        del_curve = list(map(lambda x: [y.item() for y in x], zip(*del_curve)))
        ins_curve = list(map(lambda x: [y.item() for y in x], zip(*ins_curve)))

    return del_scores, ins_scores, del_curve, ins_curve, index


def score_output(args, image, image_size, model, model_name, l, label, prompt, positions):
    input_ids = torch.cat((prompt, label), dim=1)
    probs_pred = pred_probs(args, model, input_ids, label, image, image_size).unsqueeze(0)
    scores = probs_pred[:, torch.tensor(positions).to(probs_pred.device)].sum(-1) 
    return scores / len(positions)

def pred_probs(model, inputs, generated_ids, image, target_token_position, selected_token_word_id, need_grad=False, return_log_probs=False):
    inputs_new = inputs.copy()
    
    inputs_new['input_ids'] = generated_ids
    inputs_new['attention_mask'] = torch.ones_like(generated_ids)
    
    inputs_new['pixel_values'] = image
    inputs_new = inputs_new.to(model.device)
    
    # [DEBUG] Check input image for NaN
    if torch.isnan(image).any():
        print(f"[NaN DEBUG] pred_probs: input image contains NaN!")
        print(f"  image shape: {image.shape}, NaN count: {torch.isnan(image).sum().item()}")
        # [FIX] Replace NaN with zeros to prevent propagation
        image = torch.nan_to_num(image, nan=0.0)
        inputs_new['pixel_values'] = image
    
    # Forward calculation to get all logits (including the logits of the input part)
    if need_grad:
        outputs = model(
            **inputs_new,
            return_dict=True,
            use_cache=True,
        )
        all_logits = outputs.logits  # [batch_size, seq_len, vocab_size]
    else:
        with torch.no_grad():
            outputs = model(
                **inputs_new,
                return_dict=True,
                use_cache=True,
            )
            all_logits = outputs.logits  # [batch_size, seq_len, vocab_size]
    
    # [DEBUG] Check model output logits for NaN
    if torch.isnan(all_logits).any():
        print(f"[NaN DEBUG] pred_probs: model output logits contain NaN!")
        print(f"  all_logits shape: {all_logits.shape}, NaN count: {torch.isnan(all_logits).sum().item()}")
        if not torch.isnan(all_logits).all():
            print(f"  logits range: [{all_logits[~torch.isnan(all_logits)].min().item():.4f}, {all_logits[~torch.isnan(all_logits)].max().item():.4f}]")
        # [FIX] Replace NaN with zeros to prevent propagation
        all_logits = torch.nan_to_num(all_logits, nan=0.0)
    
    returned_logits = all_logits[:, target_token_position - 1] # The reason for the minus 1 is that the generated content is in the previous position
    
    # [DEBUG] Check selected logits for NaN
    if torch.isnan(returned_logits).any():
        print(f"[NaN DEBUG] pred_probs: returned_logits (selected position) contain NaN!")
        print(f"  target_token_position: {target_token_position}")
    
    # [FIX] Use log_softmax for better numerical stability
    if return_log_probs:
        returned_logits = F.log_softmax(returned_logits, dim=-1)
    else:
        returned_logits = F.softmax(returned_logits, dim=-1)
    
    # [DEBUG] Check softmax output for NaN
    if torch.isnan(returned_logits).any():
        print(f"[NaN DEBUG] pred_probs: softmax/log_softmax output contains NaN!")
        print(f"  This usually means logits had extreme values (overflow/underflow)")
    
    selected_token_word_id = torch.tensor(selected_token_word_id).to(model.device)
    indices = selected_token_word_id.unsqueeze(0).unsqueeze(-1) # [1, N, 1]
    
    returned_logits = returned_logits.gather(dim=2, index=indices) # [1, N, 1]
    returned_logits = returned_logits.squeeze(-1)  # [1, N]
    
    # [DEBUG] Final check before return
    if torch.isnan(returned_logits).any():
        print(f"[NaN DEBUG] pred_probs: final returned_logits contain NaN!")
        print(f"  selected_token_word_id: {selected_token_word_id}")
    
    return returned_logits[0]

def save_keyword_score_distribution(output_ids, probs, probs_blur, key_word_threshold, tokenizer=None,
                                    save_dir="photoes", filename="keyword_score_distribution.png",
                                    selected_positions=None):
    """
    Save a bar chart for keyword decision scores of generated tokens.

    The score matches find_keywords: log(P(token | image)) - log(P(token | blurred image)).
    """
    os.makedirs(save_dir, exist_ok=True)

    probs_cpu = probs.detach().float().cpu()
    probs_blur_cpu = probs_blur.detach().float().cpu()
    eps = torch.finfo(probs_cpu.dtype).tiny
    scores = (
        torch.log(torch.clamp(probs_cpu, min=eps))
        - torch.log(torch.clamp(probs_blur_cpu, min=eps))
    ).numpy()

    if isinstance(output_ids, (list, tuple)):
        token_ids = output_ids[0]
    elif output_ids.dim() > 1:
        token_ids = output_ids[0]
    else:
        token_ids = output_ids
    token_ids = token_ids.detach().cpu().tolist()

    if tokenizer is None:
        token_labels = [str(token_id) for token_id in token_ids]
    else:
        token_labels = [
            tokenizer.decode([int(token_id)], skip_special_tokens=False).replace("\n", "\\n")
            for token_id in token_ids
        ]

    selected_positions = set(selected_positions or [])
    colors = [
        "tab:orange" if i in selected_positions or score > key_word_threshold else "tab:blue"
        for i, score in enumerate(scores)
    ]

    x = np.arange(len(scores))
    fig_width = max(10, min(0.45 * len(scores), 28))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    ax.bar(x, scores, color=colors)
    ax.axhline(
        key_word_threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"key_word_threshold={key_word_threshold:g}",
    )
    ax.set_xlabel("Generated token")
    ax.set_ylabel("Keyword score: log(p) - log(p_blur)")
    ax.set_title("Keyword Decision Score Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(token_labels, rotation=60, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()

    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path

def find_keywords(model, inputs, generated_ids, output_ids, image, blur_image, target_token_position, selected_token_word_id, tokenizer=None):
    # select keywords according to logits drop
    # full_prompt = torch.cat((input_ids, output_ids), dim=1)
    probs = pred_probs(model, inputs, generated_ids, image, target_token_position, selected_token_word_id)
    probs_blur = pred_probs(model, inputs, generated_ids, blur_image, target_token_position, selected_token_word_id)

    # [DEBUG] Check probs for NaN before log
    if torch.isnan(probs).any():
        print(f"[NaN DEBUG] find_keywords: probs contain NaN!")
    if torch.isnan(probs_blur).any():
        print(f"[NaN DEBUG] find_keywords: probs_blur contain NaN!")
    
    # [DEBUG] Check for zero values that would cause log(0) = -inf
    if (probs <= 0).any():
        print(f"[NaN DEBUG] find_keywords: probs contain zero or negative values, log will produce -inf")
        print(f"  zero count: {(probs <= 0).sum().item()}, min: {probs.min().item():.2e}")
    if (probs_blur <= 0).any():
        print(f"[NaN DEBUG] find_keywords: probs_blur contain zero or negative values, log will produce -inf")
        print(f"  zero count: {(probs_blur <= 0).sum().item()}, min: {probs_blur.min().item():.2e}")

    # condition = (probs_blur <= 0.4*probs) & (~torch.isin(output_ids[0], torch.tensor(special_ids).to(probs.device)))
    key_word_threshold = 4.0
    condition = (torch.log(probs)-torch.log(probs_blur) > key_word_threshold)& (probs>=0.0) & (~torch.isin(output_ids[0], torch.tensor(special_ids).to(probs.device)))
    positions = torch.where(condition)[0].tolist()
    save_keyword_score_distribution(output_ids, probs, probs_blur, key_word_threshold, tokenizer)
    
    if len(positions) == 0:
        idx = torch.argmax(probs-probs_blur).item()
        positions = [idx]
    
    keywords = [tokenizer.decode(output_ids[0][idx]).strip() for idx in positions]
    
    return positions, keywords



