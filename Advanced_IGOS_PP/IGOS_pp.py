import math
from torch.autograd import Variable
from .methods_helper import *
from .utils import *
from PIL import Image

from qwen_vl_utils import process_vision_info


# OpenAI CLIP 归一化常量
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

def pil_to_clip_tensor_bcwh(img, requires_grad=True, dtype=torch.float16, device=None):
    """
    将 PIL.Image 转为按 OpenAI CLIP 归一化的 tensor，维度为 [B, C, W, H]（B=1）。
    不进行 resize / crop；仅做 RGB 转换、[0,1] 归一化与标准化。
    
    Args:
        img: PIL.Image 或可被 PIL.Image.open 读取的路径
        requires_grad (bool): 返回的 tensor 是否需要梯度
        dtype: 返回 tensor 的数据类型（默认 float32）
        device: 返回 tensor 的设备（例如 "cuda" 或 torch.device(...)）

    Returns:
        x_bcwh: torch.Tensor, 形状 [1, 3, W, H]
    """
    # 1) 读图并确保 RGB
    if isinstance(img, (str, bytes, np.ndarray)):
        img = Image.open(img)
    img = img.convert("RGB")
    
    w, h = img.size
    new_w = round(w / 28) * 28
    new_h = round(h / 28) * 28
    
    img = img.resize((new_w, new_h), Image.BICUBIC)

    # 2) PIL -> numpy -> torch，归一到 [0,1]
    np_img = np.asarray(img, dtype=np.float32) / 255.0        # [H, W, 3]
    x = torch.from_numpy(np_img)                              # [H, W, 3]
    x = x.to(dtype=dtype)

    # 3) HWC -> CHW
    x = x.permute(2, 0, 1)                                    # [3, H, W]

    # 4) 按 CLIP 均值方差标准化
    mean = torch.tensor(OPENAI_CLIP_MEAN, dtype=dtype).view(3, 1, 1)
    std  = torch.tensor(OPENAI_CLIP_STD,  dtype=dtype).view(3, 1, 1)
    x = (x - mean) / std                                      # [3, H, W]

    # 5) 加 batch 维 -> [1, 3, H, W]
    x = x.unsqueeze(0)                                        # [1, 3, H, W]

    # 6) 设备与梯度设置
    if device is not None:
        x = x.to(device)
    x.requires_grad_(requires_grad)

    return x

def tensor2pack(patches: torch.Tensor) -> torch.Tensor:
    temporal_patch_size=2
    resized_height = patches.shape[2]
    resized_width = patches.shape[3]
    patch_size = 14
    merge_size = 2
    
    # 如果 B 不能整除 temporal_patch_size，就补齐
    if patches.shape[0] % temporal_patch_size != 0:
        repeats = patches[-1].unsqueeze(0).repeat(temporal_patch_size - 1, 1, 1, 1)
        patches = torch.cat([patches, repeats], dim=0)

    channel = patches.shape[1]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

    patches = patches.reshape(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )

    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()

    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size,
    )

    return flatten_patches

def gen_explanations_qwenvl(model, processor, image, text_prompt, tokenizer, positions=None, select_word_id=None):
    """_summary_

    Args:
        model (_type_): _description_
        processor (_type_): _description_
        image (_type_): PIL格式图片
        text_prompt (_type_): _description_
        device (_type_): _description_
    """
    # 调整图片尺寸，保持原始宽高比，最大边长不超过512
    max_size = 512
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    
    if original_width > original_height:
        new_width = min(max_size, original_width)
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = min(max_size, original_height)
        new_width = int(new_height * aspect_ratio)
    
    # 确保尺寸是28的倍数
    new_width = round(new_width / 28) * 28
    new_height = round(new_height / 28) * 28
    
    # 调整图片尺寸
    image = image.resize((new_width, new_height), Image.BICUBIC)
    
    input_size = (image.size[1], image.size[0])
    # size=32
    # opt = 'NAG'
    # diverse_k = 1
    # init_posi = 0
    # init_val = 0.0
    # L1 = 1.0
    # L2 = 0.1
    # gamma = 1.0
    # L3 = 10.0
    # momentum = 5
    # ig_iter = 10
    # iterations=10
    # lr=1.0  # 降低学习率以防止梯度爆炸

    size=48
    opt = 'NAG'
    diverse_k = 1
    init_posi = 0
    init_val = 0.5
    L1 = 0.5
    L2 = 60
    gamma = 0.5
    L3 = 30
    momentum = 5
    ig_iter = 20
    # 将 ig_iter 拆成多段依次 backward，显存峰值约按段数下降；须满足 ig_iter % ig_chunks == 0
    ig_chunks = 2
    iterations=25
    lr=0.01
    
    method = iGOS_pp
    
    i_obj = 0
    total_del, total_ins, total_time = 0, 0, 0
    all_del_scores = []
    all_ins_scores = []
    save_list = []
    
    # 开始处理数据
    image_size = [image.size]
    kernel_size = get_kernel_size(image.size)
    
    blur = cv2.GaussianBlur(np.asarray(image), (kernel_size, kernel_size), sigmaX=kernel_size-1)
    blur = Image.fromarray(blur.astype(np.uint8))
    
    # tensor
    messages1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    messages2 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": blur},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
            messages1, tokenize=False, add_generation_prompt=True)
    image_tensor, _ = process_vision_info(messages1)
    blur_tensor, _ = process_vision_info(messages2)
    
    inputs = processor(
            text=[text],
            images=image_tensor,    # 这里可以多个
            padding=True,
            return_tensors="pt",
        ).to(model.device)
    inputs_blur = processor(
            text=[text],
            images=blur_tensor,    # 这里可以多个
            padding=True,
            return_tensors="pt",
        ).to(model.device)
    
    image_tensor = inputs['pixel_values']
    blur_tensor = inputs_blur['pixel_values']
    
    input_ids = inputs['input_ids']
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            do_sample=False,      # Disable sampling and use greedy search instead
            num_beams=1,          # Set to 1 to ensure greedy search instead of beam search.
            max_new_tokens=128)
        generated_ids_trimmed = [   # 去掉图像和prompt的文本
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    selected_token_word_id = generated_ids_trimmed[0].cpu().numpy().tolist()
    selected_token_id = [i for i in range(len(selected_token_word_id))]
    target_token_position = np.array(selected_token_id) + len(inputs['input_ids'][0])
    
    if positions == None:
        positions, keywords = find_keywords(model, inputs, generated_ids, generated_ids_trimmed, image_tensor, blur_tensor, target_token_position, selected_token_word_id, tokenizer)
    else:
        keywords = processor.batch_decode(
            generated_ids_trimmed[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[positions[0]]
    
    print(f"[Model Raw Output] {output_text[0] if len(output_text) > 0 else ''}")
    print(f"[Selected Keywords] {keywords}")
    
    if select_word_id != None:
        for position, word_id in zip(positions, select_word_id):
            generated_ids_trimmed[0][position] = word_id
    
    pred_data=dict()
    pred_data['labels'] = generated_ids_trimmed
    pred_data['keywords'] = positions
    pred_data['boxes'] = np.array([[0, 0, input_size[0], input_size[1]]])
    pred_data['no_res'] = False
    pred_data['pred_text'] = output_text
    pred_data['keywords_text'] = keywords
    
    # calculate init area
    pred_data = get_initial(pred_data, k=diverse_k, init_posi=init_posi, 
                           init_val=init_val, input_size=input_size, out_size=size)
    
    for l_i, label in enumerate(pred_data['labels']):
        label = label.unsqueeze(0)
        keyword = pred_data['keywords']
        now = time.time()
        # masks, loss_del, loss_ins, loss_l1, loss_tv, loss_l2, loss_comb_del, loss_comb_ins = method(
        #         model=model,
        #         inputs = inputs, 
        #         generated_ids=generated_ids,
        #         init_mask=pred_data['init_masks'][0],
        #         image=pil_to_clip_tensor_bcwh(image).to(model.device),
        #         target_token_position=target_token_position, selected_token_word_id=selected_token_word_id,
        #         baseline=pil_to_clip_tensor_bcwh(blur).to(model.device),
        #         label=label,
        #         size=size,
        #         iterations=iterations,
        #         ig_iter=ig_iter,
        #         ig_chunks=ig_chunks,
        #         L1=L1,
        #         L2=L2,
        #         L3=L3,
        #         lr=lr,
        #         opt=opt,
        #         prompt=input_ids,
        #         image_size=image_size,
        #         positions=keyword,
        #         resolution=None,
        #         processor=tensor2pack
        #     )
        masks, loss_del, loss_ins, loss_l1, loss_tv, loss_l2, loss_comb_del, loss_comb_ins = method(
            model=model,
            inputs = inputs, 
            generated_ids=generated_ids,
            init_mask=pred_data['init_masks'][0],
            image=pil_to_clip_tensor_bcwh(image).to(model.device),
            target_token_position=target_token_position, selected_token_word_id=selected_token_word_id,
            baseline=pil_to_clip_tensor_bcwh(blur).to(model.device),
            label=label,
            size=size,
            opt=opt,
            prompt=input_ids,
            image_size=image_size,
            positions=keyword,
            resolution=None,
            processor=tensor2pack
        )
        
        total_time += time.time() - now
        
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
        masks = masks[0,0].detach().cpu().numpy()
        # 避免除以零导致的 NaN
        mask_min = np.min(masks)
        mask_max = np.max(masks)
        if mask_max > mask_min:
            masks -= mask_min
            masks /= (mask_max - mask_min)
        else:
            # 如果所有值都相同，设置为默认值
            masks = np.zeros_like(masks)
        
        image = np.array(image)
        masks = cv2.resize(masks, (image.shape[1], image.shape[0]))
        
        # 确保 masks 中没有 NaN 值
        masks = np.nan_to_num(masks, nan=0.0)
        
        # 增强mask对比度，让重要区域更突出
        vis_gamma = 0.4
        masks = np.power(masks, vis_gamma)
        
        # iGOS mask 在归一化后往往「显著区域数值小、背景大」；(1-masks) 使显著→高标量→JET 暖色；applyColorMap 为 BGR，与原图 BGR 对齐后再叠图
        heatmap = np.uint8(255 * np.clip(1.0 - masks, 0.0, 1.0))
        heatmap_bgr = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        original_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        superimposed_img = heatmap_bgr * 0.35 + original_bgr * 0.65
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        # cv2.imwrite("igos++.jpg", superimposed_img)
    
    # 释放不再使用的张量
    inputs = None
    inputs_blur = None
    image_tensor = None
    blur_tensor = None
    input_ids = None
    generated_ids = None
    generated_ids_trimmed = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return masks, superimposed_img

def gen_explanations_internvl(model, processor, image, text_prompt, tokenizer, positions=None, select_word_id=None):
    input_size = (image.size[1], image.size[0])
    size=32
    opt = 'NAG'
    diverse_k = 1
    init_posi = 0
    init_val = 0.
    L1 = 1.0
    L2 = 0.1
    gamma = 1.0
    L3 = 10.0
    momentum = 5
    ig_iter = 10
    ig_chunks = 1
    iterations=5
    lr=10
    
    method = iGOS_pp
    
    i_obj = 0
    total_del, total_ins, total_time = 0, 0, 0
    all_del_scores = []
    all_ins_scores = []
    save_list = []
    
    # 开始处理数据
    image_size = [image.size]
    kernel_size = get_kernel_size(image.size)
    
    # tensor
    messages1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    
    # Preparation for inference
    inputs = processor.apply_chat_template(messages1, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
    # inputs_blur = processor.apply_chat_template(messages1, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

    image_tensor = inputs['pixel_values']
    # blur_tensor = inputs_blur['pixel_values']
    blur_tensor = image_tensor * 0  # blur image cant choose salient word
    
    input_ids = inputs['input_ids']
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            do_sample=False,      # Disable sampling and use greedy search instead
            num_beams=1,          # Set to 1 to ensure greedy search instead of beam search.
            max_new_tokens=128)
        generated_ids_trimmed = [   # 去掉图像和prompt的文本
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    selected_token_word_id = generated_ids_trimmed[0].cpu().numpy().tolist()
    selected_token_id = [i for i in range(len(selected_token_word_id))]
    target_token_position = np.array(selected_token_id) + len(inputs['input_ids'][0])
    
    if positions == None:
        positions, keywords = find_keywords(model, inputs, generated_ids, generated_ids_trimmed, image_tensor, blur_tensor, target_token_position, selected_token_word_id, tokenizer)
    else:
        keywords = processor.batch_decode(
            generated_ids_trimmed[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[positions[0]]
    
    
    print(keywords)
    
    if select_word_id != None:
        for position, word_id in zip(positions, select_word_id):
            generated_ids_trimmed[0][position] = word_id
    
    pred_data=dict()
    pred_data['labels'] = generated_ids_trimmed
    pred_data['keywords'] = positions
    pred_data['boxes'] = np.array([[0, 0, input_size[0], input_size[1]]])
    pred_data['no_res'] = False
    pred_data['pred_text'] = output_text
    pred_data['keywords_text'] = keywords
    
    
    # new image
    messages2 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.resize((448, 448))},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(messages2, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
    input_ids = inputs["input_ids"]
    
    y = torch.stack(generated_ids_trimmed, dim=0)


    generated_ids = torch.cat([inputs["input_ids"], y if y.dim()==2 else y.unsqueeze(0)], dim=1).to(model.device)
    # inputs['attention_mask'] = torch.ones_like(inputs["input_ids"]).to(model.device)
    
    
    target_token_position = np.array(selected_token_id) + len(inputs['input_ids'][0])
    # calculate init area
    pred_data = get_initial(pred_data, k=diverse_k, init_posi=init_posi, 
                           init_val=init_val, input_size=input_size, out_size=size)
    for l_i, label in enumerate(pred_data['labels']):
        label = label.unsqueeze(0)
        keyword = pred_data['keywords']
        now = time.time()
        masks, loss_del, loss_ins, loss_l1, loss_tv, loss_l2, loss_comb_del, loss_comb_ins = method(
                model=model,
                inputs = inputs, 
                generated_ids=generated_ids,
                init_mask=pred_data['init_masks'][0],
                image=inputs['pixel_values'][-1].unsqueeze(0).to(model.device),
                target_token_position=target_token_position, selected_token_word_id=selected_token_word_id,
                baseline=inputs['pixel_values'][-1].unsqueeze(0).to(model.device)*0,
                label=label,
                size=size,
                iterations=iterations,
                ig_iter=ig_iter,
                ig_chunks=ig_chunks,
                L1=L1,
                L2=L2,
                L3=L3,
                lr=lr,
                opt=opt,
                prompt=input_ids,
                image_size=image_size,
                positions=keyword,
                resolution=None,
                processor=None
            )
        total_time += time.time() - now
        
        masks = masks[0,0].detach().cpu().numpy()
        masks -= np.min(masks)
        masks /= np.max(masks)
        
        image = np.array(image)
        masks = cv2.resize(masks, (image.shape[1], image.shape[0]))
        
        heatmap = np.uint8(255 * np.clip(1.0 - masks, 0.0, 1.0))
        heatmap_bgr = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        original_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        superimposed_img = heatmap_bgr * 0.4 + original_bgr
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        # cv2.imwrite("igos++.jpg", superimposed_img)
        
    return masks, superimposed_img

def interval_score(model, inputs, generated_ids, images, target_token_position, selected_token_word_id, baseline, up_masks, num_iter, noise=True, positions=None, processor=None, intervals=None):
    # if model_name == 'llava' or model_name == 'llava_next' or model_name == 'mgm':
    # The intervals to approximate the integral over (α = k/num_iter, k=1..num_iter).
    # If intervals is provided (e.g. IG chunked), it must be shape [chunk_len, 1, 1, 1];
    # loss is still normalized by full num_iter so chunk losses sum to the same IG estimate.
    if intervals is None:
        intervals = torch.linspace(1/num_iter, 1, num_iter, requires_grad=False).to(model.device).view(-1, 1, 1, 1)
    else:
        intervals = intervals.detach().to(device=model.device, dtype=up_masks.dtype)
        if intervals.dim() == 1:
            intervals = intervals.reshape(-1, 1, 1, 1)
    interval_masks = up_masks.unsqueeze(1) * intervals
    local_images = phi(images.unsqueeze(1), baseline.unsqueeze(1), interval_masks)

    # [DEBUG] Check up_masks for NaN
    if torch.isnan(up_masks).any():
        print(f"[NaN DEBUG] interval_score: up_masks contain NaN!")
        print(f"  up_masks shape: {up_masks.shape}, NaN count: {torch.isnan(up_masks).sum().item()}")
    
    # [DEBUG] Check local_images for NaN after phi
    if torch.isnan(local_images).any():
        print(f"[NaN DEBUG] interval_score: local_images contain NaN after phi()!")
        print(f"  local_images shape: {local_images.shape}, NaN count: {torch.isnan(local_images).sum().item()}")

    if noise:
        local_images = local_images + torch.randn_like(local_images) * .2

    local_images = local_images.transpose(0, 1)
    # input_ids = torch.cat((prompt, label), dim=1)
    positions = torch.tensor(positions).to(model.device)

    losses = torch.tensor(0.).to(model.device)
    for idx, single_img in enumerate(local_images):
        if processor == None:
            single_input = single_img
        else:
            single_input = processor(single_img)
        
        # [DEBUG] Check single_input for NaN
        if torch.isnan(single_input).any():
            print(f"[NaN DEBUG] interval_score: single_input (iter {idx}) contains NaN after processor!")
            # [FIX] Replace NaN with zeros
            single_input = torch.nan_to_num(single_input, nan=0.0)
        
        # [FIX] Use return_log_probs=True for better numerical stability
        log_probs = pred_probs(model, inputs, generated_ids, single_input, target_token_position, selected_token_word_id, need_grad=True, return_log_probs=True)
        
        # [DEBUG] Check log_probs for NaN
        if torch.isnan(log_probs).any():
            print(f"[NaN DEBUG] interval_score: log_probs (iter {idx}) contain NaN!")
            print(f"  positions: {positions}")
            # [FIX] Replace NaN with a large negative value (equivalent to very small prob)
            log_probs = torch.nan_to_num(log_probs, nan=-20.0)
        
        # [FIX] Clamp log_probs to prevent extreme values
        log_probs = torch.clamp(log_probs, min=-20.0, max=0.0)
        
        losses += log_probs[positions].sum()
        
        # [DEBUG] Check losses for NaN
        if torch.isnan(losses):
            print(f"[NaN DEBUG] interval_score: losses became NaN at iteration {idx}!")
            print(f"  positions: {positions}, positions valid range: [0, {log_probs.shape[0]-1}]")
            break

    final_loss = losses / num_iter
    
    # [DEBUG] Check final loss for NaN
    if torch.isnan(final_loss):
        print(f"[NaN DEBUG] interval_score: final_loss is NaN!")
        print(f"  num_iter: {num_iter}")
    
    return final_loss


def integrated_gradient(model, inputs, generated_ids, image, target_token_position, selected_token_word_id, baseline, up_masks, num_iter, noise=True, positions=None, processor=None, ig_chunks=1):
    """Integrated-gradient loss; optional ig_chunks splits Riemann steps to save VRAM (gradients accumulate)."""
    if ig_chunks is None or ig_chunks <= 1:
        loss = interval_score(
            model,
            inputs,
            generated_ids,
            image,
            target_token_position,
            selected_token_word_id,
            baseline,
            up_masks,
            num_iter,
            noise=noise,
            positions=positions,
            processor=processor)
        if torch.isnan(loss):
            print(f"[NaN DEBUG] integrated_gradient: loss is NaN before backward!")
            print(f"  up_masks range: [{up_masks.min().item():.4f}, {up_masks.max().item():.4f}]")
            print(f"  num_iter: {num_iter}, positions: {positions}")
            return float('nan')
        loss.sum().backward()
        if up_masks.grad is not None and torch.isnan(up_masks.grad).any():
            print(f"[NaN DEBUG] integrated_gradient: up_masks.grad contains NaN after backward!")
            print(f"  grad NaN count: {torch.isnan(up_masks.grad).sum().item()}")
        loss_value = loss.sum().item()
        if math.isnan(loss_value):
            print(f"[NaN DEBUG] integrated_gradient: final loss_value is NaN!")
        return loss_value

    if num_iter % ig_chunks != 0:
        raise ValueError(f"ig_iter ({num_iter}) must be divisible by ig_chunks ({ig_chunks})")
    chunk_size = num_iter // ig_chunks
    total_loss = 0.0
    dtype = up_masks.dtype if up_masks.dtype.is_floating_point else torch.float32
    # 与上游 upscale 解耦：各 chunk 的 backward 只作用在 detached 叶子上，避免多次 backward 释放 upscale 图后二次穿越
    up_masks_detached = up_masks.detach().requires_grad_(True)
    for b in range(ig_chunks):
        start_k = b * chunk_size
        intervals = torch.linspace(
            (start_k + 1) / num_iter,
            (start_k + chunk_size) / num_iter,
            chunk_size,
            requires_grad=False,
            device=model.device,
            dtype=dtype,
        ).view(-1, 1, 1, 1)
        loss = interval_score(
            model,
            inputs,
            generated_ids,
            image,
            target_token_position,
            selected_token_word_id,
            baseline,
            up_masks_detached,
            num_iter,
            noise=noise,
            positions=positions,
            processor=processor,
            intervals=intervals)
        if torch.isnan(loss):
            print(f"[NaN DEBUG] integrated_gradient (chunk {b}/{ig_chunks}): loss is NaN before backward!")
            print(f"  up_masks range: [{up_masks.min().item():.4f}, {up_masks.max().item():.4f}]")
            print(f"  num_iter: {num_iter}, positions: {positions}")
            return float('nan')
        loss.sum().backward()
        total_loss += loss.sum().item()
    if up_masks_detached.grad is not None and torch.isnan(up_masks_detached.grad).any():
        print(f"[NaN DEBUG] integrated_gradient: up_masks_detached.grad contains NaN after chunked backward!")
        print(f"  grad NaN count: {torch.isnan(up_masks_detached.grad).sum().item()}")
    if up_masks_detached.grad is not None:
        up_masks.backward(up_masks_detached.grad)
    if math.isnan(total_loss):
        print(f"[NaN DEBUG] integrated_gradient: final total_loss is NaN!")
    return total_loss

def iGOS_pp(
        model,
        inputs, 
        generated_ids,
        init_mask,
        image,
        target_token_position, 
        selected_token_word_id,
        baseline,
        label,
        size=32,
        iterations=15,
        ig_iter=20,
        ig_chunks=2,
        L1=1,
        L2=1,
        L3=20,
        lr=1000,
        opt='LS',
        softmax=True,
        processor=None,
        **kwargs):

    L2 = 0.1
    gamma = 1.0
    momentum = 5
    
    def regularization_loss(image, masks):
        return L1 * torch.mean(torch.abs(1 - masks).view(masks.shape[0], -1), dim=1), \
               L3 * bilateral_tv_norm(image, masks, tv_beta=2, sigma=0.01), \
               L2 * torch.sum((1 - masks)**2, dim=[1, 2, 3])

    def ins_loss_function(up_masks, indices, noise=True):
        losses = -interval_score(
            model, 
            inputs, 
            generated_ids, 
            baseline[indices], 
            target_token_position, 
            selected_token_word_id, 
            image[indices], 
            up_masks, 
            ig_iter, 
            noise, 
            positions)
        
        return losses.sum(dim=1).view(-1)

    def del_loss_function(up_masks, indices, noise=True):
        losses = interval_score(
            model, 
            inputs, 
            generated_ids, 
            baseline[indices], 
            target_token_position, 
            selected_token_word_id, 
            image[indices], 
            up_masks, 
            ig_iter, 
            noise, 
            positions)
        return losses.sum(dim=1).view(-1)

    def loss_function(up_masks, masks, indices):
        loss = del_loss_function(up_masks[:, 0], indices)
        loss += ins_loss_function(up_masks[:, 1], indices)
        loss += del_loss_function(up_masks[:, 0] * up_masks[:, 1], indices)
        loss += ins_loss_function(up_masks[:, 0] * up_masks[:, 1], indices)
        return loss + regularization_loss(image[indices], masks[:, 0] * masks[:, 1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    masks_del = torch.ones((1, 1, size, size), dtype=torch.float32, device=device)
    masks_del = masks_del * init_mask.to(device)
    masks_del = Variable(masks_del, requires_grad=True)
    masks_ins = torch.ones((image.shape[0], 1, size, size), dtype=torch.float32, device=device)
    masks_ins = masks_ins * init_mask.to(device)
    masks_ins = Variable(masks_ins, requires_grad=True)
    prompt = kwargs.get('prompt', None)
    image_size = kwargs.get('image_size', None)
    positions = kwargs.get('positions', None)
    resolution = kwargs.get('resolution', None)

    
    if opt == 'NAG':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cita_d=torch.zeros(1).to(device)
        cita_i=torch.zeros(1).to(device)
    
    prompt = kwargs.get('prompt', None)
    image_size = kwargs.get('image_size', None)
    positions = kwargs.get('positions', None)
    losses_del, losses_ins, losses_l1, losses_tv, losses_l2, losses_comb_del, losses_comb_ins = [], [], [], [], [], [], []
    for i in range(iterations):
        up_masks1 = upscale(masks_del, image)
        up_masks2 = upscale(masks_ins, image)

        # [DEBUG] Check up_masks for NaN before integrated_gradient
        if torch.isnan(up_masks1).any():
            print(f"[NaN DEBUG] iGOS_pp iter {i}: up_masks1 contains NaN!")
        if torch.isnan(up_masks2).any():
            print(f"[NaN DEBUG] iGOS_pp iter {i}: up_masks2 contains NaN!")
        
        combined_mask = up_masks1 * up_masks2
        if torch.isnan(combined_mask).any():
            print(f"[NaN DEBUG] iGOS_pp iter {i}: combined_mask (up_masks1 * up_masks2) contains NaN!")

        # Compute the integrated gradient for the combined mask, optimized for deletion
        loss_comb_del = integrated_gradient(model, inputs, generated_ids, image, target_token_position, selected_token_word_id, baseline, up_masks1 * up_masks2, ig_iter, positions=positions, processor=processor, ig_chunks=ig_chunks)
        
        # [DEBUG] Check loss_comb_del for NaN
        if math.isnan(loss_comb_del):
            print(f"[NaN DEBUG] iGOS_pp iter {i}: loss_comb_del is NaN!")
            print(f"  This is the main loss causing the issue.")
            print(f"  masks_del range: [{masks_del.min().item():.4f}, {masks_del.max().item():.4f}]")
            print(f"  masks_ins range: [{masks_ins.min().item():.4f}, {masks_ins.max().item():.4f}]")
        
        # 确保梯度存在
        if masks_del.grad is not None and masks_ins.grad is not None:
            total_grads1 = masks_del.grad.clone()
            total_grads2 = masks_ins.grad.clone()
            masks_del.grad.zero_()
            masks_ins.grad.zero_()
        else:
            # 如果梯度不存在，使用零梯度
            total_grads1 = torch.zeros_like(masks_del)
            total_grads2 = torch.zeros_like(masks_ins)

        # Compute the integrated gradient for the combined mask, optimized for insertion
        loss_comb_ins = integrated_gradient(model, inputs, generated_ids, image, target_token_position, selected_token_word_id, baseline, up_masks1 * up_masks2, ig_iter, positions=positions, processor=processor, ig_chunks=ig_chunks)
        
        # 确保梯度存在
        if masks_del.grad is not None and masks_ins.grad is not None:
            total_grads1 -= masks_del.grad.clone()  # Negative because insertion loss is 1 - score.
            total_grads2 -= masks_ins.grad.clone()
            masks_del.grad.zero_()
            masks_ins.grad.zero_()

        # Compute the integrated gradient for the deletion mask
        loss_del = integrated_gradient(model, inputs, generated_ids, image, target_token_position, selected_token_word_id, baseline, up_masks1, ig_iter, positions=positions, processor=processor, ig_chunks=ig_chunks)
        
        # 确保梯度存在
        if masks_del.grad is not None:
            total_grads1 += masks_del.grad.clone()
            masks_del.grad.zero_()

        # Compute the integrated graident for the insertion mask
        loss_ins = integrated_gradient(model, inputs, generated_ids, image, target_token_position, selected_token_word_id, baseline, up_masks2, ig_iter, positions=positions, processor=processor, ig_chunks=ig_chunks)
        
        # 确保梯度存在
        if masks_ins.grad is not None:
            total_grads2 -= masks_ins.grad.clone()
            masks_ins.grad.zero_()

        # Average them to balance out the terms with the regularization terms
        total_grads1 /= 2
        total_grads2 /= 2
        
        # [DEBUG] Check total_grads for NaN before regularization
        if torch.isnan(total_grads1).any():
            print(f"[NaN DEBUG] iGOS_pp iter {i}: total_grads1 contains NaN before reg!")
            print(f"  grad norm: {total_grads1.norm().item():.4e}")
        if torch.isnan(total_grads2).any():
            print(f"[NaN DEBUG] iGOS_pp iter {i}: total_grads2 contains NaN before reg!")
            print(f"  grad norm: {total_grads2.norm().item():.4e}")
        
        # print(f"[iGOS_pp] iter {i}: total_grads1 norm={total_grads1.norm().item():.4e}, total_grads2 norm={total_grads2.norm().item():.4e}")

        # Computer regularization for combined masks
        L2 = exp_decay(L2, i, gamma)
        loss_l1, loss_tv, loss_l2 = regularization_loss(image, masks_del * masks_ins)
        
        # [DEBUG] Check regularization losses for NaN
        if torch.isnan(loss_l1).any() or torch.isnan(loss_tv).any() or torch.isnan(loss_l2).any():
            print(f"[NaN DEBUG] iGOS_pp iter {i}: regularization loss contains NaN!")
            print(f"  loss_l1: {loss_l1}, loss_tv: {loss_tv}, loss_l2: {loss_l2}")
        
        losses = loss_l1 + loss_tv + loss_l2
        
        # [DEBUG] Check combined loss before backward
        if torch.isnan(losses).any():
            print(f"[NaN DEBUG] iGOS_pp iter {i}: combined loss (loss_l1+loss_tv+loss_l2) contains NaN before backward!")
            print(f"  masks_del*masks_ins range: [{(masks_del * masks_ins).min().item():.4f}, {(masks_del * masks_ins).max().item():.4f}]")
        
        losses.sum().backward()
        
        # [DEBUG] Check gradient from regularization for NaN
        if masks_del.grad is not None and torch.isnan(masks_del.grad).any():
            print(f"[NaN DEBUG] iGOS_pp iter {i}: masks_del.grad contains NaN after reg backward!")
        if masks_ins.grad is not None and torch.isnan(masks_ins.grad).any():
            print(f"[NaN DEBUG] iGOS_pp iter {i}: masks_ins.grad contains NaN after reg backward!")
        
        # if masks_del.grad is not None:
        #     print(f"[iGOS_pp] iter {i}: reg grad norm del={masks_del.grad.norm().item():.4e}, ins={masks_ins.grad.norm().item():.4e}")
        
        total_grads1 += masks_del.grad.clone()
        total_grads2 += masks_ins.grad.clone()
        
        # [DEBUG] Check total_grads after adding regularization gradient
        if torch.isnan(total_grads1).any():
            print(f"[NaN DEBUG] iGOS_pp iter {i}: total_grads1 contains NaN after adding reg grad!")
        if torch.isnan(total_grads2).any():
            print(f"[NaN DEBUG] iGOS_pp iter {i}: total_grads2 contains NaN after adding reg grad!")

        if opt == 'LS':
            masks = torch.cat((masks_del.unsqueeze(1), masks_ins.unsqueeze(1)), 1)
            total_grads = torch.cat((total_grads1.unsqueeze(1), total_grads2.unsqueeze(1)), 1)
            lrs = line_search(masks, total_grads, loss_function, lr)
            
            torch.nn.utils.clip_grad_norm_([total_grads1, total_grads2], max_norm=0.1)
            masks_del.data -= total_grads1 * lrs
            masks_ins.data -= total_grads2 * lrs
        
        if opt == 'NAG':
            # [DEBUG] Check total_grads before clip
            if torch.isnan(total_grads1).any():
                print(f"[NaN DEBUG] iGOS_pp iter {i}: total_grads1 contains NaN before clip!")
            if torch.isnan(total_grads2).any():
                print(f"[NaN DEBUG] iGOS_pp iter {i}: total_grads2 contains NaN before clip!")
            
            torch.nn.utils.clip_grad_norm_([total_grads1, total_grads2], max_norm=0.1)
            
            # [DEBUG] Check total_grads after clip (clip may not fix NaN)
            if torch.isnan(total_grads1).any():
                print(f"[NaN DEBUG] iGOS_pp iter {i}: total_grads1 STILL contains NaN after clip!")
                print(f"  This is likely the ROOT CAUSE - gradients became NaN and clip cannot fix it")
            if torch.isnan(total_grads2).any():
                print(f"[NaN DEBUG] iGOS_pp iter {i}: total_grads2 STILL contains NaN after clip!")
            
            e = i / (i + momentum)
            cita_d_p = cita_d
            cita_i_p = cita_i
            
            # [DEBUG] Check cita values before update
            if i > 0 and torch.isnan(cita_d_p).any():
                print(f"[NaN DEBUG] iGOS_pp iter {i}: cita_d_p (from prev iter) contains NaN!")
            if i > 0 and torch.isnan(cita_i_p).any():
                print(f"[NaN DEBUG] iGOS_pp iter {i}: cita_i_p (from prev iter) contains NaN!")
            
            cita_d = masks_del.data - lr * total_grads1
            cita_i = masks_ins.data - lr * total_grads2
            
            # [DEBUG] Check cita after gradient step
            if torch.isnan(cita_d).any():
                print(f"[NaN DEBUG] iGOS_pp iter {i}: cita_d contains NaN after gradient step!")
                print(f"  masks_del.data range: [{masks_del.data.min().item():.4f}, {masks_del.data.max().item():.4f}]")
                print(f"  lr: {lr}, total_grads1 range: [{total_grads1.min().item():.4e}, {total_grads1.max().item():.4e}]")
            if torch.isnan(cita_i).any():
                print(f"[NaN DEBUG] iGOS_pp iter {i}: cita_i contains NaN after gradient step!")
            
            momentum_term_d = cita_d - cita_d_p
            momentum_term_i = cita_i - cita_i_p
            
            # [DEBUG] Check momentum term
            if torch.isnan(momentum_term_d).any():
                print(f"[NaN DEBUG] iGOS_pp iter {i}: momentum_term_d (cita_d - cita_d_p) contains NaN!")
            if torch.isnan(momentum_term_i).any():
                print(f"[NaN DEBUG] iGOS_pp iter {i}: momentum_term_i (cita_i - cita_i_p) contains NaN!")
            
            masks_del.data = cita_d + e * momentum_term_d
            masks_ins.data = cita_i + e * momentum_term_i
            
            # [DEBUG] Final check after NAG update
            if torch.isnan(masks_del.data).any():
                print(f"[NaN DEBUG] iGOS_pp iter {i}: masks_del.data contains NaN after NAG update!")
            if torch.isnan(masks_ins.data).any():
                print(f"[NaN DEBUG] iGOS_pp iter {i}: masks_ins.data contains NaN after NAG update!")

        masks_del.grad.zero_()
        masks_ins.grad.zero_()
        masks_del.data.clamp_(0,1)
        masks_ins.data.clamp_(0,1)

        losses_del.append(loss_del)
        losses_ins.append(loss_ins)
        losses_comb_del.append(loss_comb_del)
        losses_comb_ins.append(loss_comb_ins)
        losses_l1.append(loss_l1.item())
        losses_tv.append(loss_tv.item())
        losses_l2.append(loss_l2.item())
        print(f'iteration: {i} lr: {lr:.4f} loss_comb_del: {loss_comb_del:.4f}, loss_comb_ins: {loss_comb_ins:.4f}, loss_del: {loss_del:.4f}, loss_ins: {loss_ins:.4f}, loss_l1: {loss_l1.item():.4f}, loss_tv: {loss_tv.item():.4f}, loss_l2: {loss_l2.item():.4f}')

    return masks_del * masks_ins, losses_del, losses_ins, losses_l1, losses_tv, losses_l2, losses_comb_del, losses_comb_ins


