

from PIL import Image
import io
import os
import numpy as np
import torch
from collections import defaultdict


def _load_matting_array(path):
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, :3].mean(axis=2)
    return arr


def _resize_matting_array(arr, height, width, resample=Image.BILINEAR):
    if arr.shape[:2] == (height, width):
        return arr
    image = Image.fromarray(arr.astype(np.float32))
    return np.array(image.resize((width, height), resample))


def _gauss(x, sigma):
    return np.exp(-(x**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def _dgauss(x, sigma):
    return -x * _gauss(x, sigma) / (sigma**2)


def _gaussgradient(im, sigma):
    import scipy.ndimage

    epsilon = 1e-2
    halfsize = np.ceil(
        sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))
    ).astype(np.int_)
    size = 2 * halfsize + 1
    hx = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            u = [i - halfsize, j - halfsize]
            hx[i, j] = _gauss(u[0], sigma) * _dgauss(u[1], sigma)

    hx = hx / np.sqrt(np.sum(np.abs(hx) * np.abs(hx)))
    hy = hx.transpose()
    return (
        scipy.ndimage.convolve(im, hx, mode="nearest"),
        scipy.ndimage.convolve(im, hy, mode="nearest"),
    )


def _largest_cc(segmentation):
    from skimage.measure import label

    labels = label(segmentation, connectivity=1)
    counts = np.bincount(labels.flat)
    if len(counts) <= 1:
        return np.zeros_like(segmentation, dtype=bool)
    return labels == np.argmax(counts)


def _connectivity_error(pred, target, trimap=None, step=0.1):
    h, w = pred.shape
    thresh_steps = list(np.arange(0, 1 + step, step))
    l_map = np.ones((h, w), dtype=np.float32) * -1
    for i in range(1, len(thresh_steps)):
        pred_alpha_thresh = (pred >= thresh_steps[i]).astype(np.int_)
        target_alpha_thresh = (target >= thresh_steps[i]).astype(np.int_)
        omega = _largest_cc(pred_alpha_thresh * target_alpha_thresh).astype(np.int_)
        flag = ((l_map == -1) & (omega == 0)).astype(np.int_)
        l_map[flag == 1] = thresh_steps[i - 1]

    l_map[l_map == -1] = 1
    pred_d = pred - l_map
    target_d = target - l_map
    pred_phi = 1 - pred_d * (pred_d >= 0.15).astype(np.int_)
    target_phi = 1 - target_d * (target_d >= 0.15).astype(np.int_)
    error = np.abs(pred_phi - target_phi)
    if trimap is not None:
        error = error[trimap == 128]
    return np.sum(error) / 1000.0


def _matting_metrics(pred, alpha, trimap):
    pred = np.nan_to_num(pred.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    alpha = np.nan_to_num(alpha.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if pred.max() > 1.0:
        pred = pred / 255.0
    if alpha.max() > 1.0:
        alpha = alpha / 255.0
    pred = np.clip(pred, 0.0, 1.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    if trimap is not None:
        if trimap.max() <= 1:
            trimap = (trimap * 255).round()
        trimap = trimap.astype(np.uint8)
        unknown = trimap == 128
        pred_eval = pred.copy()
        pred_eval[trimap == 255] = 1.0
        pred_eval[trimap == 0] = 0.0
    else:
        unknown = np.ones_like(pred, dtype=bool)
        pred_eval = pred

    pixel = float(unknown.sum())
    if pixel <= 0:
        pixel = float(pred.size)
        unknown = np.ones_like(pred, dtype=bool)

    abs_diff = np.abs(pred_eval - alpha)
    sq_diff = (pred_eval - alpha) ** 2
    mse = float(np.sum(sq_diff[unknown]) / pixel)
    mad = float(np.sum(abs_diff[unknown]) / pixel)
    sad = float(np.sum(abs_diff) / 1000.0)

    pred_x, pred_y = _gaussgradient(pred, 1.4)
    alpha_x, alpha_y = _gaussgradient(alpha, 1.4)
    grad_map = (np.sqrt(pred_x**2 + pred_y**2) - np.sqrt(alpha_x**2 + alpha_y**2)) ** 2
    grad = float(np.sum(grad_map[unknown]) / 1000.0)
    conn = float(_connectivity_error(pred, alpha, trimap))
    return mse, mad, sad, grad, conn, pixel


def _matting_metrics_e2p_aligned(pred, alpha, trimap):
    pred = np.nan_to_num(pred.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    alpha = np.nan_to_num(alpha.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if pred.max() > 1.0:
        pred = pred / 255.0
    if alpha.max() > 1.0:
        alpha = alpha / 255.0
    pred = np.clip(pred, 0.0, 1.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    if trimap is not None:
        if trimap.max() <= 1:
            trimap = (trimap * 255).round()
        trimap = trimap.astype(np.uint8)
        unknown = trimap == 128
        known = ~unknown

        pred_unknown_eval = pred.copy()
        pred_unknown_eval[trimap == 255] = 1.0
        pred_unknown_eval[trimap == 0] = 0.0
    else:
        unknown = np.ones_like(pred, dtype=bool)
        known = np.zeros_like(pred, dtype=bool)
        pred_unknown_eval = pred

    unknown_pixels = float(unknown.sum())
    if unknown_pixels <= 0:
        unknown = np.ones_like(pred, dtype=bool)
        unknown_pixels = float(pred.size)
    known_pixels = float(known.sum())

    abs_whole = np.abs(pred - alpha)
    sq_whole = (pred - alpha) ** 2
    abs_unknown = np.abs(pred_unknown_eval - alpha)
    sq_unknown = (pred_unknown_eval - alpha) ** 2

    pred_x, pred_y = _gaussgradient(pred, 1.4)
    alpha_x, alpha_y = _gaussgradient(alpha, 1.4)
    grad_map = (np.sqrt(pred_x**2 + pred_y**2) - np.sqrt(alpha_x**2 + alpha_y**2)) ** 2

    metrics = {
        # Match E2P eval_matting.py -> compute_matting_metrics(..., whole=True).
        "whole_mse": float(np.sum(sq_whole) / pred.size),
        "whole_mad": float(np.sum(abs_whole) / pred.size),
        "whole_sad": float(np.sum(abs_whole) / 1000.0),
        "whole_grad": float(np.sum(grad_map) / 1000.0),
        "whole_conn": float(_connectivity_error(pred, alpha, None)),
        # Keep trimap-sensitive boundary pressure as auxiliary signal.
        "unknown_mse": float(np.sum(sq_unknown[unknown]) / unknown_pixels),
        "unknown_mad": float(np.sum(abs_unknown[unknown]) / unknown_pixels),
        "unknown_grad": float(np.sum(grad_map[unknown]) / 1000.0),
        "unknown_conn": float(_connectivity_error(pred, alpha, trimap)),
        "unknown_pixels": unknown_pixels,
    }
    metrics["known_l1"] = (
        float(np.sum(abs_whole[known]) / known_pixels) if known_pixels > 0 else 0.0
    )
    return metrics


def matting_score(device):
    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = images.detach().float().cpu().clamp(0, 1).numpy()
            images = images.transpose(0, 2, 3, 1)

        scores = []
        details = defaultdict(list)
        # SAD-oriented reward for leaderboard tuning. Keep a few auxiliary
        # terms only as guardrails so SAD remains the dominant optimization
        # signal near the current SOTA regime.
        weights = {
            "whole_sad": 1.00,
            "whole_mad": 0.25,
            "whole_grad": 0.15,
            "whole_conn": 0.10,
            "known_l1": 0.20,
        }
        scales = {
            "whole_sad": 15.0,
            "whole_mad": 0.025,
            "whole_grad": 20.0,
            "whole_conn": 15.0,
            "known_l1": 0.015,
        }

        for image, meta in zip(images, metadata):
            alpha_path = meta.get("alpha") or meta.get("alpha_path") or meta.get("gt")
            trimap_path = meta.get("trimap") or meta.get("trimap_path")
            dataset_root = meta.get("__dataset_root", "")
            if not alpha_path:
                scores.append(-10.0)
                continue
            if not os.path.isabs(alpha_path):
                alpha_path = os.path.join(dataset_root, alpha_path)
            if trimap_path and not os.path.isabs(trimap_path):
                trimap_path = os.path.join(dataset_root, trimap_path)

            pred = image.mean(axis=2)
            alpha = _load_matting_array(alpha_path)
            alpha = _resize_matting_array(alpha, pred.shape[0], pred.shape[1])
            trimap = None
            if trimap_path:
                trimap = _load_matting_array(trimap_path)
                trimap = _resize_matting_array(trimap, pred.shape[0], pred.shape[1], Image.NEAREST)

            metric_values = _matting_metrics_e2p_aligned(pred, alpha, trimap)
            normalized_error = sum(
                weights[name] * min(metric_values[name] / scales[name], 10.0)
                for name in weights
            )
            reward = float(-normalized_error)
            scores.append(reward)

            for name, value in metric_values.items():
                if name == "unknown_pixels":
                    continue
                details[f"matting_{name}"].append(float(value))
                if name in weights:
                    details[f"matting_{name}_term"].append(
                        float(weights[name] * min(value / scales[name], 10.0))
                    )
            # Backward-compatible aliases used by train_flux_kontext.py summaries.
            details["matting_mse"].append(float(metric_values["whole_mse"]))
            details["matting_mad"].append(float(metric_values["whole_mad"]))
            details["matting_sad"].append(float(metric_values["whole_sad"]))
            details["matting_grad"].append(float(metric_values["whole_grad"]))
            details["matting_conn"].append(float(metric_values["whole_conn"]))
            details["matting_unknown_pixels"].append(float(metric_values["unknown_pixels"]))
            details["matting_error"].append(float(normalized_error))

        return scores, dict(details)

    return _fn

def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew/500, meta

    return _fn

def aesthetic_score():
    from flow_grpo.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn

def clip_score(device):
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def image_similarity_score(device):
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device).cuda()

    def _fn(images, ref_images):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        if not isinstance(ref_images, torch.Tensor):
            ref_images = [np.array(img) for img in ref_images]
            ref_images = np.array(ref_images)
            ref_images = ref_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            ref_images = torch.tensor(ref_images, dtype=torch.uint8)/255.0
        scores = scorer.image_similarity(images, ref_images)
        return scores, {}

    return _fn

def pickscore_score(device):
    from flow_grpo.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def imagereward_score(device):
    from flow_grpo.imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def qwenvl_score(device):
    from flow_grpo.qwenvl import QwenVLScorer

    scorer = QwenVLScorer(dtype=torch.bfloat16, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

    
def ocr_score(device):
    from flow_grpo.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def video_ocr_score(device):
    from flow_grpo.ocr import OcrScorer_video_or_image

    scorer = OcrScorer_video_or_image()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1) 
            elif images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def deqa_score_remote(device):
    """Submits images to DeQA and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://127.0.0.1:18086"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        all_scores = []
        for image_batch in images_batched:
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn

def geneval_score(device):
    """Submits images to GenEval and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://127.0.0.1:18085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadatas, only_strict):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "meta_datas": list(metadata_batched),
                "only_strict": only_strict,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["scores"]
            all_rewards += response_data["rewards"]
            all_strict_rewards += response_data["strict_rewards"]
            all_group_strict_rewards.append(response_data["group_strict_rewards"])
            all_group_rewards.append(response_data["group_rewards"])
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict

    return _fn

def unifiedreward_score_remote(device):
    """Submits images to DeQA and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://10.82.120.15:18085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "prompts": prompt_batch
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            print("response: ", response)
            print("response: ", response.content)
            response_data = pickle.loads(response.content)

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn

def unifiedreward_score_sglang(device):
    import asyncio
    from openai import AsyncOpenAI
    import base64
    from io import BytesIO
    import re 

    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(text_outputs):
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    client = AsyncOpenAI(base_url="http://127.0.0.1:17140/v1", api_key="flowgrpo")
        
    async def evaluate_image(prompt, image):
        question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after \'Final Score:\'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
        images_base64 = pil_image_to_base64(image)
        response = await client.chat.completions.create(
            model="UnifiedReward-7b-v1.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": images_base64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    async def evaluate_batch_image(images, prompts):
        tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, metadata):
        # 处理Tensor类型转换
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        # 转换为PIL Image并调整尺寸
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        # 执行异步批量评估
        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc/5.0 for sc in score]
        return score, {}
    
    return _fn

def multi_score(device, score_dict):
    score_functions = {
        "deqa": deqa_score_remote,
        "ocr": ocr_score,
        "video_ocr": video_ocr_score,
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "qwenvl": qwenvl_score,
        "aesthetic": aesthetic_score,
        "jpeg_compressibility": jpeg_compressibility,
        "unifiedreward": unifiedreward_score_sglang,
        "geneval": geneval_score,
        "clipscore": clip_score,
        "image_similarity": image_similarity_score,
        "matting": matting_score,
    }
    score_fns={}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = score_functions[score_name](device) if 'device' in score_functions[score_name].__code__.co_varnames else score_functions[score_name]()

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, ref_images=None, only_strict=True):
        total_scores = []
        score_details = {}
        
        for score_name, weight in score_dict.items():
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](images, prompts, metadata, only_strict)
                score_details['accuracy'] = rewards
                score_details['strict_accuracy'] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f'{key}_strict_accuracy'] = value
                for key, value in group_rewards.items():
                    score_details[f'{key}_accuracy'] = value
            elif score_name == "image_similarity":
                scores, rewards = score_fns[score_name](images, ref_images)
            else:
                scores, rewards = score_fns[score_name](images, prompts, metadata)
                if isinstance(rewards, dict):
                    for key, value in rewards.items():
                        score_details[key] = value
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]
            
            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]
        
        score_details['avg'] = total_scores
        return score_details, {}

    return _fn

def main():
    import torchvision.transforms as transforms

    image_paths = [
        "nasa.jpg",
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    images = torch.stack([transform(Image.open(image_path).convert('RGB')) for image_path in image_paths])
    prompts=[
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    metadata = {}  # Example metadata
    score_dict = {
        "unifiedreward": 1.0
    }
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata)
    # Print the scores
    print("Scores:", scores)


if __name__ == "__main__":
    main()
