# styleid_enhancement.py
"""
StyleID 渲染增强模块

将 StyleID（基于 Stable Diffusion 的无训练风格迁移）集成到气泡渲染流程中。
以 single_bubble_renders/ 中的 3D 渲染图作为内容图（提供结构/形状，对应 Q），
以 pre_stitch_bubbles/ 中经过旋转对齐的高质量气泡作为风格图（提供纹理风格，对应 K、V），
通过 DDIM 反演 + 自注意力特征注入实现在初始渲染图上的渲染增强。

用法：
    enhancer = StyleIDEnhancer(styleid_repo_dir, model_config, model_ckpt)
    enhanced_img = enhancer.enhance(cnt_path, sty_img_array)
    enhancer.cleanup()

集成方式为即插即用：若 styleid_enhancer 参数为 None，流程回退到原有行为。
"""

import os
import sys
import copy
import numpy as np
import cv2
from pathlib import Path
from typing import Optional
import torch


STYLEID_REPO_DIR = str(Path(__file__).resolve().parents[3] / "StyleID")


class StyleIDEnhancer:
    """封装 StyleID 推理逻辑，支持将高质量气泡风格迁移到 3D 渲染图上。"""

    def __init__(
        self,
        styleid_repo_dir: str = STYLEID_REPO_DIR,
        model_config: Optional[str] = None,
        model_ckpt: Optional[str] = None,
        start_step: int = 40,
        gamma: float = 0.75,
        T: float = 2.0,
        ddim_steps: int = 50,
        ddim_eta: float = 0.0,
        attn_layers: str = "6,7,8,9,10,11",
        precision: str = "autocast",
        min_size: int = 128,
    ):
        self.styleid_repo_dir = styleid_repo_dir
        self.start_step = start_step
        self.gamma = gamma
        self.T = T
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.attn_layers = list(map(int, attn_layers.split(",")))
        self.precision = precision
        self.min_size = min_size

        # 默认路径
        repo = Path(styleid_repo_dir)
        self.model_config = model_config or str(
            repo / "models/ldm/stable-diffusion-v1/v1-inference.yaml"
        )
        self.model_ckpt = model_ckpt or str(
            repo / "models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt"
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.sampler = None
        self.unet_model = None
        self.uc = None
        self.idx_time_dict = {}
        self.time_idx_dict = {}
        self._feat_maps = []

        self._load_model()

    # ------------------------------------------------------------------
    # 内部：加载 SD 模型（只加载一次）
    # ------------------------------------------------------------------
    def _load_model(self):
        if self.styleid_repo_dir not in sys.path:
            sys.path.insert(0, self.styleid_repo_dir)

        from omegaconf import OmegaConf
        from ldm.util import instantiate_from_config
        from ldm.models.diffusion.ddim import DDIMSampler

        print(f"[StyleIDEnhancer] 加载模型: {self.model_ckpt}")
        config = OmegaConf.load(self.model_config)
        pl_sd = torch.load(self.model_ckpt, map_location="cpu")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        model.load_state_dict(sd, strict=False)
        model.to(self.device)
        model.eval()
        self.model = model

        self.unet_model = model.model.diffusion_model
        sampler = DDIMSampler(model)
        sampler.make_schedule(
            ddim_num_steps=self.ddim_steps, ddim_eta=self.ddim_eta, verbose=False
        )
        self.sampler = sampler

        time_range = np.flip(sampler.ddim_timesteps)
        for i, t in enumerate(time_range):
            self.idx_time_dict[t] = i
            self.time_idx_dict[i] = t

        self.uc = model.get_learned_conditioning([""])
        print("[StyleIDEnhancer] 模型加载完成")

    # ------------------------------------------------------------------
    # 内部：图像预处理（灰度/彩色 → RGB tensor，确保尺寸 ≥ min_size 且为 8 的倍数）
    # ------------------------------------------------------------------
    def _preprocess(self, img: np.ndarray) -> tuple:
        """将 OpenCV 灰度或 BGR 图像转为 SD 输入 tensor。

        Returns:
            tensor: shape (1, 3, H, W)，值域 [-1, 1]
            original_size: (H, W)
            process_size: (H, W)  处理时实际使用的尺寸
        """
        from PIL import Image

        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        oh, ow = rgb.shape[:2]

        # 保证尺寸 ≥ min_size 且为 8 的倍数
        ph = max(self.min_size, (oh // 8) * 8)
        pw = max(self.min_size, (ow // 8) * 8)
        if ph != oh or pw != ow:
            pil = Image.fromarray(rgb).resize((pw, ph), Image.Resampling.LANCZOS)
            rgb = np.array(pil)

        arr = rgb.astype(np.float32) / 255.0
        t = torch.from_numpy(arr[None].transpose(0, 3, 1, 2))
        t = 2.0 * t - 1.0
        return t, (oh, ow), (ph, pw)

    # ------------------------------------------------------------------
    # 内部：DDIM 反演并捕获自注意力特征
    # ------------------------------------------------------------------
    def _invert(self, init_img_tensor: torch.Tensor) -> tuple:
        """对图像做 DDIM 反演，返回 (z_enc, feat_maps)。"""
        self._feat_maps = [{"config": {"gamma": self.gamma, "T": self.T}} for _ in range(50)]

        def _callback(pred_x0, xt, i):
            self._save_attn_maps(i)
            cur = self.idx_time_dict[i]
            self._feat_maps[cur]["z_enc"] = xt.detach().clone()

        init_z = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(init_img_tensor)
        )
        end_step = self.time_idx_dict[self.ddim_steps - 1 - self.start_step]
        z_enc, _ = self.sampler.encode_ddim(
            init_z.clone(),
            num_steps=self.ddim_steps,
            unconditional_conditioning=self.uc,
            end_step=end_step,
            callback_ddim_timesteps=self.ddim_steps,
            img_callback=_callback,
        )
        feat = copy.deepcopy(self._feat_maps)
        z_enc = self._feat_maps[0]["z_enc"]
        return z_enc, feat

    def _save_attn_maps(self, timestep):
        cur = self.idx_time_dict[timestep]
        for block_idx, block in enumerate(self.unet_model.output_blocks):
            if (
                len(block) > 1
                and "SpatialTransformer" in str(type(block[1]))
                and block_idx in self.attn_layers
            ):
                attn = block[1].transformer_blocks[0].attn1
                key = f"output_block_{block_idx}_self_attn"
                self._feat_maps[cur][key + "_q"] = attn.q.detach().clone()
                self._feat_maps[cur][key + "_k"] = attn.k.detach().clone()
                self._feat_maps[cur][key + "_v"] = attn.v.detach().clone()

    # ------------------------------------------------------------------
    # 内部：合并内容/风格特征（Q 来自内容，K/V 来自风格）
    # ------------------------------------------------------------------
    def _merge_feats(self, cnt_feats, sty_feats):
        merged = [
            {"config": {"gamma": self.gamma, "T": self.T, "timestep": i}}
            for i in range(50)
        ]
        inject_from = 50 - self.start_step
        for i in range(inject_from, 50):
            cf = cnt_feats[i]
            sf = sty_feats[i]
            for k in sf:
                if k.endswith("_q"):
                    merged[i][k] = cf[k]
                elif k.endswith("_k") or k.endswith("_v"):
                    merged[i][k] = sf[k]
        return merged

    # ------------------------------------------------------------------
    # 内部：AdaIN 对齐潜码统计
    # ------------------------------------------------------------------
    @staticmethod
    def _adain(cnt_z, sty_z):
        cm = cnt_z.mean(dim=[0, 2, 3], keepdim=True)
        cs = cnt_z.std(dim=[0, 2, 3], keepdim=True)
        sm = sty_z.mean(dim=[0, 2, 3], keepdim=True)
        ss = sty_z.std(dim=[0, 2, 3], keepdim=True)
        return (cnt_z - cm) / (cs + 1e-8) * ss + sm

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    def enhance(
        self,
        cnt_img: np.ndarray,
        sty_img: np.ndarray,
    ) -> np.ndarray:
        """对单张气泡做 StyleID 渲染增强。

        Args:
            cnt_img: 内容图（single_bubble_renders 中的 3D 渲染，灰度或 BGR uint8）
            sty_img: 风格图（pre_stitch_bubbles 中的高质量气泡，灰度或 BGR uint8）

        Returns:
            增强后的灰度图（uint8，尺寸与 cnt_img 相同）
        """
        from torch import autocast
        from contextlib import nullcontext
        from einops import rearrange

        cnt_tensor, (cnt_oh, cnt_ow), cnt_ps = self._preprocess(cnt_img)
        sty_tensor, _, _ = self._preprocess(sty_img)

        cnt_tensor = cnt_tensor.to(self.device)
        sty_tensor = sty_tensor.to(self.device)

        # DDIM 反演（捕获自注意力特征）
        sty_z, sty_feat = self._invert(sty_tensor)
        cnt_z, cnt_feat = self._invert(cnt_tensor)

        # AdaIN + 特征合并 + 去噪采样
        precision_scope = autocast if self.precision == "autocast" else nullcontext
        shape = [4, cnt_ps[0] // 8, cnt_ps[1] // 8]

        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    adain_z = self._adain(cnt_z, sty_z)
                    feat_maps = self._merge_feats(cnt_feat, sty_feat)

                    samples, _ = self.sampler.sample(
                        S=self.ddim_steps,
                        batch_size=1,
                        shape=shape,
                        verbose=False,
                        unconditional_conditioning=self.uc,
                        eta=self.ddim_eta,
                        x_T=adain_z,
                        injected_features=feat_maps,
                        start_step=self.start_step,
                    )

                    out = self.model.decode_first_stage(samples)
                    out = torch.clamp((out + 1.0) / 2.0, 0.0, 1.0)
                    out = out.cpu().permute(0, 2, 3, 1).numpy()
                    out = (out[0] * 255).astype(np.uint8)

        # 转回灰度，并还原到原始 cnt 尺寸
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        if (cnt_oh, cnt_ow) != (gray.shape[0], gray.shape[1]):
            gray = cv2.resize(gray, (cnt_ow, cnt_oh), interpolation=cv2.INTER_LANCZOS4)

        # 清理本次推理的中间显存
        del cnt_tensor, sty_tensor, cnt_z, sty_z, cnt_feat, sty_feat
        del adain_z, feat_maps, samples
        torch.cuda.empty_cache()

        return gray

    def cleanup(self):
        """释放 SD 模型显存。"""
        if self.model is not None:
            del self.model, self.sampler, self.unet_model, self.uc
            self.model = None
            torch.cuda.empty_cache()
            print("[StyleIDEnhancer] 模型已释放")
