import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CONFIGS = {"tiny": "configs/sam2.1/sam2.1_hiera_t.yaml", "small": "configs/sam2.1/sam2.1_hiera_s.yaml"}
BOX = ["x1", "y1", "x2", "y2"]
CKPTS = {"tiny": "sam2.1_hiera_tiny.pt", "small": "sam2.1_hiera_small.pt"}
CKPT_URLS = {"tiny": "https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/de431c4043854a71d8101e17995dfe596bf101a5/",
             "small": "https://huggingface.co/facebook/sam2.1-hiera-small/resolve/main/"}


def load_frame(path: str | Path, size: int):
    import torch

    img = Image.open(path).convert("RGB").resize((size, size))
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
    return (x - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)


def mask_box(m: np.ndarray, mode: str = "largest") -> tuple[list[float], int, int]:
    from scipy import ndimage

    if not m.any():
        return [np.nan] * 4, 0, 0
    lab, n = ndimage.label(m)
    if mode == "largest" and n > 1:
        sizes = np.bincount(lab.ravel())[1:]
        m = lab == (np.argmax(sizes) + 1)
    ys = np.flatnonzero(m.any(axis=1))
    xs = np.flatnonzero(m.any(axis=0))
    return [float(xs[0]), float(ys[0]), float(xs[-1] + 1), float(ys[-1] + 1)], int(m.sum()), int(n)


def calibrate(boxes: np.ndarray, mask_box_ref: np.ndarray, ref: np.ndarray, lo: float = 0.5, hi: float = 4.0) -> np.ndarray:
    if not (np.isfinite(mask_box_ref).all() and np.isfinite(ref).all()):
        return boxes
    mw, mh = mask_box_ref[2] - mask_box_ref[0], mask_box_ref[3] - mask_box_ref[1]
    if mw <= 0 or mh <= 0:
        return boxes
    sx = np.clip((ref[2] - ref[0]) / mw, lo, hi)
    sy = np.clip((ref[3] - ref[1]) / mh, lo, hi)
    dx = ((ref[0] + ref[2]) - (mask_box_ref[0] + mask_box_ref[2])) / 2 / mw
    dy = ((ref[1] + ref[3]) - (mask_box_ref[1] + mask_box_ref[3])) / 2 / mh
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    cx = (boxes[:, 0] + boxes[:, 2]) / 2 + dx * w
    cy = (boxes[:, 1] + boxes[:, 3]) / 2 + dy * h
    return np.stack([cx - sx * w / 2, cy - sy * h / 2, cx + sx * w / 2, cy + sy * h / 2], axis=1)


class _Frames:
    def __init__(self, n: int):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        raise RuntimeError("frames are served from the feature cache")


class Propagator:
    def __init__(self, model: str = "tiny", ckpt_dir: str | Path = "checkpoints", device: str = "cuda",
                 fp16: bool = True):
        import torch
        import sam2.sam2_video_predictor as svp
        from sam2.build_sam import build_sam2_video_predictor

        ckpt = self.fetch_checkpoint(model, ckpt_dir)
        self.torch = torch
        self.device = device
        self.dtype = torch.float16 if fp16 and device.startswith("cuda") else torch.float32
        self.box_mode = "largest"
        self.calibrate = True
        self.amp = dict(device_type="cuda", dtype=self.dtype, enabled=self.dtype != torch.float32)
        self.pred = build_sam2_video_predictor(CONFIGS[model], str(ckpt), device=device)
        self.size = self.pred.image_size
        self.cache: dict[int, list] = {}
        self.pos = None
        self.offset = 0
        self.stats: dict[int, tuple[float, float]] = {}
        self.hw = (720, 1280)
        svp.load_video_frames = self._load_video_frames
        self.pred._get_image_feature = self._get_image_feature
        heads = self.pred._forward_sam_heads
        last: dict = {}

        def spy_heads(*a, **k):
            out = heads(*a, **k)
            last["iou"] = float(out[2].max().float().item())
            last["obj"] = float(out[6].max().float().item())
            return out

        self.pred._forward_sam_heads = spy_heads
        single = self.pred._run_single_frame_inference

        def spy_single(*a, **k):
            res = single(*a, **k)
            fi = k["frame_idx"] if "frame_idx" in k else a[2]
            self.stats[self.offset + fi] = (last.get("iou", np.nan), last.get("obj", np.nan))
            return res

        self.pred._run_single_frame_inference = spy_single

    @staticmethod
    def fetch_checkpoint(model: str, ckpt_dir: str | Path) -> Path:
        ckpt = Path(ckpt_dir) / CKPTS[model]
        if not ckpt.exists():
            import urllib.request

            ckpt.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(CKPT_URLS[model] + CKPTS[model], ckpt)
        return ckpt

    def _load_video_frames(self, video_path, image_size, offload_video_to_cpu, img_mean=MEAN, img_std=STD,
                           async_loading_frames=False, compute_device=None):
        return _Frames(int(video_path)), self.hw[0], self.hw[1]

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        fpn = self.cache[self.offset + frame_idx]
        bo = {"backbone_fpn": [f.expand(batch_size, -1, -1, -1) for f in fpn],
              "vision_pos_enc": [p.expand(batch_size, -1, -1, -1) for p in self.pos]}
        return (None,) + self.pred._prepare_backbone_features(bo)

    def encode(self, paths: list[str | Path], batch: int = 8, workers: int = 4) -> float:
        torch = self.torch
        self.cache, self.stats = {}, {}
        self.hw = Image.open(paths[0]).size[::-1]
        t0 = time.time()
        with ThreadPoolExecutor(workers) as ex, torch.inference_mode(), torch.autocast(**self.amp):
            for s in range(0, len(paths), batch):
                imgs = list(ex.map(lambda p: load_frame(p, self.size), paths[s:s + batch]))
                x = torch.stack(imgs).to(self.device)
                out = self.pred.forward_image(x)
                if self.pos is None:
                    self.pos = [p[:1].clone() for p in out["vision_pos_enc"]]
                for j in range(x.shape[0]):
                    self.cache[s + j] = [f[j:j + 1].clone() for f in out["backbone_fpn"]]
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return time.time() - t0

    def run(self, start: int, box: np.ndarray, length: int, reverse: bool = False) -> pd.DataFrame:
        torch = self.torch
        n = len(self.cache)
        if reverse:
            lo = max(0, start - length + 1)
            self.offset, count, prompt = lo, start - lo + 1, start - lo
        else:
            self.offset, count, prompt = start, min(length, n - start), 0
        rows = []
        with torch.inference_mode(), torch.autocast(**self.amp):
            st = self.pred.init_state(video_path=str(count), offload_video_to_cpu=True)
            self.pred.add_new_points_or_box(st, frame_idx=prompt, obj_id=1, box=np.asarray(box, dtype=np.float32))
            for fi, _, logits in self.pred.propagate_in_video(st, start_frame_idx=prompt, reverse=reverse):
                m = (logits[0, 0] > 0).cpu().numpy()
                f = self.offset + fi
                iou_pred, obj = self.stats.get(f, (np.nan, np.nan))
                b, area, n_cc = mask_box(m, self.box_mode)
                rows.append([f, *b, area, n_cc, iou_pred, obj])
            self.pred.reset_state(st)
        df = pd.DataFrame(rows, columns=["frame", "x1", "y1", "x2", "y2", "mask_area", "n_components", "pred_iou",
                                         "obj_logit"]).sort_values("frame").reset_index(drop=True)
        if self.calibrate:
            raw = df[BOX].to_numpy(dtype=float)
            ref = raw[df["frame"].to_numpy() == start]
            df[[f"raw_{c}" for c in BOX]] = raw
            df[BOX] = calibrate(raw, ref[0] if len(ref) else np.full(4, np.nan), np.asarray(box, dtype=float))
        return df

    def run_boxes(self, start: int, box: np.ndarray, length: int) -> np.ndarray:
        df = self.run(start, box, length)
        return df[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
