import os
import sys
import random
from dataclasses import dataclass, asdict
from pathlib import Path

# Ensure CPU-only before importing TensorFlow
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # reduce TF log verbosity
os.environ.setdefault("PYTHONHASHSEED", "0")

try:
    import numpy as np
except Exception as e:  # pragma: no cover
    np = None


@dataclass
class Config:
    # Paths
    content_dir: Path = Path("data/content")
    style_dir: Path = Path("data/style")
    outputs_runs_dir: Path = Path("outputs/runs")
    outputs_plots_dir: Path = Path("outputs/plots")
    metrics_dir: Path = Path("metrics")

    # Image + optimization
    image_size: int = 256  # final working square size
    steps: int = 100
    alpha: float = 1.0  # content weight
    beta: float = 1e3   # style weight
    gamma: float = 10.0 # total variation weight

    # Reproducibility
    seed: int = 42

    # Handling mixed image sizes/aspects
    # Options: "square-resize" or "center-crop-then-resize"
    resize_policy: str = "square-resize"

    # Optimization config (steps 5 & 6)
    learning_rate: float = 0.02
    log_every: int = 25
    check_every: int = 50
    early_stop_from_step: int = 300
    early_stop_rel_improve: float = 0.005  # 0.5%
    save_every: int = 50  # Step 7 checkpoint interval

    # Optional orchestration toggles (Steps 9–12)
    do_sanity_tests: bool = True
    do_batch_grid: bool = True
    do_portfolio_gallery: bool = True
    do_analysis_plots: bool = True


def ensure_dirs(cfg: Config) -> None:
    for p in [
        cfg.content_dir,
        cfg.style_dir,
        cfg.outputs_runs_dir,
        cfg.outputs_plots_dir,
        cfg.metrics_dir,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import tensorflow as tf  # noqa: F401
        tf.random.set_seed(seed)
    except Exception:
        pass
    if np is not None:
        np.random.seed(seed)


def tf_cpu_only_report() -> str:
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        # Also try to hide GPUs via TF runtime in case env var is ignored
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            # If GPUs were never visible or drivers missing, this may throw
            pass
        cpus = tf.config.list_physical_devices("CPU")
        return (
            f"TensorFlow {tf.__version__} | GPUs visible: {len(gpus)} | CPUs visible: {len(cpus)}"
        )
    except ImportError:
        return (
            "TensorFlow not installed. Install 'tensorflow-cpu' (CPU-only) to proceed."
        )


def print_config(cfg: Config) -> None:
    print("=== Neural Style Transfer (CPU) - Run Config ===")
    for k, v in asdict(cfg).items():
        print(f"{k}: {v}")
    print("================================================")


# Shared formatting for deterministic filenames
def _fmt_float_str(x: float) -> str:
    return "%g" % float(x)


def _deterministic_base(cfg: Config, content_stem: str, style_stem: str) -> str:
    return (
        f"{content_stem}__{style_stem}__"
        f"{cfg.image_size}px_{cfg.steps}s_"
        f"a{_fmt_float_str(cfg.alpha)}_b{_fmt_float_str(cfg.beta)}_tv{_fmt_float_str(cfg.gamma)}_"
        f"lr{_fmt_float_str(cfg.learning_rate)}_seed{cfg.seed}"
    )


def load_image(path: Path, size: int, policy: str = "square-resize"):
    """Read an image file -> float32 [0,1], square-resized, with batch dim.

    Returns: tf.Tensor with shape (1, size, size, 3), dtype=float32
    """
    import tensorflow as tf

    img_bytes = tf.io.read_file(str(path))
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # now in [0,1]

    if policy == "center-crop-then-resize":
        shape = tf.shape(img)
        h, w = shape[0], shape[1]
        min_side = tf.minimum(h, w)
        img = tf.image.resize_with_crop_or_pad(img, min_side, min_side)
        img = tf.image.resize(img, [size, size], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
    else:  # "square-resize" (default)
        img = tf.image.resize(img, [size, size], method=tf.image.ResizeMethod.BILINEAR, antialias=True)

    img = tf.expand_dims(img, axis=0)
    return img


def save_image(tensor, path: Path) -> None:
    """Clip to [0,1], remove batch dim if present, write PNG."""
    import tensorflow as tf

    img = tensor
    # Remove batch dim if provided
    if len(img.shape) == 4 and img.shape[0] == 1:
        img = tf.squeeze(img, axis=0)
    # Ensure correct dtype and range
    img = tf.clip_by_value(tf.convert_to_tensor(img, dtype=tf.float32), 0.0, 1.0)
    img = tf.cast(tf.round(img * 255.0), tf.uint8)
    png = tf.image.encode_png(img)
    tf.io.write_file(str(path), png)


def _first_image_in(dir_path: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for p in sorted(dir_path.glob("**/*")):
        if p.suffix.lower() in exts and p.is_file():
            return p
    return None


def verify_image_io_roundtrip(cfg: Config) -> None:
    """Load -> save for one content and one style image if available."""
    try:
        import tensorflow as tf  # noqa: F401
    except Exception:
        print("Skipping image I/O verification: TensorFlow not available.")
        return

    # Pick sample images
    samples = [("content", _first_image_in(cfg.content_dir)), ("style", _first_image_in(cfg.style_dir))]

    for label, img_path in samples:
        if img_path is None:
            print(f"No {label} images found in '{cfg.content_dir if label=='content' else cfg.style_dir}'. Skipping {label} round-trip.")
            continue

        print(f"Round-trip check for {label}: {img_path}")
        img = load_image(img_path, cfg.image_size, cfg.resize_policy)

        # Basic sanity on tensor
        import tensorflow as tf
        t_min = float(tf.reduce_min(img).numpy())
        t_max = float(tf.reduce_max(img).numpy())
        print(f"  loaded shape: {tuple(img.shape)}, dtype: {img.dtype}, min/max: {t_min:.4f}/{t_max:.4f}")

        # Save a round-trip PNG under outputs/plots
        out_name = f"step2_roundtrip_{label}_{cfg.image_size}px.png"
        out_path = cfg.outputs_plots_dir / out_name
        save_image(img, out_path)
        print(f"  saved: {out_path}")


def build_vgg_feature_extractor(style_layers=None, content_layers=None):
    """Create a frozen VGG19 model that returns selected layer activations.

    Returns an object with call(input[0,1]) -> {'style': {..}, 'content': {..}}
    """
    import tensorflow as tf
    from tensorflow.keras.applications import vgg19

    if style_layers is None:
        style_layers = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
    if content_layers is None:
        content_layers = ["block5_conv2"]

    # Load VGG19, exclude top classification layers
    vgg = vgg19.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in (style_layers + content_layers)]
    model = tf.keras.Model(inputs=vgg.input, outputs=outputs)
    model.trainable = False

    class VGGFeatureExtractor(tf.keras.models.Model):
        def __init__(self, backbone, style_layers, content_layers):
            super().__init__()
            self.vgg = backbone
            self.vgg.trainable = False
            self.style_layers = list(style_layers)
            self.content_layers = list(content_layers)
            self.num_style = len(self.style_layers)

        def call(self, inputs):
            x = inputs
            # Expect inputs in [0,1], convert to VGG expected range
            x = x * 255.0
            x = vgg19.preprocess_input(x)
            outputs = self.vgg(x)
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            style_outputs = outputs[: self.num_style]
            content_outputs = outputs[self.num_style :]

            style_dict = {name: value for name, value in zip(self.style_layers, style_outputs)}
            content_dict = {name: value for name, value in zip(self.content_layers, content_outputs)}
            return {"style": style_dict, "content": content_dict}

    extractor = VGGFeatureExtractor(model, style_layers, content_layers)
    print(
        "VGG feature taps -> style:",
        ", ".join(style_layers),
        "| content:",
        ", ".join(content_layers),
    )
    # Sanity: ensure frozen
    trainable_vars = sum(int(v.trainable) for v in extractor.vgg.variables)
    if trainable_vars != 0:
        print(f"WARNING: VGG has {trainable_vars} trainable variables; expected 0.")
    return extractor


def verify_vgg_extractor(cfg: Config) -> None:
    try:
        import tensorflow as tf  # noqa: F401
    except Exception:
        print("Skipping VGG verification: TensorFlow not available.")
        return

    extractor = build_vgg_feature_extractor()

    sample = _first_image_in(cfg.content_dir) or _first_image_in(cfg.style_dir)
    if sample is None:
        print("No images found for VGG verification. Add images to data/content or data/style.")
        return

    img = load_image(sample, cfg.image_size, cfg.resize_policy)
    outs = extractor(img)
    style_feats = outs["style"]
    content_feats = outs["content"]

    print("VGG forward pass outputs:")
    for name, feat in style_feats.items():
        try:
            shape = tuple(feat.shape)
            print(f"  style[{name}]: {shape}")
        except Exception:
            print(f"  style[{name}]: <tensor>")
    for name, feat in content_feats.items():
        try:
            shape = tuple(feat.shape)
            print(f"  content[{name}]: {shape}")
        except Exception:
            print(f"  content[{name}]: <tensor>")


def gram_matrix(features):
    """Compute Gram matrix for NHWC features, normalized by H*W.

    Accepts (1, H, W, C) or (H, W, C); returns (1, C, C) or (B, C, C).
    """
    import tensorflow as tf

    x = tf.convert_to_tensor(features, dtype=tf.float32)
    if x.shape.rank == 3:
        x = tf.expand_dims(x, axis=0)  # (1, H, W, C)

    shape = tf.shape(x)
    b, h, w, c = shape[0], shape[1], shape[2], shape[3]
    # sum over spatial dims, correlate channels
    gram = tf.einsum('bhwc,bhwd->bcd', x, x)
    norm = tf.cast(h * w, tf.float32)
    gram = gram / tf.maximum(norm, 1.0)
    return gram


def grams_from_style_features(style_feats: dict) -> dict:
    return {name: gram_matrix(t) for name, t in style_feats.items()}


def verify_gram_matrices(cfg: Config) -> None:
    try:
        import tensorflow as tf  # noqa: F401
    except Exception:
        print("Skipping Gram verification: TensorFlow not available.")
        return

    extractor = build_vgg_feature_extractor()
    sample = _first_image_in(cfg.content_dir) or _first_image_in(cfg.style_dir)
    if sample is None:
        print("No images found for Gram verification. Add images to data/content or data/style.")
        return

    img = load_image(sample, cfg.image_size, cfg.resize_policy)
    outs = extractor(img)
    grams1 = grams_from_style_features(outs['style'])
    grams2 = grams_from_style_features(extractor(img)['style'])

    print("Gram matrices (normalized by H*W):")
    for name in grams1.keys():
        g1 = grams1[name]
        g2 = grams2[name]
        shape = tuple(g1.shape)
        # Determinism check
        diff = tf.reduce_max(tf.abs(g1 - g2)).numpy()
        print(f"  {name}: {shape}, max|Δ|={diff:.3e}")


def _mse(a, b):
    import tensorflow as tf
    a = tf.convert_to_tensor(a, dtype=tf.float32)
    b = tf.convert_to_tensor(b, dtype=tf.float32)
    return tf.reduce_mean(tf.square(a - b))


def prepare_targets(extractor, content_img, style_img):
    """Compute fixed targets for content and style.

    Returns: (content_targets: dict, style_gram_targets: dict)
    """
    outs_c = extractor(content_img)
    outs_s = extractor(style_img)
    content_targets = outs_c["content"]
    style_gram_targets = grams_from_style_features(outs_s["style"])
    return content_targets, style_gram_targets


def compute_losses(extractor, image, content_targets, style_gram_targets, alpha, beta, gamma):
    import tensorflow as tf
    outs = extractor(image)
    content_outs = outs["content"]
    style_outs = outs["style"]

    # Content loss: average MSE across selected content layers
    content_losses = []
    for name, target in content_targets.items():
        content_losses.append(_mse(content_outs[name], target))
    content_loss = tf.add_n(content_losses) / tf.cast(len(content_losses), tf.float32)

    # Style loss: average MSE across Gram matrices
    grams = grams_from_style_features(style_outs)
    style_losses = []
    for name, target in style_gram_targets.items():
        style_losses.append(_mse(grams[name], target))
    style_loss = tf.add_n(style_losses) / tf.cast(len(style_losses), tf.float32)

    # TV loss: smoothness prior
    tv_loss = tf.reduce_mean(tf.image.total_variation(image))

    total = alpha * content_loss + beta * style_loss + gamma * tv_loss
    return total, content_loss, style_loss, tv_loss


def _clone_cfg(cfg: Config, **overrides) -> Config:
    data = asdict(cfg)
    data.update(overrides)
    return Config(**data)


def _optimize_with_targets(
    cfg: Config,
    extractor,
    content_img,
    style_img,
    content_targets,
    style_gram_targets,
    content_path: Path,
    style_path: Path,
):
    """Runs NST optimization on the first content/style images found, with logging.

    Saves a preview final image to outputs/plots.
    """
    try:
        import tensorflow as tf
        import time
    except Exception:
        print("Skipping optimization: TensorFlow not available.")
        return

    print(f"Optimization pair -> content: {content_path.name} | style: {style_path.name}")

    image = tf.Variable(content_img)
    opt = tf.optimizers.Adam(learning_rate=cfg.learning_rate)

    last_check_loss = None
    best_loss = None
    best_image = None
    last_step = 0

    # Naming helpers for Step 7
    def _fmt_float(x: float) -> str:
        # compact, deterministic float formatting
        return ("%g" % float(x))

    base = (
        f"{content_path.stem}__{style_path.stem}__"
        f"{cfg.image_size}px_{cfg.steps}s_"
        f"a{_fmt_float(cfg.alpha)}_b{_fmt_float(cfg.beta)}_tv{_fmt_float(cfg.gamma)}_"
        f"lr{_fmt_float(cfg.learning_rate)}_seed{cfg.seed}"
    )

    def _checkpoint_path(step_num: int) -> Path:
        width = max(3, len(str(cfg.steps)))
        return cfg.outputs_runs_dir / f"{base}__step{step_num:0{width}d}.png"

    def _final_path() -> Path:
        return cfg.outputs_runs_dir / f"{base}__final.png"

    start_time = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        with tf.GradientTape() as tape:
            total, c_loss, s_loss, tv_loss = compute_losses(
                extractor, image, content_targets, style_gram_targets, cfg.alpha, cfg.beta, cfg.gamma
            )
        grad = tape.gradient(total, image)
        opt.apply_gradients([(grad, image)])
        # Keep image in valid range
        image.assign(tf.clip_by_value(image, 0.0, 1.0))

        if step % cfg.log_every == 0 or step == 1 or step == cfg.steps:
            print(
                f"step {step:04d}/{cfg.steps}: total={float(total.numpy()):.4e} "
                f"content={float(c_loss.numpy()):.4e} style={float(s_loss.numpy()):.4e} tv={float(tv_loss.numpy()):.4e}"
            )

        # Track best
        if best_loss is None or float(total.numpy()) < best_loss:
            best_loss = float(total.numpy())
            best_image = tf.identity(image)
        last_step = step

        # Step 7: checkpoint every N steps
        if step % cfg.save_every == 0:
            ckpt_path = _checkpoint_path(step)
            save_image(image, ckpt_path)
            # Minimal progress print for saved file
            print(f"  saved checkpoint: {ckpt_path}")

        # Early stopping (relative improvement insufficient)
        if step >= cfg.early_stop_from_step and step % cfg.check_every == 0:
            if last_check_loss is not None:
                rel_improve = (last_check_loss - float(total.numpy())) / max(abs(last_check_loss), 1e-12)
                print(f"  early-stop check @ {step}: rel_improve={rel_improve:.3%}")
                if rel_improve < cfg.early_stop_rel_improve:
                    print("Early stopping: relative improvement below threshold.")
                    break
            last_check_loss = float(total.numpy())

    # Save final image under outputs/runs with deterministic name
    final = best_image if best_image is not None else image
    final_path = _final_path()
    save_image(final, final_path)
    print(f"Saved final image: {final_path}")

    # Also save last-step image for auditability if not already saved
    last_ckpt = _checkpoint_path(last_step)
    if not last_ckpt.exists():
        save_image(image, last_ckpt)
        print(f"Saved last-step image: {last_ckpt}")

    # Step 8: append metrics CSV
    try:
        # Compute SSIM vs content (on best/final image)
        ssim = float(tf.image.ssim(tf.squeeze(final, axis=0), tf.squeeze(content_img, axis=0), max_val=1.0).numpy())
    except Exception:
        ssim = None

    try:
        from datetime import datetime, timezone
        import csv
        wall_time_s = time.perf_counter() - start_time
        ts = datetime.now(timezone.utc).isoformat()
        metrics_file = cfg.metrics_dir / "metrics.csv"
        header = [
            "timestamp","content","style","size","steps","alpha","beta","gamma","lr","seed",
            "final_total_loss","wall_time_s","ssim_vs_content","final_image"
        ]
        row = [
            ts,
            content_path.name,
            style_path.name,
            str(cfg.image_size),
            str(cfg.steps),
            _fmt_float(cfg.alpha),
            _fmt_float(cfg.beta),
            _fmt_float(cfg.gamma),
            _fmt_float(cfg.learning_rate),
            str(cfg.seed),
            f"{best_loss:.6e}" if best_loss is not None else "",
            f"{wall_time_s:.3f}",
            f"{ssim:.6f}" if ssim is not None else "",
            str(final_path)
        ]
        write_header = not metrics_file.exists()
        with open(metrics_file, mode="a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow(row)
        print(f"Appended metrics: {metrics_file}")
    except Exception as e:
        print(f"WARNING: failed to write metrics.csv: {e}")


def run_single_nst(cfg: Config, content_path: Path, style_path: Path, **overrides) -> None:
    """Single NST run for a specified pair, with optional config overrides."""
    try:
        import tensorflow as tf  # noqa: F401
    except Exception:
        print("TensorFlow not available; skipping run.")
        return

    cfg2 = _clone_cfg(cfg, **overrides)
    extractor = build_vgg_feature_extractor()
    content_img = load_image(content_path, cfg2.image_size, cfg2.resize_policy)
    style_img = load_image(style_path, cfg2.image_size, cfg2.resize_policy)
    content_targets, style_gram_targets = prepare_targets(extractor, content_img, style_img)
    _optimize_with_targets(cfg2, extractor, content_img, style_img, content_targets, style_gram_targets, content_path, style_path)


def run_style_transfer_step6(cfg: Config) -> None:
    """Default single run using the first images in content/style folders."""
    content_path = _first_image_in(cfg.content_dir)
    style_path = _first_image_in(cfg.style_dir)
    if content_path is None or style_path is None:
        print("No content/style images found for optimization. Place files in data/content and data/style.")
        return
    run_single_nst(cfg, content_path, style_path)


def run_sanity_tests(cfg: Config) -> None:
    """Step 9: quick sanity tests.

    - Smoke test: 256px, 50 steps, alpha=1000
    - Beta sweep at 384px: beta in {1000, 2000, 5000} with fixed alpha,gamma
    - Resolution check: sizes {256,384,512} with fixed weights
    """
    content_path = _first_image_in(cfg.content_dir)
    style_path = _first_image_in(cfg.style_dir)
    if content_path is None or style_path is None:
        print("Sanity tests: need at least one content and one style image.")
        return

    print("[Sanity] Smoke test: 256px, 50 steps, alpha=1000")
    run_single_nst(cfg, content_path, style_path, image_size=256, steps=50, alpha=1000.0)

    print("[Sanity] Beta sweep at 256px: beta in {1000,2000,5000}")
    for b in [1000.0, 2000.0, 5000.0]:
        run_single_nst(cfg, content_path, style_path, image_size=256, steps=100, beta=b)

    print("[Sanity] Resolution check at fixed weights: sizes in {256,384,512}")
    for sz in [256, 384, 512]:
        run_single_nst(cfg, content_path, style_path, image_size=sz, steps=100)


def run_batch_grid(cfg: Config, max_content: int = None, max_style: int = None, betas=None, size: int = 256) -> None:
    """Step 10: batch runner over content×style grid and beta presets at fixed size (default 256px).

    Avoids recomputing targets inside inner loop by caching per pair.
    """
    try:
        import tensorflow as tf  # noqa: F401
    except Exception:
        print("TensorFlow not available; skipping batch runner.")
        return

    content_list = sorted([p for p in cfg.content_dir.glob("**/*") if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}])
    style_list = sorted([p for p in cfg.style_dir.glob("**/*") if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}])
    if max_content:
        content_list = content_list[:max_content]
    if max_style:
        style_list = style_list[:max_style]
    if not content_list or not style_list:
        print("Batch runner: no images found in data/content or data/style.")
        return

    if betas is None:
        betas = [1000.0, 2000.0, 5000.0]

    print(f"[Batch] Running grid: {len(content_list)} contents × {len(style_list)} styles × {len(betas)} betas @ {size}px")

    extractor = build_vgg_feature_extractor()
    for ci, cpath in enumerate(content_list, 1):
        for sj, spath in enumerate(style_list, 1):
            try:
                cimg = load_image(cpath, size, cfg.resize_policy)
                simg = load_image(spath, size, cfg.resize_policy)
                c_targets, s_gram_targets = prepare_targets(extractor, cimg, simg)

                for bk, b in enumerate(betas, 1):
                    print(f"[Batch] ({ci}/{len(content_list)} x {sj}/{len(style_list)}) beta={b}")
                    cfg_b = _clone_cfg(cfg, image_size=size, beta=b)
                    _optimize_with_targets(cfg_b, extractor, cimg, simg, c_targets, s_gram_targets, cpath, spath)
            except Exception as e:
                print(f"[Batch] ERROR for pair ({cpath.name}, {spath.name}): {e}")
                continue


def run_portfolio_and_gallery(
    cfg: Config,
    preferred_beta: float = 2000.0,
    grid: int = 6,
    run_size: int = 256,
    render_size: int = 512,
) -> None:
    """Step 11: run a 6x6 grid mostly at a smaller size, render gallery at higher size.

    - Optimization runs happen at `run_size` (default 256px) for speed.
    - Gallery tiles are rendered at `render_size` (default 512px) by resizing tiles.
    """
    # Select first N content/style images deterministically
    content_list = sorted([p for p in cfg.content_dir.glob("**/*") if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}])[:grid]
    style_list = sorted([p for p in cfg.style_dir.glob("**/*") if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}])[:grid]
    if len(content_list) < grid or len(style_list) < grid:
        print(f"Portfolio: need at least {grid} content and {grid} style images.")
        return

    # Run the grid at the smaller run_size and beta
    print(f"[Portfolio] Running {grid}x{grid} at {run_size}px, beta={preferred_beta}")
    run_batch_grid(cfg, max_content=grid, max_style=grid, betas=[preferred_beta], size=run_size)

    # Assemble gallery from final images
    try:
        from PIL import Image
    except Exception:
        print("Pillow not available; skipping gallery assembly.")
        return

    # Build list of expected final paths
    def final_path_for(stem_c: str, stem_s: str) -> Path:
        # Final images were produced at run_size
        cfg_tmp = _clone_cfg(cfg, image_size=run_size, beta=preferred_beta)
        base = _deterministic_base(cfg_tmp, stem_c, stem_s)
        return cfg.outputs_runs_dir / f"{base}__final.png"

    tiles: list[list[Path]] = []
    for c in content_list:
        row = []
        for s in style_list:
            row.append(final_path_for(c.stem, s.stem))
        tiles.append(row)

    # Compose a grid
    tile_w = render_size
    tile_h = render_size
    gallery_w = grid * tile_w
    gallery_h = grid * tile_h
    gallery = Image.new("RGB", (gallery_w, gallery_h), color=(255, 255, 255))

    for i, row in enumerate(tiles):
        for j, path in enumerate(row):
            try:
                im = Image.open(path).convert("RGB")
                if im.size != (tile_w, tile_h):
                    im = im.resize((tile_w, tile_h))
                gallery.paste(im, (j * tile_w, i * tile_h))
            except Exception as e:
                print(f"[Gallery] Missing or unreadable tile {path}: {e}")

    out_name = (
        f"gallery_{grid}x{grid}_{render_size}px_"
        f"a{_fmt_float_str(cfg.alpha)}_b{_fmt_float_str(preferred_beta)}_tv{_fmt_float_str(cfg.gamma)}_"
        f"lr{_fmt_float_str(cfg.learning_rate)}_seed{cfg.seed}.png"
    )
    out_path = cfg.outputs_plots_dir / out_name
    gallery.save(out_path)
    print(f"[Portfolio] Saved gallery: {out_path}")


def analysis_plots_from_csv(cfg: Config) -> None:
    """Step 12: read metrics.csv and produce analysis plots in outputs/plots/.

    - SSIM vs beta (lines by size)
    - Runtime vs size (bar)
    - Runtime vs beta (bar)
    """
    import csv
    from collections import defaultdict
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping analysis plots.")
        return

    metrics_file = cfg.metrics_dir / "metrics.csv"
    if not metrics_file.exists():
        print(f"No metrics file found at {metrics_file}; run some NST first.")
        return

    rows = []
    with open(metrics_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "content": r["content"],
                    "style": r["style"],
                    "size": int(float(r["size"])) if r["size"] else None,
                    "steps": int(float(r["steps"])) if r["steps"] else None,
                    "alpha": float(r["alpha"]) if r["alpha"] else None,
                    "beta": float(r["beta"]) if r["beta"] else None,
                    "gamma": float(r["gamma"]) if r["gamma"] else None,
                    "lr": float(r["lr"]) if r["lr"] else None,
                    "seed": int(float(r["seed"])) if r["seed"] else None,
                    "final_total_loss": float(r["final_total_loss"]) if r["final_total_loss"] else None,
                    "wall_time_s": float(r["wall_time_s"]) if r["wall_time_s"] else None,
                    "ssim": float(r["ssim_vs_content"]) if r["ssim_vs_content"] else None,
                })
            except Exception:
                continue

    if not rows:
        print("metrics.csv has no valid data rows.")
        return

    # SSIM vs beta by size
    ssim_by_size_beta = defaultdict(list)
    for r in rows:
        if r["ssim"] is not None and r["beta"] is not None and r["size"] is not None:
            ssim_by_size_beta[(r["size"], r["beta"])].append(r["ssim"])

    if ssim_by_size_beta:
        plt.figure(figsize=(7, 5))
        sizes = sorted(set(k[0] for k in ssim_by_size_beta.keys()))
        for sz in sizes:
            betas = sorted(b for (s, b) in ssim_by_size_beta.keys() if s == sz)
            means = []
            for b in betas:
                vals = ssim_by_size_beta[(sz, b)]
                means.append(sum(vals) / max(len(vals), 1))
            plt.plot(betas, means, marker='o', label=f"{sz}px")
        plt.xlabel("beta (style weight)")
        plt.ylabel("SSIM vs content (avg)")
        plt.title("SSIM vs beta (by size)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        out = cfg.outputs_plots_dir / "ssim_vs_beta_by_size.png"
        plt.tight_layout(); plt.savefig(out); plt.close()
        print(f"[Analysis] Saved: {out}")

    # Runtime vs size (average)
    rt_by_size = defaultdict(list)
    for r in rows:
        if r["wall_time_s"] is not None and r["size"] is not None:
            rt_by_size[r["size"]].append(r["wall_time_s"])
    if rt_by_size:
        plt.figure(figsize=(6, 4))
        sizes = sorted(rt_by_size.keys())
        means = [sum(v)/len(v) for v in (rt_by_size[s] for s in sizes)]
        plt.bar([str(s) for s in sizes], means)
        plt.xlabel("image size (px)")
        plt.ylabel("runtime (s, avg)")
        plt.title("Runtime vs size")
        out = cfg.outputs_plots_dir / "runtime_vs_size.png"
        plt.tight_layout(); plt.savefig(out); plt.close()
        print(f"[Analysis] Saved: {out}")

    # Runtime vs beta (average)
    rt_by_beta = defaultdict(list)
    for r in rows:
        if r["wall_time_s"] is not None and r["beta"] is not None:
            rt_by_beta[r["beta"]].append(r["wall_time_s"])
    if rt_by_beta:
        plt.figure(figsize=(6, 4))
        betas = sorted(rt_by_beta.keys())
        means = [sum(v)/len(v) for v in (rt_by_beta[b] for b in betas)]
        plt.bar([_fmt_float_str(b) for b in betas], means)
        plt.xlabel("beta (style weight)")
        plt.ylabel("runtime (s, avg)")
        plt.title("Runtime vs beta")
        out = cfg.outputs_plots_dir / "runtime_vs_beta.png"
        plt.tight_layout(); plt.savefig(out); plt.close()
        print(f"[Analysis] Saved: {out}")

    # Runtime vs beta (grouped by size)
    rt_by_beta_size = defaultdict(list)
    sizes_all = set()
    betas_all = set()
    for r in rows:
        if r["wall_time_s"] is not None and r["beta"] is not None and r["size"] is not None:
            rt_by_beta_size[(r["beta"], r["size"])].append(r["wall_time_s"])
            sizes_all.add(r["size"]); betas_all.add(r["beta"])
    if rt_by_beta_size:
        import numpy as np
        sizes = sorted(sizes_all)
        betas = sorted(betas_all)
        x = np.arange(len(betas))
        width = 0.8 / max(len(sizes), 1)
        plt.figure(figsize=(8, 4.5))
        for i, sz in enumerate(sizes):
            means = []
            for b in betas:
                vals = rt_by_beta_size.get((b, sz), [])
                means.append(sum(vals)/len(vals) if vals else 0.0)
            plt.bar(x + (i - len(sizes)/2)*width + width/2, means, width=width, label=f"{sz}px")
        plt.xticks(x, [_fmt_float_str(b) for b in betas])
        plt.xlabel("beta (style weight)")
        plt.ylabel("runtime (s, avg)")
        plt.title("Runtime vs beta (grouped by size)")
        plt.legend()
        out = cfg.outputs_plots_dir / "runtime_vs_beta_by_size.png"
        plt.tight_layout(); plt.savefig(out); plt.close()
        print(f"[Analysis] Saved: {out}")

    # Runtime per step summaries
    rps_by_size = defaultdict(list)
    rps_by_beta = defaultdict(list)
    for r in rows:
        if r["wall_time_s"] is not None and r["steps"] not in (None, 0):
            per_step = r["wall_time_s"] / max(int(r["steps"]), 1)
            if r["size"] is not None:
                rps_by_size[r["size"]].append(per_step)
            if r["beta"] is not None:
                rps_by_beta[r["beta"]].append(per_step)
    if rps_by_size:
        plt.figure(figsize=(6, 4))
        sizes = sorted(rps_by_size.keys())
        means = [sum(v)/len(v) for v in (rps_by_size[s] for s in sizes)]
        plt.bar([str(s) for s in sizes], means)
        plt.xlabel("image size (px)")
        plt.ylabel("runtime per step (s, avg)")
        plt.title("Runtime per step vs size")
        out = cfg.outputs_plots_dir / "runtime_per_step_vs_size.png"
        plt.tight_layout(); plt.savefig(out); plt.close(); print(f"[Analysis] Saved: {out}")
    if rps_by_beta:
        plt.figure(figsize=(6, 4))
        betas = sorted(rps_by_beta.keys())
        means = [sum(v)/len(v) for v in (rps_by_beta[b] for b in betas)]
        plt.bar([_fmt_float_str(b) for b in betas], means)
        plt.xlabel("beta (style weight)")
        plt.ylabel("runtime per step (s, avg)")
        plt.title("Runtime per step vs beta")
        out = cfg.outputs_plots_dir / "runtime_per_step_vs_beta.png"
        plt.tight_layout(); plt.savefig(out); plt.close(); print(f"[Analysis] Saved: {out}")

    # SSIM boxplots by beta for each size
    ssim_values = defaultdict(list)  # key: (size, beta) -> list of ssim
    sizes_all, betas_all = set(), set()
    for r in rows:
        if r["ssim"] is not None and r["size"] is not None and r["beta"] is not None:
            ssim_values[(r["size"], r["beta"])].append(r["ssim"])
            sizes_all.add(r["size"]); betas_all.add(r["beta"])
    if ssim_values:
        import numpy as np
        betas = sorted(betas_all)
        for sz in sorted(sizes_all):
            data = [ssim_values.get((sz, b), []) for b in betas]
            if not any(len(d) for d in data):
                continue
            plt.figure(figsize=(8, 4.5))
            plt.boxplot([d if d else [None] for d in data], labels=[_fmt_float_str(b) for b in betas], showmeans=True)
            plt.xlabel("beta (style weight)")
            plt.ylabel("SSIM vs content")
            plt.title(f"SSIM distribution by beta — {sz}px")
            out = cfg.outputs_plots_dir / f"ssim_box_by_beta_{sz}px.png"
            plt.tight_layout(); plt.savefig(out); plt.close(); print(f"[Analysis] Saved: {out}")

    # SSIM vs beta scatter across pairs (color by size)
    try:
        import numpy as np
        import matplotlib.cm as cm
        sizes = sorted(set(r["size"] for r in rows if r["size"] is not None))
        if sizes:
            color_map = {s: cm.tab10(i % 10) for i, s in enumerate(sizes)}
            plt.figure(figsize=(7, 5))
            for r in rows:
                if r["ssim"] is None or r["beta"] is None or r["size"] is None:
                    continue
                jitter = (np.random.rand() - 0.5) * 0.05
                plt.scatter(r["beta"] + jitter, r["ssim"], s=18, alpha=0.6, color=color_map[r["size"]])
            handles = [plt.Line2D([0],[0], marker='o', color='w', label=f"{s}px", markerfacecolor=color_map[s], markersize=6) for s in sizes]
            plt.legend(handles=handles, title="size")
            plt.xlabel("beta (style weight)")
            plt.ylabel("SSIM vs content")
            plt.title("SSIM vs beta (scatter across pairs)")
            out = cfg.outputs_plots_dir / "ssim_vs_beta_scatter_by_pair.png"
            plt.tight_layout(); plt.savefig(out); plt.close(); print(f"[Analysis] Saved: {out}")
    except Exception as e:
        print(f"[Analysis] Scatter plot skipped: {e}")

    # Summary CSV: top runs by SSIM
    try:
        top = sorted([r for r in rows if r["ssim"] is not None], key=lambda x: x["ssim"], reverse=True)[:20]
        out_csv = cfg.metrics_dir / "summary_top_by_ssim.csv"
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["rank","content","style","size","beta","alpha","gamma","lr","steps","ssim","runtime_s"])
            for i, r in enumerate(top, 1):
                w.writerow([i, r["content"], r["style"], r["size"], r["beta"], r["alpha"], r["gamma"], r["lr"], r["steps"], f"{r['ssim']:.6f}", f"{(r['wall_time_s'] or 0.0):.3f}"])
        print(f"[Analysis] Saved summary: {out_csv}")
    except Exception as e:
        print(f"[Analysis] Failed to write summary: {e}")

def main() -> int:
    # Small editable config block
    cfg = Config(
        # You can tweak these inline or wire up arg parsing later
        image_size=256,
        steps=50,
        alpha=1.0,
        beta=1e3,
        gamma=10.0,
        seed=42,
        resize_policy="square-resize",  # or "center-crop-then-resize"
    )

    # Print config at the start of the run
    print_config(cfg)

    # Ensure folders exist (creates empty directories)
    ensure_dirs(cfg)
    print("Folders ensured: content/style/outputs/metrics are present.")

    # Reproducibility
    set_global_seed(cfg.seed)

    # Verify TensorFlow import and CPU-only mode
    report = tf_cpu_only_report()
    print(report)
    if report.startswith("TensorFlow not installed"):
        # Fail clearly if TF is missing, since NST depends on it
        print("ERROR: TensorFlow is required for NST. Install CPU build and re-run.")
        return 1

    # Announce chosen resize policy
    if cfg.resize_policy not in ("square-resize", "center-crop-then-resize"):
        print(
            f"WARNING: Unknown resize_policy '{cfg.resize_policy}'. Defaulting to 'square-resize'."
        )

    print(f"Resize policy: {cfg.resize_policy}")

    # Step 2: Image I/O & Preprocessing (deterministic) round-trip
    verify_image_io_roundtrip(cfg)

    print("Step 2 complete: loader/saver implemented and verified (if sample images present).")

    # Step 3: Encoder (VGG-19) feature taps, frozen
    verify_vgg_extractor(cfg)
    print("Step 3 complete: VGG feature extractor built, frozen, and verified.")

    # Step 4: Style representation via Gram matrices (normalized)
    verify_gram_matrices(cfg)
    print("Step 4 complete: Gram matrices computed, shapes verified, determinism checked.")

    # Step 5 & 6: Loss functions and optimization loop
    run_style_transfer_step6(cfg)
    print("Step 5&6 complete: losses logged; checkpoints and final image saved.")

    # Optional: Steps 9–12 based on config toggles
    if cfg.do_sanity_tests:
        run_sanity_tests(cfg)
    if cfg.do_batch_grid:
        run_batch_grid(cfg)
    if cfg.do_portfolio_gallery:
        run_portfolio_and_gallery(cfg)
    if cfg.do_analysis_plots:
        analysis_plots_from_csv(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
