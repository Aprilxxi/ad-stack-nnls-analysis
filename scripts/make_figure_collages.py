# %%
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\arialbd.ttf"),
    Path(r"C:\Windows\Fonts\Arialbd.ttf"),
    Path(r"C:\Windows\Fonts\calibrib.ttf"),
    Path(r"C:\Windows\Fonts\segoeuib.ttf"),
]

BACKGROUND = (255, 255, 255, 255)
OUTPUT_DPI = (600, 600)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(str(font_path), size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit_size(src_size: tuple[int, int], dst_size: tuple[int, int]) -> tuple[int, int]:
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    scale = min(dst_w / src_w, dst_h / src_h)
    return max(1, round(src_w * scale)), max(1, round(src_h * scale))


ImageSpec = str | tuple[str, tuple[int, int, int, int]] | tuple[str, tuple[int, int, int, int], object]


def open_image_spec(figures_dir: Path, spec: ImageSpec) -> Image.Image:
    if isinstance(spec, str):
        return Image.open(figures_dir / spec).convert("RGBA")

    image_name = spec[0]
    crop_box = spec[1]
    clean_boxes = spec[2] if len(spec) == 3 else (0, 0, 240, 180)
    image = Image.open(figures_dir / image_name).convert("RGBA")
    try:
        cropped = image.crop(crop_box)
    finally:
        image.close()

    # Remove the panel letters already embedded in shap_combined.png before relabeling.
    draw = ImageDraw.Draw(cropped, "RGBA")
    if isinstance(clean_boxes, list):
        for clean_box in clean_boxes:
            draw.rectangle(clean_box, fill=BACKGROUND)
    else:
        draw.rectangle(clean_boxes, fill=BACKGROUND)
    return cropped


def save_collage_outputs(canvas: Image.Image, output_path: Path, dpi: tuple[int, int]) -> None:
    rgb_canvas = canvas.convert("RGB")
    rgb_canvas.save(output_path, quality=95, dpi=dpi)
    rgb_canvas.save(output_path.with_suffix(".pdf"), "PDF", resolution=float(dpi[0]))
    rgb_canvas.save(output_path.with_suffix(".tif"), dpi=dpi)


def build_collage(
    figures_dir: Path,
    output_name: str,
    image_names: list[ImageSpec],
    labels: list[str],
    cols: int,
    rows: int,
    square_cells: bool = False,
    cell_size: tuple[int, int] | None = None,
    gap: int = 36,
    outer: int = 40,
) -> Path:
    images = [open_image_spec(figures_dir, name) for name in image_names]
    try:
        if cell_size is not None:
            cell_w, cell_h = cell_size
        elif square_cells:
            cell_size = max(max(image.width, image.height) for image in images)
            cell_w = cell_size
            cell_h = cell_size
        else:
            cell_w = max(image.width for image in images)
            cell_h = max(image.height for image in images)

        canvas_w = outer * 2 + cols * cell_w + (cols - 1) * gap
        canvas_h = outer * 2 + rows * cell_h + (rows - 1) * gap
        canvas = Image.new("RGBA", (canvas_w, canvas_h), BACKGROUND)
        font = load_font(max(88, round(cell_h * 0.07)))
        draw = ImageDraw.Draw(canvas, "RGBA")

        for index, (image, label) in enumerate(zip(images, labels)):
            row = index // cols
            col = index % cols
            cell_x = outer + col * (cell_w + gap)
            cell_y = outer + row * (cell_h + gap)

            new_w, new_h = fit_size(image.size, (cell_w, cell_h))
            resized = image.resize((new_w, new_h), Image.LANCZOS)
            paste_x = cell_x + (cell_w - new_w) // 2
            paste_y = cell_y + (cell_h - new_h) // 2
            canvas.alpha_composite(resized, (paste_x, paste_y))
            draw.text((cell_x + 18, cell_y + 6), label, font=font, fill=(0, 0, 0, 255))

        output_path = figures_dir / output_name
        save_collage_outputs(canvas, output_path, dpi=OUTPUT_DPI)
        return output_path
    finally:
        for image in images:
            image.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the figure collages used for the manuscript."
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
        help="Directory that contains the source PNG files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures_dir = args.figures_dir.resolve()

    if not figures_dir.exists():
        raise FileNotFoundError(f"Figures directory not found: {figures_dir}")

    outputs = [
        build_collage(
            figures_dir=figures_dir,
            output_name="combined_2x2_abcd.png",
            image_names=[
                "roc_ci.png",
                "pr_ci.png",
                "external_roc_ci_bootstrap_bca.png",
                "external_pr_ci_bootstrap_bca.png",
            ],
            labels=["A", "B", "C", "D"],
            cols=2,
            rows=2,
            square_cells=True,
        ),
        build_collage(
            figures_dir=figures_dir,
            output_name="combined_1x3_abc.png",
            image_names=[
                "decision_curve.png",
                "external_decision_curve_bootstrap_bca.png",
                "shap_bar.png",
                "shap_beeswarm.png",
            ],
            labels=["A", "B", "C", "D"],
            cols=2,
            rows=2,
            square_cells=False,
            cell_size=(2128, 1651),
        ),
    ]

    for output in outputs:
        print(output)
        print(output.with_suffix(".pdf"))
        print(output.with_suffix(".tif"))


if __name__ == "__main__":
    main()
