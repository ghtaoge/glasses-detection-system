import io

from PIL import Image, ImageDraw, ImageFont

from app.domain.inference import Detection

COLORS = {"no_glasses": "#34a853", "eyeglasses": "#2d74da", "sunglasses": "#dc5c3a"}


def render_detections(image: Image.Image, detections: tuple[Detection, ...]) -> bytes:
    rendered = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()
    line_width = max(2, min(rendered.size) // 250)
    for item in detections:
        color = COLORS[item.class_name.value]
        box = (item.box.x1, item.box.y1, item.box.x2, item.box.y2)
        draw.rectangle(box, outline=color, width=line_width)
        label = f"{item.class_name.value} {item.confidence:.0%}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        label_w, label_h = right - left + 8, bottom - top + 6
        label_y = max(0, item.box.y1 - label_h)
        draw.rectangle((item.box.x1, label_y, item.box.x1 + label_w, label_y + label_h), fill=color)
        draw.text((item.box.x1 + 4, label_y + 3), label, fill="white", font=font)
    output = io.BytesIO()
    rendered.save(output, "JPEG", quality=90, optimize=True)
    return output.getvalue()
