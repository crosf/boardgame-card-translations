"""Вспомогательные функции для отрисовки перевода поверх изображения карты.

Первый вариант рендера больше не закрывает текст белыми прямоугольниками. Теперь
он делает результат ближе к примеру с карточкой: размывает увеличенный фрагмент
фона под старым текстом, вставляет его обратно в область OCR-бокса и рисует
перевод светлым цветом с тенью.

Это всё ещё не полноценное восстановление фона, но такой подход лучше подходит
для красных плашек, текстурных панелей и тёмных текстовых зон, чем простая белая
заливка. Позже этот слой можно заменить на LaMa/FLUX для настоящего удаления
оригинального текста.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Порядок поиска шрифта для русского текста.
# 1. Сначала используем шрифты из проекта, чтобы результат был воспроизводимым.
# 2. Затем пробуем стандартные Windows-шрифты с поддержкой кириллицы.
# 3. Если ничего не найдено, _load_font() использует дефолтный шрифт Pillow.
DEFAULT_FONT_CANDIDATES = [
    Path("fonts/arialbd.ttf"),
    Path("fonts/arial.ttf"),
    Path("fonts/Arial.ttf"),
    Path("C:/Windows/Fonts/arialbd.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/georgia.ttf"),
]

# Стиль текста по умолчанию подобран под пример: светлые буквы на красной или
# тёмной области, лёгкая тень и центрирование внутри исходного OCR-бокса.
TEXT_FILL = (255, 239, 205)
TEXT_SHADOW = (72, 35, 24)
TEXT_STROKE = (105, 49, 33)


def _box_bounds(box):
    """Преобразовать четырёхточечный OCR-полигон в прямоугольные границы."""
    # PaddleOCR возвращает четыре угловые точки. Операции рисования в Pillow
    # проще выполнять с прямоугольником, поэтому берём минимумы и максимумы.
    points = np.array(box, dtype=float)
    min_x = int(points[:, 0].min())
    min_y = int(points[:, 1].min())
    max_x = int(points[:, 0].max())
    max_y = int(points[:, 1].max())
    return min_x, min_y, max_x, max_y


def _load_font(size):
    """Загрузить первый доступный шрифт формата TrueType нужного размера."""
    for font_path in DEFAULT_FONT_CANDIDATES:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)

    # Этот резервный вариант позволяет скрипту запускаться даже без файла шрифта,
    # но кириллица может отображаться неправильно. README рекомендует добавить TTF.
    return ImageFont.load_default()


def _wrap_text(draw, text, font, max_width):
    """Разбить текст на строки так, чтобы каждая строка помещалась по ширине."""
    words = text.split()
    if not words:
        return text

    lines = []
    current_line = words[0]

    for word in words[1:]:
        candidate = f"{current_line} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line = candidate
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)
    return "\n".join(lines)


def _fit_wrapped_font(draw, text, max_width, max_height, max_size=58, min_size=12):
    """Подобрать самый крупный шрифт и перенос строк для заданного прямоугольника."""
    for size in range(max_size, min_size - 1, -1):
        font = _load_font(size)
        wrapped_text = _wrap_text(draw, text, font, max_width)
        bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=6)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        if width <= max_width and height <= max_height:
            return font, wrapped_text

    font = _load_font(min_size)
    return font, _wrap_text(draw, text, font, max_width)


def _expanded_bounds(bounds, image_size, padding):
    """Расширить прямоугольник, не выходя за границы изображения."""
    min_x, min_y, max_x, max_y = bounds
    image_width, image_height = image_size
    return (
        max(min_x - padding, 0),
        max(min_y - padding, 0),
        min(max_x + padding, image_width),
        min(max_y + padding, image_height),
    )


def _erase_text_with_blurred_background(image, bounds):
    """Скрыть старый текст размытым фрагментом окружающего фона."""
    min_x, min_y, max_x, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y

    # Чем крупнее текстовый блок, тем сильнее размытие. Это помогает убрать
    # контрастные буквы, но сохранить общий цвет и текстуру плашки карты.
    padding = max(8, min(width, height) // 2)
    blur_radius = max(6, min(width, height) // 3)
    sample_bounds = _expanded_bounds(bounds, image.size, padding)

    sample = image.crop(sample_bounds).filter(ImageFilter.GaussianBlur(blur_radius))
    sample_center = sample.crop((
        min_x - sample_bounds[0],
        min_y - sample_bounds[1],
        max_x - sample_bounds[0],
        max_y - sample_bounds[1],
    ))
    image.paste(sample_center, (min_x, min_y))


def _draw_translated_text(draw, bounds, text):
    """Нарисовать перевод с переносом строк, светлой заливкой и мягкой тенью."""
    min_x, min_y, max_x, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y
    horizontal_padding = max(4, width // 18)
    vertical_padding = max(2, height // 10)
    available_width = max(width - horizontal_padding * 2, 1)
    available_height = max(height - vertical_padding * 2, 1)

    font, wrapped_text = _fit_wrapped_font(
        draw,
        text,
        available_width,
        available_height,
    )
    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=6, stroke_width=1)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = min_x + (width - text_width) // 2
    text_y = min_y + (height - text_height) // 2

    # Тень рисуется отдельным слоем через смещение. Это создаёт эффект, похожий
    # на пример: светлый текст лучше читается на красной текстурной плашке.
    shadow_offset = max(2, font.size // 14) if hasattr(font, "size") else 2
    draw.multiline_text(
        (text_x + shadow_offset, text_y + shadow_offset),
        wrapped_text,
        fill=TEXT_SHADOW,
        font=font,
        spacing=6,
        align="center",
        stroke_width=1,
        stroke_fill=TEXT_SHADOW,
    )
    draw.multiline_text(
        (text_x, text_y),
        wrapped_text,
        fill=TEXT_FILL,
        font=font,
        spacing=6,
        align="center",
        stroke_width=1,
        stroke_fill=TEXT_STROKE,
    )


def render_translations(image_path, blocks, output_path):
    """Скрыть старый текст и нарисовать перевод в OCR-боксах."""
    # Переводим изображение в RGB, чтобы рисование и сохранение одинаково
    # работали для PNG и JPEG.
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for block in blocks:
        # Преобразуем координаты OCR-полигона в обычный прямоугольник.
        bounds = _box_bounds(block["box"])

        # Вместо белой заливки используем размытый фрагмент исходной области,
        # чтобы результат был ближе к текстурным плашкам из примера пользователя.
        _erase_text_with_blurred_background(image, bounds)

        # Модуль отрисовки принимает как уже переведённые блоки, так и сырые
        # OCR-блоки. Если перевода нет, он безопасно нарисует исходный текст.
        translated_text = block.get("translation", block["text"])
        _draw_translated_text(draw, bounds, translated_text)

    # Создаём папку для результата перед сохранением итогового изображения.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
