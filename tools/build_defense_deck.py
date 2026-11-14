#!/usr/bin/env python3
"""Build the CompactFactExtraction defense deck from the checked-in numbers.

Requires python-pptx. The deck is intentionally text-first: all claims and
caveats from docs/SLIDES.md and docs/SPEECH.md are on the slides or in notes.
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

OUT = Path(__file__).resolve().parents[1] / "docs" / "defense-deck.pptx"

W, H = Inches(13.333), Inches(7.5)
BG = RGBColor(0x0B, 0x12, 0x20)
PANEL = RGBColor(0x13, 0x1C, 0x2E)
PANEL_2 = RGBColor(0x1B, 0x27, 0x3D)
TEXT = RGBColor(0xE2, 0xE8, 0xF0)
MUTED = RGBColor(0x94, 0xA3, 0xB8)
ACCENT = RGBColor(0x5E, 0xEA, 0xD4)
GREEN = RGBColor(0x34, 0xD3, 0x99)
WARN = RGBColor(0xFB, 0xBF, 0x24)
RED = RGBColor(0xF8, 0x71, 0x71)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Arial"
MONO = "Consolas"


def fill(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color


def no_line(shape):
    shape.line.fill.background()


def textbox(slide, x, y, w, h, text="", size=18, color=TEXT, bold=False,
            font=FONT, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.0):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = spacing
    run = p.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def rect(slide, x, y, w, h, color, radius=True):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, x, y, w, h)
    fill(shape, color)
    no_line(shape)
    if radius:
        shape.adjustments[0] = 0.08
    return shape


def background(slide, number, total=10):
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG
    textbox(slide, Inches(0.6), Inches(0.3), Inches(7.0), Inches(0.25),
            "COMPACTFACTEXTRACTION  /  ORGANIZER PACKAGE 2.1", 9, MUTED)
    textbox(slide, Inches(11.7), Inches(7.08), Inches(1.0), Inches(0.25),
            f"{number:02d} / {total:02d}", 9, MUTED, align=PP_ALIGN.RIGHT)


def title(slide, text, kicker=None, size=30):
    y = Inches(0.62)
    if kicker:
        textbox(slide, Inches(0.6), Inches(0.58), Inches(12.0), Inches(0.3),
                kicker.upper(), 10, ACCENT, bold=True)
        y = Inches(0.92)
    textbox(slide, Inches(0.6), y, Inches(12.1), Inches(0.8), text, size,
            WHITE, bold=True)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6),
                                  Inches(1.75), Inches(1.1), Pt(3))
    fill(line, ACCENT)
    no_line(line)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def bullets(slide, x, y, w, h, items, size=18, color=TEXT, gap=10):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    first = True
    for item in items:
        if isinstance(item, tuple):
            text, level = item
        else:
            text, level = item, 0
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level
        p.space_after = Pt(gap)
        p.line_spacing = 1.12
        run = p.add_run()
        run.text = ("•  " if level == 0 else "–  ") + text
        run.font.name = FONT
        run.font.size = Pt(size - 2 * level)
        run.font.color.rgb = color if level == 0 else MUTED
    return box


def kpi(slide, x, y, w, h, label, value, sub="", color=ACCENT,
        value_size=26):
    rect(slide, x, y, w, h, PANEL)
    textbox(slide, x + Inches(0.22), y + Inches(0.16), w - Inches(0.44),
            Inches(0.25), label.upper(), 9, MUTED, bold=True)
    textbox(slide, x + Inches(0.22), y + Inches(0.5), w - Inches(0.44),
            Inches(0.55), value, value_size, color, bold=True)
    if sub:
        textbox(slide, x + Inches(0.22), y + Inches(1.12), w - Inches(0.44),
                Inches(0.3), sub, 10, MUTED)


def table(slide, x, y, w, data, col_widths, row_h=0.42, font_size=12,
          header_size=10, highlight_col=None, highlight_color=ACCENT,
          mono=True):
    rows, cols = len(data), len(data[0])
    shape = slide.shapes.add_table(rows, cols, x, y, w,
                                   Inches(row_h * rows))
    tbl = shape.table
    tbl.first_row = True
    tbl.horz_banding = False
    for idx, cw in enumerate(col_widths):
        tbl.columns[idx].width = Inches(cw)
    for r in range(rows):
        tbl.rows[r].height = Inches(row_h)
        for c in range(cols):
            cell = tbl.cell(r, c)
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.04)
            cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            fill(cell, PANEL_2 if r == 0 else PANEL)
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            run = p.add_run()
            run.text = str(data[r][c])
            run.font.name = MONO if (mono and r > 0 and c > 0) else FONT
            run.font.size = Pt(header_size if r == 0 else font_size)
            run.font.bold = r == 0 or (highlight_col is not None and
                                       c == highlight_col)
            run.font.color.rgb = (MUTED if r == 0 else
                                  highlight_color if highlight_col == c
                                  else TEXT)
    return tbl


def build():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    blank = prs.slide_layouts[6]

    # 1. Title ------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    background(s, 1)
    rect(s, Inches(0.6), Inches(1.25), Inches(0.16), Inches(3.6), ACCENT)
    textbox(s, Inches(1.0), Inches(1.2), Inches(11.6), Inches(0.4),
            "MWS HACKATHON  ·  DEFENSE", 12, ACCENT, bold=True)
    textbox(s, Inches(1.0), Inches(1.7), Inches(11.4), Inches(1.1),
            "CompactFactExtraction", 52, WHITE, bold=True)
    textbox(s, Inches(1.0), Inches(2.9), Inches(10.8), Inches(0.9),
            "Извлечение фактов из телефонных звонков на 0.6B", 26, TEXT)
    textbox(s, Inches(1.0), Inches(3.55), Inches(11.0), Inches(0.5),
            "QLoRA  ·  constrained JSON  ·  даты кодом", 16, MUTED)
    for i, (label, value, color) in enumerate([
        ("Модель", "0.6B", ACCENT),
        ("Формат", "100 %", GREEN),
        ("2.1: T / S", "0.4663 / 0.5152", WARN),
    ]):
        x = Inches(1.0 + i * 3.55)
        rect(s, x, Inches(4.65), Inches(3.25), Inches(1.05), PANEL)
        textbox(s, x + Inches(0.2), Inches(4.78), Inches(2.9), Inches(0.22),
                label.upper(), 9, MUTED, bold=True)
        textbox(s, x + Inches(0.2), Inches(5.04), Inches(2.9), Inches(0.45),
                value, 24, color, bold=True)
    textbox(s, Inches(1.0), Inches(6.55), Inches(11.5), Inches(0.3),
            "Защита · 16.11.2026  ·  числа 2.1 — на сохранённых предсказаниях 003",
            10, MUTED)
    notes(s, "Мы взяли Qwen3-0.6B, дообучили QLoRA и довели до уровня, где на "
             "новом пакете 2.1 держим даты и состояния в допуске 0.02. Сейчас "
             "покажу, как из модели, которая ломала JSON, получился рабочий "
             "экстрактор. Сразу оговорка: T/S — сохранённые предсказания 003, "
             "не свежий end-to-end.")

    # 2. Task -------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    background(s, 2)
    title(s, "Что требует кейс", "постановка")
    bullets(s, Inches(0.6), Inches(2.05), Inches(6.9), Inches(4.6), [
        ("Модель 0.5–1.5B, без внешних teacher-API", 0),
        ("Схема фактов: объект, атрибут, значение, полярность, valid_from/to, source", 0),
        ("Скрытый adversarial-срез: ASR-шум, перефразы, относительные даты", 0),
        ("Валидный JSON на каждом ответе; 100 % формата", 0),
        ("Не потерять больше 0.02 к baseline по T и S", 0),
    ], size=17, gap=14)
    kpi(s, Inches(7.85), Inches(2.05), Inches(2.45), Inches(1.55),
        "baseline 2.1 · T", "0.4706", "glm-5.3, ~10× больше", MUTED)
    kpi(s, Inches(10.5), Inches(2.05), Inches(2.45), Inches(1.55),
        "baseline 2.1 · S", "0.5345", "", MUTED)
    kpi(s, Inches(7.85), Inches(3.85), Inches(2.45), Inches(1.55),
        "цель · T", "≥ 0.4506", "допуск −0.02", ACCENT)
    kpi(s, Inches(10.5), Inches(3.85), Inches(2.45), Inches(1.55),
        "цель · S", "≥ 0.5145", "допуск −0.02", ACCENT)
    rect(s, Inches(7.85), Inches(5.65), Inches(5.1), Inches(0.85), PANEL)
    textbox(s, Inches(8.05), Inches(5.78), Inches(4.7), Inches(0.6),
            "Отсечение: T < 0.4206 — решение не рассматривают", 13, RED,
            bold=True)
    notes(s, "База — glm-5.3, примерно в десять раз больше. Нам нельзя "
             "потерять больше 0.02 по temporal и state; если temporal падает "
             "ниже 0.4206, решение вообще не рассматривают. Плюс скрытый "
             "adversarial с ASR-шумом и требование валидной схемы.")

    # 3. Starting point ---------------------------------------------------
    s = prs.slides.add_slide(blank)
    background(s, 3)
    title(s, "Старт: две разные проблемы", "точка отсчёта · пакет 1")
    kpi(s, Inches(0.6), Inches(2.05), Inches(2.55), Inches(1.6),
        "невалидный JSON", "22 / 383", "отказа", RED)
    kpi(s, Inches(3.35), Inches(2.05), Inches(2.55), Inches(1.6),
        "формат", "94 %", "", WARN)
    kpi(s, Inches(6.1), Inches(2.05), Inches(2.55), Inches(1.6),
        "temporal_recall", "0.3348", "отсечение 0.371", RED)
    kpi(s, Inches(8.85), Inches(2.05), Inches(2.55), Inches(1.6),
        "precision / recall", "0.347 / 0.368", "", MUTED, value_size=18)
    rect(s, Inches(0.6), Inches(4.15), Inches(5.85), Inches(2.2), PANEL)
    textbox(s, Inches(0.85), Inches(4.32), Inches(5.4), Inches(0.3),
            "ПРОБЛЕМА 1 · СИНТАКСИС", 10, ACCENT, bold=True)
    textbox(s, Inches(0.85), Inches(4.65), Inches(5.4), Inches(0.5),
            "Модель ломает JSON", 22, WHITE, bold=True)
    textbox(s, Inches(0.85), Inches(5.2), Inches(5.4), Inches(0.9),
            "Лечится декодированием: грамматика не даёт выйти за схему.", 14,
            MUTED)
    rect(s, Inches(6.75), Inches(4.15), Inches(5.85), Inches(2.2), PANEL)
    textbox(s, Inches(7.0), Inches(4.32), Inches(5.4), Inches(0.3),
            "ПРОБЛЕМА 2 · СЕМАНТИКА", 10, WARN, bold=True)
    textbox(s, Inches(7.0), Inches(4.65), Inches(5.4), Inches(0.5),
            "Даты не сходятся", 22, WHITE, bold=True)
    textbox(s, Inches(7.0), Inches(5.2), Inches(5.4), Inches(0.9),
            "Лечится кодом: проверка cue по реплике-источнику и календарь.", 14,
            MUTED)
    notes(s, "Первая модель ломала JSON в 22 звонках из 383 и давала temporal "
             "0.3348 при отсечении 0.371. Важно: это две разные проблемы. "
             "Формат лечится декодированием, даты — кодом. Мы не пытались "
             "решить всё одной моделью.")

    # 4. XGrammar ---------------------------------------------------------
    s = prs.slides.add_slide(blank)
    background(s, 4)
    title(s, "Решение 1: JSON по построению", "xgrammar")
    steps = [
        ("Facts.model_json_schema()", "схема кейса"),
        ("XGrammar", "компиляция в грамматику"),
        ("Разрешённые токены", "только корректный JSON"),
    ]
    for i, (head, sub) in enumerate(steps):
        x = Inches(0.6 + i * 4.15)
        rect(s, x, Inches(2.05), Inches(3.55), Inches(1.35), PANEL)
        textbox(s, x + Inches(0.2), Inches(2.22), Inches(3.15), Inches(0.45),
                head, 15, ACCENT if i == 1 else WHITE, bold=True)
        textbox(s, x + Inches(0.2), Inches(2.75), Inches(3.15), Inches(0.35),
                sub, 12, MUTED)
        if i < 2:
            textbox(s, x + Inches(3.62), Inches(2.45), Inches(0.45), Inches(0.4),
                    "→", 22, MUTED, bold=True)
    table(s, Inches(0.6), Inches(3.9), Inches(12.1), [
        ["", "до", "после"],
        ["отказы JSON", "22", "0"],
        ["валидный формат", "94 %", "100 %"],
        ["recall / state", "0.3675 / 0.3339", "0.3908 / 0.3467"],
    ], [4.3, 3.9, 3.9], row_h=0.48, font_size=15, header_size=11,
        highlight_col=2, highlight_color=GREEN)
    textbox(s, Inches(0.6), Inches(6.45), Inches(12.0), Inches(0.3),
            "recall / state — пакет 1; эффект формата (22→0, 94→100 %) "
            "переносится на 2.1", 10, MUTED)
    notes(s, "Мы компилируем JSON-схему Facts в грамматику XGrammar и на "
             "каждом шаге разрешаем только токены, которые продолжают "
             "корректный JSON. Это не постобработка: модель физически не может "
             "выдать битую структуру. Получили 0 отказов и 100 % валидного "
             "формата. Recall/state в таблице — ещё пакет 1.")

    # 5. Training ---------------------------------------------------------
    s = prs.slides.add_slide(blank)
    background(s, 5)
    title(s, "Решение 2: три цикла по одной эпохе", "обучение · пакет 1")
    table(s, Inches(0.6), Inches(2.05), Inches(12.1), [
        ["Модель", "проходов", "T", "S", "P / R"],
        ["baseline-001", "1", "0.3315", "0.3467", "0.3625 / 0.3908"],
        ["continue-002", "2", "0.3566", "0.3852", "0.4008 / 0.4119"],
        ["continue-003", "3", "0.4377", "0.3949", "0.4149 / 0.4136"],
    ], [3.7, 1.7, 2.0, 2.0, 2.7], row_h=0.52, font_size=15, header_size=11,
        highlight_col=2, highlight_color=ACCENT, mono=True)
    rect(s, Inches(0.6), Inches(4.55), Inches(5.9), Inches(1.55), PANEL)
    textbox(s, Inches(0.85), Inches(4.72), Inches(5.4), Inches(0.28),
            "EVAL LOSS", 10, MUTED, bold=True)
    textbox(s, Inches(0.85), Inches(5.02), Inches(5.4), Inches(0.5),
            "0.1529 → 0.1345 → 0.1298", 23, GREEN, bold=True)
    textbox(s, Inches(0.85), Inches(5.58), Inches(5.4), Inches(0.3),
            "каждый цикл — новый оптимизатор, не один --epochs 3", 11, MUTED)
    rect(s, Inches(6.8), Inches(4.55), Inches(5.9), Inches(1.55), PANEL)
    textbox(s, Inches(7.05), Inches(4.72), Inches(5.4), Inches(0.28),
            "ЧТО ЭТО ЗНАЧИТ", 10, ACCENT, bold=True)
    textbox(s, Inches(7.05), Inches(5.02), Inches(5.4), Inches(0.95),
            "Каждая эпоха улучшала все метрики. Это пакет 1; на 2.1 "
            "абсолютные числа другие.", 14, TEXT)
    notes(s, "Одна эпоха — базовая. Вторая и третья — отдельные циклы по одной "
             "эпохе с новым оптимизатором, суммарно три прохода. Каждая эпоха "
             "улучшала все метрики, а eval_loss падал. Это пакет 1; на 2.1 "
             "абсолютные числа другие, но вывод про обучение сохраняется.")

    # 6. Dates ------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    background(s, 6)
    title(s, "Решение 3: даты считает код", "temporal normalization · task")
    chips = ["вчера", "N дней назад", "с пятницы", "через неделю", "позавчера"]
    for i, chip in enumerate(chips):
        x = Inches(0.6 + (i % 5) * 2.45)
        rect(s, x, Inches(2.0), Inches(2.25), Inches(0.6), PANEL_2)
        textbox(s, x, Inches(2.12), Inches(2.25), Inches(0.35), chip, 12,
                ACCENT, align=PP_ALIGN.CENTER, bold=True)
    bullets(s, Inches(0.6), Inches(2.95), Inches(7.3), Inches(3.4), [
        ("cue ищем в реплике-источнике, не во всём звонке", 0),
        ("action без времени → дата звонка", 0),
        ("agreement с будущим → дата события", 0),
        ("«принята» переносим внутри одной серии", 0),
        ("ASR-варианты cue: «рабате», «таявка», «изфиняюсь»", 0),
        ("исторические даты service_state/tariff не выводим", 0),
    ], size=15, gap=11)
    kpi(s, Inches(8.3), Inches(2.95), Inches(2.2), Inches(1.45),
        "T · до", "0.3566", "пакет 1", MUTED)
    kpi(s, Inches(10.7), Inches(2.95), Inches(2.2), Inches(1.45),
        "T · после", "0.4237", "пакет 1", GREEN)
    rect(s, Inches(8.3), Inches(4.7), Inches(4.6), Inches(1.6), PANEL)
    textbox(s, Inches(8.55), Inches(4.88), Inches(4.1), Inches(0.3),
            "ГЛАВНОЕ", 10, WARN, bold=True)
    textbox(s, Inches(8.55), Inches(5.2), Inches(4.1), Inches(0.95),
            "Модель пишет ISO-даты, но арифметику и источник проверяет код.",
            14, TEXT)
    notes(s, "Модель по-прежнему пишет ISO-даты, но код проверяет их по "
             "реплике-источнику: «вчера», «N дней назад», «с пятницы», «через "
             "неделю». Для action без времени берём дату звонка, для agreement "
             "с будущим временем — дату события, «принята» переносим внутри "
             "одной серии. Плюс распознаём типичные ASR-искажения cue.")

    # 7. v2.1 numbers -----------------------------------------------------
    s = prs.slides.add_slide(blank)
    background(s, 7)
    title(s, "Пакет 2.1: обе цели достигнуты", "результат · saved predictions 003")
    table(s, Inches(0.6), Inches(2.0), Inches(12.1), [
        ["метрика", "база 2.1", "003", "003 + recovery", "цель"],
        ["T", "0.4706", "0.4492", "0.4663", "≥ 0.4506"],
        ["S", "0.5345", "0.4350", "0.5152", "≥ 0.5145"],
        ["P / R", "0.3348 / 0.3805", "0.5003 / 0.4986", "0.4929 / 0.5279", "—"],
        ["G", "0.7835", "0.7141", "0.7023", "—"],
    ], [2.4, 2.5, 2.4, 2.7, 2.1], row_h=0.58, font_size=14, header_size=10,
        highlight_col=3, highlight_color=GREEN)
    rect(s, Inches(0.6), Inches(5.05), Inches(5.9), Inches(1.55), PANEL)
    textbox(s, Inches(0.85), Inches(5.2), Inches(5.4), Inches(0.3),
            "ЧТО ДАЁТ RECOVERY", 10, ACCENT, bold=True)
    textbox(s, Inches(0.85), Inches(5.5), Inches(5.4), Inches(0.9),
            "Добавляет только отсутствующий текущий тариф/статус: из каталога "
            "и явной фразы. Не перезаписывает модель.", 13, TEXT)
    rect(s, Inches(6.8), Inches(5.05), Inches(5.9), Inches(1.55), PANEL)
    textbox(s, Inches(7.05), Inches(5.2), Inches(5.4), Inches(0.3),
            "ЧЕСТНАЯ ОГОВОРКА", 10, WARN, bold=True)
    textbox(s, Inches(7.05), Inches(5.5), Inches(5.4), Inches(0.95),
            "Saved predictions 003. По половинам val: 0.4096/0.4907 и "
            "0.5216/0.5415. Свежий end-to-end 2.1 ещё не выполнен.", 12, TEXT)
    notes(s, "На 2.1 старые предсказания 003 без recovery дают 0.4492 и "
             "0.4350 — этого мало. Мы добавили детерминированное восстановление "
             "пропущенного текущего тарифа и статуса: только когда текущего "
             "факта нет, из каталога и явной фразы. Получили 0.4663 и 0.5152 — "
             "обе цели 2.1 достигнуты на сохранённых предсказаниях. Запас по "
             "state маленький; свежий end-to-end 2.1 ещё не выполнен. "
             "По половинам val разброс большой, это не гарантия скрытого среза.")

    # 8. Errors -----------------------------------------------------------
    s = prs.slides.add_slide(blank)
    background(s, 8)
    title(s, "Где ещё теряем", "ошибки · 2.1")
    textbox(s, Inches(0.6), Inches(1.95), Inches(5.9), Inches(0.4),
            "1 082 эталонных факта не совпали", 20, WHITE, bold=True)
    rows = [
        ("значение", 584, RED),
        ("факт не извлечён", 399, WARN),
        ("прочее", 95, MUTED),
        ("другой объект", 4, ACCENT),
        ("полярность", 0, GREEN),
    ]
    maxv = 584
    for i, (label, value, color) in enumerate(rows):
        y = Inches(2.55 + i * 0.72)
        textbox(s, Inches(0.6), y, Inches(2.4), Inches(0.3), label, 12, TEXT)
        if value:
            bar_w = Emu(int(Inches(3.0) * max(value / maxv, 0.012)))
            rect(s, Inches(3.05), y + Inches(0.04), bar_w, Inches(0.22), color,
                 radius=False)
        textbox(s, Inches(3.05 + 3.1), y, Inches(0.8), Inches(0.3),
                str(value), 12, color, bold=True)
    rect(s, Inches(7.1), Inches(2.0), Inches(5.6), Inches(2.55), PANEL)
    textbox(s, Inches(7.35), Inches(2.16), Inches(5.1), Inches(0.3),
            "ГЛАВНЫЙ РАЗРЫВ · TARIFF STATE", 10, WARN, bold=True)
    textbox(s, Inches(7.35), Inches(2.55), Inches(5.1), Inches(0.5),
            "0.415 → 0.764", 28, GREEN, bold=True)
    textbox(s, Inches(7.35), Inches(3.15), Inches(5.1), Inches(0.9),
            "56 звонков без текущего тарифа. Recovery закрывает разрыв, но "
            "общий state — 0.515.", 12, TEXT)
    rect(s, Inches(7.1), Inches(4.8), Inches(5.6), Inches(1.8), PANEL)
    textbox(s, Inches(7.35), Inches(4.96), Inches(5.1), Inches(0.3),
            "ЧТО НЕ ГЛАВНОЕ", 10, ACCENT, bold=True)
    textbox(s, Inches(7.35), Inches(5.3), Inches(5.1), Inches(1.0),
            "Объект — 4 ошибки, полярность — 0. Теряются формулировки значений "
            "и целые факты, а не структура.", 13, TEXT)
    notes(s, "Главная потеря — не объект и не полярность: объекта всего "
             "четыре, полярности ноль. Теряются формулировки значений и целые "
             "факты. Самый большой разрыв был в tariff state: модель вообще не "
             "предсказывала тариф в 56 звонках. Recovery поднимает tariff state "
             "с 0.415 до 0.764, но общий state — только 0.515, запас 0.0007.")

    # 9. Rejected ---------------------------------------------------------
    s = prs.slides.add_slide(blank)
    background(s, 9)
    title(s, "Что проверили и отклонили", "отрицательные результаты")
    table(s, Inches(0.6), Inches(2.05), Inches(12.1), [
        ["гипотеза", "результат"],
        ["Глобальный repetition penalty", "ломает качество и грамматику"],
        ["Фильтр длинных agreement", "precision растёт, temporal падает"],
        ["Массовое закрытие valid_to", "T −0.0059, S −0.0032"],
        ["Починка source по словам", "нестабильно на половинах val"],
        ["1.7B", "вне лимита 1.5B, нужно разрешение"],
        ["Синтетика учителем", "запрещена локально, не чинит даты"],
    ], [4.6, 7.5], row_h=0.5, font_size=13, header_size=10, mono=False)
    rect(s, Inches(0.6), Inches(5.75), Inches(12.1), Inches(0.8), PANEL)
    textbox(s, Inches(0.85), Inches(5.92), Inches(11.6), Inches(0.4),
            "Мы не только добавляли, но и измеряли и откатывали. "
            "Это часть решения, а не побочный результат.", 14, ACCENT, bold=True)
    notes(s, "Мы не только добавляли, но и измеряли и откатывали. Глобальный "
             "repetition penalty ломает качество, фильтр длинных agreement "
             "улучшает precision, но убивает temporal, массовое закрытие "
             "valid_to ухудшает обе метрики, починка source нестабильна. 1.7B "
             "выходит за лимит 1,5B и требует разрешения. Синтетика учителем "
             "запрещена в локальной валидации и не чинит исторические даты.")

    # 10. Demo ------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    background(s, 10)
    title(s, "Демо и следующие шаги", "финал")
    for i, (head, sub) in enumerate([
        ("1 · /health", "модель загружена, constrained_json=true"),
        ("2 · synthetic call", "«вчера», «сегодня», «в пятницу» → ISO"),
        ("3 · report.json", "0 отказов, 100 % формата, T 0.4663 (003+recovery)"),
    ]):
        y = Inches(2.0 + i * 0.95)
        rect(s, Inches(0.6), y, Inches(7.0), Inches(0.8), PANEL)
        textbox(s, Inches(0.8), y + Inches(0.1), Inches(2.2), Inches(0.35),
                head, 14, ACCENT, bold=True)
        textbox(s, Inches(3.0), y + Inches(0.12), Inches(4.4), Inches(0.55),
                sub, 12, TEXT)
    rect(s, Inches(0.6), Inches(5.0), Inches(7.0), Inches(1.55), PANEL)
    textbox(s, Inches(0.85), Inches(5.15), Inches(6.5), Inches(0.3),
            "ОГРАНИЧЕНИЯ", 10, WARN, bold=True)
    textbox(s, Inches(0.85), Inches(5.45), Inches(6.5), Inches(0.95),
            "Скрытый adversarial не проверен · state-запас 0.0007 · "
            "grounding ниже базы · шумный 2.1 не проверен end-to-end",
            12, TEXT)
    rect(s, Inches(7.95), Inches(2.0), Inches(4.75), Inches(4.55), PANEL)
    textbox(s, Inches(8.2), Inches(2.18), Inches(4.25), Inches(0.3),
            "ДАЛЬШЕ", 10, ACCENT, bold=True)
    bullets(s, Inches(8.2), Inches(2.55), Inches(4.25), Inches(3.8), [
        ("свежий end-to-end 2.1", 0),
        ("шумный val 2.1", 0),
        ("GPU batch 1 / 2 / 4", 0),
        ("Docker GPU / vLLM", 0),
        ("финальная модель — по полному val", 0),
    ], size=13, gap=13)
    textbox(s, Inches(8.2), Inches(5.95), Inches(4.2), Inches(0.4),
            "не фиксируем по одному удачному прогону", 10, MUTED)
    notes(s, "В демо показываем health, один синтетический звонок с "
             "«вчера/сегодня/в пятницу» и report.json: 0 отказов и 100 % "
             "формата. Полный val 2.1 на сохранённых предсказаниях даёт "
             "T 0.4663. Дальше — свежий end-to-end 2.1, ресурсная кривая "
             "batch1/2/4 и Docker. Ограничения: скрытый adversarial не "
             "проверен, state-запас маленький, grounding ниже базы. Поэтому "
             "финальную модель фиксируем по полному val и сохранённым "
             "предсказаниям, а не по одному удачному прогону.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    build()
