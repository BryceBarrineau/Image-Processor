# Image Classification Processor

A desktop GUI tool for reviewing and classifying images for ML training datasets.

## Features

- **Modern card-based UI**: Each class is a card on the left with a color-coded progress bar
  - 🟢 Green = kept
  - 🔴 Red = deleted
  - 🟠 Orange = moved
  - ⚪ Gray = pending
- **MS Photos-like image viewer**: scroll to zoom toward your cursor, drag to pan when zoomed in, double-click to toggle fit/100%
- **Class management**: rename classes with the ✏ button, create new classes with "+ New Class"
- **Keyboard-driven workflow**: `A`/`D` navigate, `Enter` keep, `X` delete, `C` move, `Ctrl+Z` undo
- **Class filtering**: click a class card to work only on that class; click "All Classes" for everything
- **Progress tracking**: per-class progress bars + overall counts
- **History log**: color-coded action history on the right
- **Gamification**: keep streaks 🔥, score 🎯, model accuracy % 📈, and levels 🏆
- **Session persistence**: save/resume your progress (Ctrl+S, auto-prompt on close)
- **Safe processing**: original images are never modified; cleaned dataset is copied to `processed_output/`

## Gamification

A slim strip under the action bar keeps you motivated while labeling:

| Element | What it does |
|---------|--------------|
| 🔥 Streak | Consecutive "Keep"s in a row (your model trusted you that many times!) |
| 🏆 Best | Your longest streak ever this session |
| 📈 Model ACC | Kept ÷ processed — how often the model predictions are correct |
| 🎯 Score | +10 keep, +5 move, +2 delete (undo reverses it) |
| 🏅 Level | Titles like "Data Tagger" → "Label Lord" → "ML Zen Master" based on decisions |

Milestone toasts appear in the status bar: streak benchmarks (5/10/25/50/100),
accuracy badges (60/75/85/90/95/99%), and level-ups. Everything is cosmetic —
nothing slows down the keyboard workflow.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
   ```bash
   python image_processor.py
   ```

2. Drag & drop your image folder onto the window, or click "📁 Open Folder"

3. The folder structure should be:
   ```
   your_folder/
   ├── class_a/
   │   ├── image1.bmp
   │   ├── image2.bmp
   │   └── ...
   ├── class_b/
   │   ├── image3.bmp
   │   └── ...
   └── class_c/
       └── ...
   ```

4. Use keyboard shortcuts to review images

## Features in Detail

### Class Cards & Filtering
- **Click a class card** to work only on that class (nav cycles within it)
- **Click "All Classes"** to see everything
- Each card shows a **multi-segment progress bar**:
  - 🟢 green = kept, 🔴 red = deleted, 🟠 orange = moved, ⚪ gray = pending
- **Rename** a class with its ✏ button (renames the folder too)
- **"+ New Class"** creates a new folder you can move images into

### Zoom & Pan (MS Photos style)
- **Scroll wheel** zooms in/out toward the cursor — no Ctrl needed
- **Click & drag** to pan when zoomed in
- **Double-click** toggles between fit-to-window and 100%
- `F` = fit to view, `+`/`-` = zoom, `0` = 100%

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `A` or `←` | Previous image (respects class filter) |
| `D` or `→` | Next image (respects class filter) |
| `Enter` | Keep image (copy to same class in output) |
| `X` | Delete image (skip, don't copy to output) |
| `C` | Move to another class (opens popup) |
| `Ctrl+Z` | Undo last action |
| `F` | Fit image to view |
| `+` / `-` | Zoom in / out |
| `0` | Reset zoom to 100% |
| `Ctrl+S` | Save session |

## Output

When you click "✔ Export", the tool creates:

```
your_folder/
├── processed_output/
│   ├── class_a/          # Kept images from class_a
│   ├── class_b/          # Kept images from class_b
│   ├── target_class/     # Images moved to different classes
│   └── _deleted/         # Deleted images (recoverable)
```

- **Kept images**: Moved from source → `processed_output/<original_class>/`
- **Deleted images**: Moved to `processed_output/_deleted/` (recoverable, never destroyed)
- **Moved images**: Moved from source → `processed_output/<target_class>/`

> **Files move immediately** on each action — so if you quit and restart, processed images are already gone from the source folders. A progress file (`image_processor_session.json`) auto-saves after every action, so you never lose track.

## Tips

- Double-click the image to fit it to the view
- Use scroll wheel to zoom in/out toward cursor
- The history log on the right shows all your actions with color coding
- Session is automatically saved when you close the app (you'll be prompted)
- **Progress auto-saves after every action** — quit any time and resume exactly where you left off
- Files are **moved immediately**, not copied — so reopening shows only unprocessed images
- Deleted images go to `_deleted/` — you can recover them manually if needed
