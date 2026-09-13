# Image Classification Processor

A desktop GUI tool for reviewing and classifying images for ML training datasets.

## Features

- **Drag & Drop**: Simply drag your image folder onto the application window
- **Keyboard Navigation**: Fast keyboard-driven workflow
- **Large Image Preview**: Auto-fits images to screen with zoom capability
- **Progress Tracking**: See statistics and progress bar for your review session
- **Undo Support**: Ctrl+Z to undo the last action
- **Session Persistence**: Auto-saves progress, resume where you left off
- **Safe Processing**: Original images are never modified; processed images are copied to a new `processed_output` folder

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
   ```bash
   python image_processor.py
   ```

2. Drag & drop your image folder onto the drop zone, or click "Browse for Folder"

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

### Class Filtering
- **Dropdown menu** at the top lets you filter by class
- Select "All Classes" to see all images, or pick a specific class
- Navigation (A/D) respects the current filter - cycles only through selected class
- Statistics and progress bar update to show filtered results
- **Prominent class badge** displays the current image's class with color coding

### Mouse-Position Zooming
- **Ctrl + Scroll wheel** to zoom
- Zooming focuses on where your mouse cursor is positioned
- Smooth scroll-to-zoom experience for detailed inspection
- Double-click the image to fit it to the view

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `A` or `←` | Previous image (respects class filter) |
| `D` or `→` | Next image (respects class filter) |
| `Enter` | Keep image (copy to same class in output) |
| `X` | Delete image (skip, don't copy to output) |
| `C` | Change class (opens popup to select new class) |
| `Ctrl+Z` | Undo last action |
| `F` | Fit image to view |
| `Ctrl + Scroll` | Zoom in/out (toward mouse position) |
| `+` / `-` | Zoom in / out |
| `0` | Reset zoom to 100% |
| `Ctrl+S` | Save session |

## Output

When you click "Apply Changes & Export", the tool creates:

```
your_folder/
├── processed_output/
│   ├── class_a/          # Kept images from class_a
│   ├── class_b/          # Kept images from class_b
│   └── target_class/     # Images moved to different classes
```

- **Kept images**: Copied to their original class folder
- **Deleted images**: Not copied (excluded from output)
- **Moved images**: Copied to their new target class folder

## Tips

- Double-click the image to fit it to the view
- Use Ctrl+scroll wheel to zoom in/out
- The history log on the right shows all your actions with color coding
- Session is automatically saved when you close the app (you'll be prompted)
