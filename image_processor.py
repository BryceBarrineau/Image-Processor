#!/usr/bin/env python3
"""
Image Classification Processor
A desktop GUI tool for reviewing and classifying images for ML training.
"""

import sys
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QSplitter,
    QDialog, QDialogButtonBox, QFileDialog, QMessageBox,
    QScrollArea, QFrame, QStatusBar, QProgressBar, QShortcut, QComboBox,
    QLineEdit, QInputDialog
)
from PyQt5.QtCore import Qt, QTimer, QMimeData, pyqtSignal, QSize
from PyQt5.QtGui import (
    QPixmap, QImage, QKeySequence, QWheelEvent, QDragEnterEvent,
    QDropEvent, QFont, QColor, QPalette, QPainter
)
from PIL import Image


# ============================================================================
# Data Classes and State Management
# ============================================================================

class ImageItem:
    """Represents a single image in the processing queue."""
    
    def __init__(self, path: Path, class_name: str):
        self.path = path  # current location (may change as files are moved)
        self.original_path = path  # where the file started (for undo)
        self.class_name = class_name
        self.status = "pending"  # pending, kept, deleted, moved
        self.original_class = class_name
        self.moved_to: Optional[str] = None
        
    def __repr__(self):
        return f"ImageItem({self.path.name}, class={self.class_name}, status={self.status})"


class HistoryEntry:
    """Represents a single action in the history log."""
    
    def __init__(self, image_name: str, action: str, from_class: str, to_class: Optional[str] = None):
        self.timestamp = datetime.now().strftime("%H:%M:%S")
        self.image_name = image_name
        self.action = action  # keep, delete, move
        self.from_class = from_class
        self.to_class = to_class
        
    def __str__(self):
        if self.action == "move":
            return f"[{self.timestamp}] {self.image_name}: {self.from_class} → {self.to_class}"
        elif self.action == "delete":
            return f"[{self.timestamp}] {self.image_name}: DELETED (was in {self.from_class})"
        else:
            return f"[{self.timestamp}] {self.image_name}: KEPT in {self.from_class}"


class ProcessingState:
    """Manages the complete state of the processing session."""
    
    def __init__(self):
        self.source_root: Optional[Path] = None
        self.output_root: Optional[Path] = None
        self.classes: List[str] = []
        self.images: List[ImageItem] = []
        self.current_index: int = 0
        self.history: List[HistoryEntry] = []
        self.undo_stack: List[Tuple[ImageItem, int, str]] = []  # (image, index, previous_status)
        
        # Statistics
        self.stats = {
            "pending": 0,
            "kept": 0,
            "deleted": 0,
            "moved": 0
        }
        
    def load_from_directory(self, source_path: Path) -> bool:
        """Load images from a directory structure."""
        self.source_root = source_path
        self.output_root = source_path / "processed_output"
        
        # Find all class folders
        self.classes = []
        self.images = []
        
        for item in sorted(source_path.iterdir()):
            if item.is_dir() and item.name != "processed_output":
                self.classes.append(item.name)
                # Load all .bmp files from this class
                for img_file in sorted(item.glob("*.bmp")):
                    self.images.append(ImageItem(img_file, item.name))
        
        self.current_index = 0
        self.update_stats()
        return len(self.classes) > 0
    
    def update_stats(self):
        """Update statistics counters."""
        self.stats = defaultdict(int)
        for img in self.images:
            self.stats[img.status] += 1
        self.stats = dict(self.stats)
    
    def get_current_image(self) -> Optional[ImageItem]:
        """Get the current image being reviewed."""
        if 0 <= self.current_index < len(self.images):
            return self.images[self.current_index]
        return None
    
    def get_next_pending_index(self, start: int = 0) -> int:
        """Find the next pending image starting from index."""
        for i in range(start, len(self.images)):
            if self.images[i].status == "pending":
                return i
        for i in range(0, start):
            if self.images[i].status == "pending":
                return i
        return -1  # No pending images
    
    def get_next_pending_index_in_class(self, class_name: str, start: int = 0) -> int:
        """Find the next pending image in a specific class."""
        for i in range(start, len(self.images)):
            if self.images[i].class_name == class_name and self.images[i].status == "pending":
                return i
        for i in range(0, start):
            if self.images[i].class_name == class_name and self.images[i].status == "pending":
                return i
        return -1
    
    def get_filtered_images(self, class_name: Optional[str] = None) -> List[ImageItem]:
        """Get images filtered by class name."""
        if not class_name:
            return self.images
        return [img for img in self.images if img.class_name == class_name]
    
    def get_class_stats(self, class_name: Optional[str] = None) -> Dict[str, int]:
        """Get statistics for a class. class_name None = all classes."""
        stats = {"kept": 0, "deleted": 0, "moved": 0, "pending": 0}
        for img in self.images:
            if class_name is None or img.class_name == class_name:
                if img.status in stats:
                    stats[img.status] += 1
        return stats
    
    def add_class(self, class_name: str) -> bool:
        """Create a new class folder. Returns True on success."""
        if not class_name or class_name in self.classes or not self.source_root:
            return False
        try:
            (self.source_root / class_name).mkdir()
        except OSError:
            return False
        self.classes.append(class_name)
        return True
    
    def rename_class(self, old_name: str, new_name: str) -> bool:
        """Rename a class folder and update all references. Returns True on success."""
        if old_name not in self.classes or new_name in self.classes or not self.source_root:
            return False
        old_dir = self.source_root / old_name
        new_dir = self.source_root / new_name
        try:
            old_dir.rename(new_dir)
        except OSError:
            return False
        self.classes[self.classes.index(old_name)] = new_name
        for img in self.images:
            if img.class_name == old_name:
                img.class_name = new_name
                img.path = new_dir / img.path.name
            if img.original_class == old_name:
                img.original_class = new_name
            if img.moved_to == old_name:
                img.moved_to = new_name
        return True
    
    def keep_image(self) -> bool:
        """Keep current image: move it into processed_output/<class>."""
        img = self.get_current_image()
        if not img or img.status != "pending":
            return False
        if not self.output_root:
            return False
        dest_dir = self.output_root / img.class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = self._safe_dest(dest_dir, img.path.name)
        if not self._move_file(img, dest):
            return False
        img.status = "kept"
        self.undo_stack.append((img, self.current_index, "pending"))
        self.history.append(HistoryEntry(img.path.name, "keep", img.class_name))
        self.update_stats()
        return True
    
    def delete_image(self) -> bool:
        """Delete current image: move it into processed_output/_deleted (recoverable)."""
        img = self.get_current_image()
        if not img or img.status != "pending":
            return False
        if not self.output_root:
            return False
        dest_dir = self.output_root / "_deleted"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = self._safe_dest(dest_dir, img.path.name)
        if not self._move_file(img, dest):
            return False
        img.status = "deleted"
        self.undo_stack.append((img, self.current_index, "pending"))
        self.history.append(HistoryEntry(img.path.name, "delete", img.class_name))
        self.update_stats()
        return True
    
    def move_image(self, target_class: str) -> bool:
        """Move current image to another class in processed_output/<target>."""
        img = self.get_current_image()
        if not img or img.status != "pending" or target_class == img.class_name:
            return False
        if target_class not in self.classes:
            return False
        if not self.output_root:
            return False
        dest_dir = self.output_root / target_class
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = self._safe_dest(dest_dir, img.path.name)
        if not self._move_file(img, dest):
            return False
        img.status = "moved"
        img.moved_to = target_class
        self.undo_stack.append((img, self.current_index, "pending"))
        self.history.append(HistoryEntry(img.path.name, "move", img.class_name, target_class))
        self.update_stats()
        return True
    
    # ------------------------------------------------------------- file mgmt
    def _safe_dest(self, dest_dir: Path, filename: str) -> Path:
        """Return a destination path, appending a number if it already exists."""
        dest = dest_dir / filename
        stem = dest.stem
        suffix = dest.suffix
        counter = 1
        while dest.exists():
            dest = dest_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        return dest
    
    def _move_file(self, img: ImageItem, dest: Path) -> bool:
        """Move the file backing an image to dest, updating its path. Safe if missing."""
        if not img.path.exists():
            # File already gone (e.g. user deleted it) - treat as already moved
            img.path = dest
            return True
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(img.path), str(dest))
            img.path = dest
            return True
        except OSError as e:
            print(f"File move error: {e}")
            return False
    
    def undo_last(self) -> bool:
        """Undo the last action, moving the file back to its original source folder."""
        if not self.undo_stack:
            return False
        img, idx, prev_status = self.undo_stack.pop()
        
        # Move the file back to its pre-action location (the source class folder)
        if img.status != "pending" and self.source_root:
            back_dir = self.source_root / img.original_class
            back_dir.mkdir(parents=True, exist_ok=True)
            back = self._safe_dest(back_dir, img.path.name)
            if img.path.exists():
                try:
                    shutil.move(str(img.path), str(back))
                    img.path = back
                except OSError as e:
                    print(f"Undo move error: {e}")
        
        img.status = prev_status
        img.moved_to = None
        if self.history:
            self.history.pop()
        self.current_index = idx
        self.update_stats()
        return True
    
    def save_progress(self, filepath: Path):
        """Save current progress to a JSON file."""
        data = {
            "source_root": str(self.source_root),
            "output_root": str(self.output_root),
            "classes": self.classes,
            "current_index": self.current_index,
            "images": [
                {
                    "path": str(img.path),
                    "original_path": str(img.original_path),
                    "class_name": img.class_name,
                    "status": img.status,
                    "original_class": img.original_class,
                    "moved_to": img.moved_to
                }
                for img in self.images
            ],
            "history": [
                {
                    "timestamp": h.timestamp,
                    "image_name": h.image_name,
                    "action": h.action,
                    "from_class": h.from_class,
                    "to_class": h.to_class
                }
                for h in self.history
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_progress(self, filepath: Path) -> bool:
        """Load progress from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.source_root = Path(data["source_root"])
            self.output_root = Path(data["output_root"])
            self.classes = data["classes"]
            self.current_index = data["current_index"]
            
            self.images = []
            for img_data in data["images"]:
                img = ImageItem(Path(img_data["path"]), img_data["class_name"])
                img.original_path = Path(img_data.get("original_path", img_data["path"]))
                img.status = img_data["status"]
                img.original_class = img_data["original_class"]
                img.moved_to = img_data.get("moved_to")
                self.images.append(img)
            
            self.history = []
            for h_data in data["history"]:
                h = HistoryEntry(h_data["image_name"], h_data["action"], h_data["from_class"], h_data.get("to_class"))
                h.timestamp = h_data["timestamp"]
                self.history.append(h)
            
            self.update_stats()
            return True
        except Exception as e:
            print(f"Error loading progress: {e}")
            return False
    
    def apply_changes(self) -> Tuple[int, int, int]:
        """Verify final state and copy any leftovers. Moves happen immediately."""
        if not self.output_root:
            return 0, 0, 0
        
        kept_count = 0
        deleted_count = 0
        moved_count = 0
        
        for img in self.images:
            if img.status == "kept":
                # File should already be in output; copy if still in source (idempotent)
                if img.path.exists() and self.source_root and img.path.is_relative_to(self.source_root):
                    dest_dir = self.output_root / img.class_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest = self._safe_dest(dest_dir, img.path.name)
                    shutil.copy2(img.path, dest)
                kept_count += 1
                
            elif img.status == "deleted":
                # Should already be in _deleted
                if img.path.exists() and self.source_root and img.path.is_relative_to(self.source_root):
                    dest_dir = self.output_root / "_deleted"
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest = self._safe_dest(dest_dir, img.path.name)
                    shutil.copy2(img.path, dest)
                deleted_count += 1
                
            elif img.status == "moved" and img.moved_to:
                if img.path.exists() and self.source_root and img.path.is_relative_to(self.source_root):
                    dest_dir = self.output_root / img.moved_to
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest = self._safe_dest(dest_dir, img.path.name)
                    shutil.copy2(img.path, dest)
                moved_count += 1
        
        return kept_count, deleted_count, moved_count


# ============================================================================
# Custom Widgets
# ============================================================================

class ImageDisplay(QScrollArea):
    """Custom scrollable image display with MS Photos-like zoom behavior."""
    
    double_clicked = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.zoom_factor = 1.0
        self.min_zoom = 0.05
        self.max_zoom = 20.0
        self.current_pixmap: Optional[QPixmap] = None
        self.fit_mode = True  # Start in fit-to-window mode
        self.last_mouse_pos = None
        self.is_panning = False
        
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignCenter)
        self.setBackgroundRole(QPalette.Dark)
        self.setMouseTracking(True)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1a1a1a;")
        self.image_label.setScaledContents(False)
        self.setWidget(self.image_label)
        
        self.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #1a1a1a;
            }
        """)
    
    def mousePressEvent(self, event):
        """Start panning with left mouse button."""
        if event.button() == Qt.LeftButton and self.zoom_factor > self._get_fit_zoom():
            self.is_panning = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Pan the image when dragging."""
        if self.is_panning and self.last_mouse_pos:
            delta = event.pos() - self.last_mouse_pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.last_mouse_pos = event.pos()
        else:
            # Update cursor based on zoom level
            if self.zoom_factor > self._get_fit_zoom():
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Stop panning."""
        if event.button() == Qt.LeftButton:
            self.is_panning = False
            if self.zoom_factor > self._get_fit_zoom():
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)
        
    def load_image(self, path: Path) -> bool:
        """Load an image from file."""
        try:
            pil_image = Image.open(path)
            
            # Convert PIL image to QImage
            if pil_image.mode == 'RGB':
                data = pil_image.tobytes('raw', 'RGB')
                qimage = QImage(data, pil_image.width, pil_image.height, 
                               pil_image.width * 3, QImage.Format_RGB888)
            elif pil_image.mode == 'RGBA':
                data = pil_image.tobytes('raw', 'RGBA')
                qimage = QImage(data, pil_image.width, pil_image.height,
                               pil_image.width * 4, QImage.Format_RGBA8888)
            else:
                pil_image = pil_image.convert('RGB')
                data = pil_image.tobytes('raw', 'RGB')
                qimage = QImage(data, pil_image.width, pil_image.height,
                               pil_image.width * 3, QImage.Format_RGB888)
            
            self.current_pixmap = QPixmap.fromImage(qimage)
            self.fit_mode = True
            self.fit_to_view()
            return True
            
        except Exception as e:
            print(f"Error loading image: {e}")
            self.image_label.setText(f"Error loading image:\n{e}")
            self.current_pixmap = None
            return False
    
    def _get_fit_zoom(self) -> float:
        """Calculate the zoom factor to fit image in viewport."""
        if not self.current_pixmap:
            return 1.0
        viewport_size = self.viewport().size()
        img_size = self.current_pixmap.size()
        scale_x = viewport_size.width() / img_size.width()
        scale_y = viewport_size.height() / img_size.height()
        return min(scale_x, scale_y)
    
    def _update_display(self):
        """Update the displayed image with current zoom."""
        if self.current_pixmap:
            scaled_size = self.current_pixmap.size() * self.zoom_factor
            scaled = self.current_pixmap.scaled(
                scaled_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)
            self.image_label.resize(scaled_size)
    
    def fit_to_view(self):
        """Fit image to the view area (MS Photos behavior)."""
        if self.current_pixmap:
            self.zoom_factor = self._get_fit_zoom()
            self.fit_mode = True
            self._update_display()
    
    def zoom_in(self):
        """Zoom in by 10% (smoother than 25%)."""
        self.fit_mode = False
        self.zoom_factor = min(self.zoom_factor * 1.1, self.max_zoom)
        self._update_display()
    
    def zoom_out(self):
        """Zoom out by 10%."""
        self.fit_mode = False
        self.zoom_factor = max(self.zoom_factor / 1.1, self.min_zoom)
        self._update_display()
    
    def reset_zoom(self):
        """Reset to 100% zoom."""
        self.fit_mode = False
        self.zoom_factor = 1.0
        self._update_display()
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming (MS Photos-like)."""
        if not self.current_pixmap:
            return
        
        # Get mouse position relative to image
        mouse_pos = event.pos()
        old_zoom = self.zoom_factor
        
        # Zoom in/out
        if event.angleDelta().y() > 0:
            self.zoom_factor = min(self.zoom_factor * 1.1, self.max_zoom)
        else:
            self.zoom_factor = max(self.zoom_factor / 1.1, self.min_zoom)
        
        self.fit_mode = False
        
        # Calculate scroll adjustment to keep mouse point steady
        zoom_ratio = self.zoom_factor / old_zoom
        
        # Get scroll bar values before zoom
        h_val = self.horizontalScrollBar().value()
        v_val = self.verticalScrollBar().value()
        
        # Update display
        self._update_display()
        
        # Adjust scroll to keep point under mouse
        viewport_pos = self.viewport().mapFromGlobal(event.globalPos())
        new_h = int((h_val + viewport_pos.x()) * zoom_ratio - viewport_pos.x())
        new_v = int((v_val + viewport_pos.y()) * zoom_ratio - viewport_pos.y())
        
        self.horizontalScrollBar().setValue(new_h)
        self.verticalScrollBar().setValue(new_v)
        
        event.accept()
    
    def mouseDoubleClickEvent(self, event):
        """Toggle between fit and 100% zoom."""
        if not self.current_pixmap:
            return
        
        if self.fit_mode or self.zoom_factor < 0.99:
            self.reset_zoom()
        else:
            self.fit_to_view()
        
        self.double_clicked.emit()
    
    def clear(self):
        """Clear the displayed image."""
        self.image_label.clear()
        self.image_label.setText("No image loaded")
        self.current_pixmap = None
        self.fit_mode = True


class SegmentBar(QWidget):
    """Multi-segment colored progress bar (kept/deleted/moved/pending)."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(8)
        self.setMinimumWidth(50)
        self.segments: List[Tuple[float, QColor]] = []

    def set_data(self, kept: int, deleted: int, moved: int, pending: int):
        total = kept + deleted + moved + pending
        self.segments = []
        if total > 0:
            self.segments = [
                (kept / total, QColor("#22c55e")),
                (deleted / total, QColor("#ef4444")),
                (moved / total, QColor("#f59e0b")),
                (pending / total, QColor("#3f4146")),
            ]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#2a2b2e"))
        painter.drawRoundedRect(rect, 4, 4)

        x = 0
        for frac, color in self.segments:
            if frac <= 0:
                continue
            w = round(rect.width() * frac)
            painter.setBrush(color)
            painter.drawRect(rect.x() + x, rect.y(), w, rect.height())
            x += w


class ClassCard(QFrame):
    """Modern card widget for displaying a class with progress."""
    
    clicked = pyqtSignal(str)  # Emits class name
    rename_requested = pyqtSignal(str)  # Emits class name
    
    def __init__(self, class_name: str, color: str, show_rename: bool = True, parent=None):
        super().__init__(parent)
        self.class_name = class_name
        self.color = color
        self.is_selected = False
        self.show_rename = show_rename
        
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(90)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(6)
        
        # Header with class name and rename button
        header = QHBoxLayout()
        header.setSpacing(6)
        self.name_label = QLabel(class_name)
        self.name_label.setWordWrap(True)
        self.name_label.setStyleSheet("""
            font-size: 14px;
            font-weight: 600;
            color: #ececec;
            background: transparent;
        """)
        header.addWidget(self.name_label, 1)
        
        if show_rename:
            rename_btn = QPushButton("\u270f")
            rename_btn.setFixedSize(24, 24)
            rename_btn.setToolTip("Rename class")
            rename_btn.setCursor(Qt.PointingHandCursor)
            rename_btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    border: none;
                    font-size: 12px;
                    color: #8a8a8a;
                }
                QPushButton:hover {
                    color: #ffffff;
                }
            """)
            rename_btn.clicked.connect(lambda: self.rename_requested.emit(self.class_name))
            header.addWidget(rename_btn)
        
        layout.addLayout(header)
        
        # Stats label
        self.stats_label = QLabel("0 images")
        self.stats_label.setStyleSheet("color: #9a9a9a; font-size: 11px; background: transparent;")
        layout.addWidget(self.stats_label)
        
        # Multi-segment progress bar
        self.progress_bar = SegmentBar()
        layout.addWidget(self.progress_bar)
        
        self._update_style()
    
    def mousePressEvent(self, event):
        """Handle click."""
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.class_name)
    
    def set_selected(self, selected: bool):
        """Set selection state."""
        self.is_selected = selected
        self._update_style()
    
    def update_stats(self, kept: int, deleted: int, moved: int, pending: int):
        """Update statistics and progress bar."""
        total = kept + deleted + moved + pending
        self.stats_label.setText(f"{total} images")
        self.progress_bar.set_data(kept, deleted, moved, pending)
    
    def _update_style(self):
        """Update card styling based on state."""
        if self.is_selected:
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: #24262a;
                    border: 2px solid {self.color};
                    border-radius: 8px;
                }}
            """)
        else:
            self.setStyleSheet("""
                QFrame {
                    background-color: #1e1f22;
                    border: 2px solid transparent;
                    border-radius: 8px;
                }
                QFrame:hover {
                    background-color: #26282c;
                }
            """)


class ClassListWidget(QWidget):
    """Modern scrollable list of class cards."""
    
    class_selected = pyqtSignal(object)  # Emits class name or None for "All"
    class_renamed = pyqtSignal(str, str)  # Emits (old_name, new_name)
    
    def __init__(self):
        super().__init__()
        self.cards = {}
        self.colors = ["#0078d4", "#2d8a3e", "#c4314b", "#986f0b", "#7a3b9e", "#008272"]
        self.selected_class = None
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = QLabel("Classes")
        header.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: 600;
                color: #fff;
                padding: 16px;
                background-color: #1a1a1a;
            }
        """)
        layout.addWidget(header)
        
        # Scroll area for cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #1a1a1a;
            }
        """)
        
        self.cards_container = QWidget()
        self.cards_container.setStyleSheet("background-color: #1a1a1a;")
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(8, 8, 8, 8)
        self.cards_layout.setSpacing(8)
        
        # "All Classes" card (no rename button)
        self.all_card = ClassCard("All Classes", "#9ca3af", show_rename=False)
        self.all_card.clicked.connect(self._on_all_clicked)
        self.cards_layout.addWidget(self.all_card)
        
        # Stretch to keep cards at top
        self.cards_layout.addStretch()
        
        scroll.setWidget(self.cards_container)
        layout.addWidget(scroll, 1)
        
        # Add class button
        add_btn = QPushButton("+ New Class")
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px;
                font-size: 13px;
                font-weight: 600;
                margin: 8px;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
            QPushButton:pressed {
                background-color: #0b6cb8;
            }
        """)
        add_btn.clicked.connect(self._add_new_class)
        layout.addWidget(add_btn)
    
    def set_classes(self, classes: List[str]):
        """Set the list of classes."""
        # Remove old class cards (keep index 0 = All Classes card, and stretch)
        while self.cards_layout.count() > 2:
            item = self.cards_layout.takeAt(1)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.cards.clear()
        
        # Create cards for each class
        for idx, class_name in enumerate(classes):
            color = self.colors[idx % len(self.colors)]
            card = ClassCard(class_name, color)
            card.clicked.connect(self._on_card_clicked)
            card.rename_requested.connect(self._rename_class)
            self.cards[class_name] = card
            self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)
    
    def _on_card_clicked(self, class_name: str):
        """Handle class card selection."""
        # Deselect all
        self.all_card.set_selected(False)
        for card in self.cards.values():
            card.set_selected(False)
        
        # Select clicked card
        if class_name in self.cards:
            self.cards[class_name].set_selected(True)
            self.selected_class = class_name
            self.class_selected.emit(class_name)
    
    def _on_all_clicked(self, _name: str):
        """Handle 'All Classes' card selection."""
        self.all_card.set_selected(True)
        for card in self.cards.values():
            card.set_selected(False)
        self.selected_class = None
        self.class_selected.emit(None)
    
    def select_all(self):
        """Deselect all cards (show all classes)."""
        self.all_card.set_selected(True)
        for card in self.cards.values():
            card.set_selected(False)
        self.selected_class = None
        self.class_selected.emit(None)
    
    def update_class_stats(self, class_name: str, kept: int, deleted: int, moved: int, pending: int):
        """Update statistics for a class card."""
        if class_name in self.cards:
            self.cards[class_name].update_stats(kept, deleted, moved, pending)
    
    def update_all_stats(self, kept: int, deleted: int, moved: int, pending: int):
        """Update statistics for the 'All Classes' card."""
        self.all_card.update_stats(kept, deleted, moved, pending)
    
    def _rename_class(self, old_name: str):
        """Prompt to rename a class."""
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Class",
            f"Enter new name for '{old_name}':",
            text=old_name
        )
        if ok and new_name and new_name != old_name:
            self.class_renamed.emit(old_name, new_name)
    
    def _add_new_class(self):
        """Add a new class."""
        name, ok = QInputDialog.getText(
            self,
            "New Class",
            "Enter name for new class:"
        )
        if ok and name:
            # Emit rename with empty old name to signal "add"
            self.class_renamed.emit("", name)


class ClassSelectionDialog(QDialog):
    """Dialog for selecting a target class to move an image to."""
    
    def __init__(self, classes: List[str], current_class: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Target Class")
        self.setMinimumWidth(300)
        self.selected_class: Optional[str] = None
        
        layout = QVBoxLayout(self)
        
        label = QLabel(f"Move image from '{current_class}' to:")
        label.setStyleSheet("font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(label)
        
        # List of classes (excluding current)
        self.class_list = QListWidget()
        self.class_list.setStyleSheet("""
            QListWidget {
                font-size: 13px;
                border: 1px solid #555;
            }
            QListWidget::item {
                padding: 8px;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
                color: white;
            }
        """)
        
        for cls in classes:
            if cls != current_class:
                self.class_list.addItem(cls)
        
        self.class_list.itemDoubleClicked.connect(self.accept_selection)
        layout.addWidget(self.class_list)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept_selection)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Focus the list
        self.class_list.setFocus()
        if self.class_list.count() > 0:
            self.class_list.setCurrentRow(0)
    
    def accept_selection(self):
        """Accept with the selected class."""
        current = self.class_list.currentItem()
        if current:
            self.selected_class = current.text()
            self.accept()
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.accept_selection()
        elif event.key() == Qt.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(event)


class DropArea(QFrame):
    """Widget that accepts drag-and-drop for folder selection."""
    
    folder_dropped = pyqtSignal(Path)
    
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(150)
        
        self.setStyleSheet("""
            QFrame {
                border: 3px dashed #666;
                border-radius: 10px;
                background-color: #2d2d2d;
            }
            QFrame:hover {
                border-color: #0078d4;
                background-color: #333;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        self.label = QLabel("Drag & Drop Image Folder Here\n\nor click 'Browse' below")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #999; font-size: 16px;")
        layout.addWidget(self.label)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                QFrame {
                    border: 3px dashed #0078d4;
                    border-radius: 10px;
                    background-color: #1a3a5c;
                }
            """)
    
    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QFrame {
                border: 3px dashed #666;
                border-radius: 10px;
                background-color: #2d2d2d;
            }
            QFrame:hover {
                border-color: #0078d4;
                background-color: #333;
            }
        """)
    
    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = Path(urls[0].toLocalFile())
            if path.is_dir():
                self.folder_dropped.emit(path)
        
        self.setStyleSheet("""
            QFrame {
                border: 3px dashed #666;
                border-radius: 10px;
                background-color: #2d2d2d;
            }
            QFrame:hover {
                border-color: #0078d4;
                background-color: #333;
            }
        """)


class HistoryListWidget(QListWidget):
    """Custom list widget for displaying history with colors."""
    
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QListWidget {
                background-color: #252525;
                border: 1px solid #444;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 4px 8px;
                border-bottom: 1px solid #333;
            }
        """)
    
    def add_history_entry(self, entry: HistoryEntry):
        """Add a history entry with appropriate styling."""
        item = QListWidgetItem(str(entry))
        
        if entry.action == "delete":
            item.setForeground(QColor("#ff6b6b"))
        elif entry.action == "move":
            item.setForeground(QColor("#69db7c"))
        else:
            item.setForeground(QColor("#74c0fc"))
        
        self.insertItem(0, item)
        self.scrollToTop()


# ============================================================================
# Main Window
# ============================================================================

class ImageProcessorWindow(QMainWindow):
    """Main application window for image classification processing."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Classification Processor")
        self.setMinimumSize(1280, 820)
        self.resize(1500, 950)
        
        self.state = ProcessingState()
        self.session_file = Path("image_processor_session.json")
        self.current_filter_class = None  # None means "All Classes"
        self.setAcceptDrops(True)
        
        # --- Gamification state (session-scoped) ---
        self.game_log: List[str] = []          # parallel to history: keep/delete/move
        self.streak: int = 0                    # consecutive keeps
        self.best_streak: int = 0
        self.score: int = 0
        self.kept_total: int = 0
        self.deleted_total: int = 0
        self.moved_total: int = 0
        self.achievements: set = set()
        
        self._setup_ui()
        self._setup_shortcuts()
        self._check_resume()
    
    def _setup_ui(self):
        """Set up the main UI layout."""
        # Central widget
        central = QWidget()
        central.setStyleSheet("background: #161719;")
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # === Left Panel: Class cards ===
        self.class_list = ClassListWidget()
        self.class_list.setFixedWidth(300)
        self.class_list.class_selected.connect(self._on_class_selected)
        self.class_list.class_renamed.connect(self._on_class_renamed)
        root.addWidget(self.class_list)

        # === Center Panel: Image Display ===
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        # Top bar: open folder, class badge, info, export
        top_bar = QWidget()
        top_bar.setStyleSheet("background: #1e1f22;")
        top_lay = QHBoxLayout(top_bar)
        top_lay.setContentsMargins(14, 10, 14, 10)
        top_lay.setSpacing(10)

        open_btn = QPushButton("\U0001f4c2 Open Folder")
        open_btn.setCursor(Qt.PointingHandCursor)
        open_btn.setStyleSheet("""
            QPushButton {
                background: #2c2e32; color: #ececec; border: none; border-radius: 6px;
                padding: 9px 16px; font-size: 13px; font-weight: 600;
            }
            QPushButton:hover { background: #383b40; }
        """)
        open_btn.clicked.connect(self._browse_folder)
        top_lay.addWidget(open_btn)

        # Class badge (large, prominent display)
        self.class_badge = QLabel("No class")
        self.class_badge.setAlignment(Qt.AlignCenter)
        self.class_badge.setMinimumWidth(150)
        self.class_badge.setStyleSheet("""
            QLabel {
                background-color: #3b82f6; color: white; border-radius: 10px;
                padding: 8px 18px; font-size: 16px; font-weight: 700;
            }
        """)
        top_lay.addWidget(self.class_badge)

        # Image info label
        self.info_label = QLabel("Open a folder to begin")
        self.info_label.setStyleSheet("color: #a5a5a5; font-size: 12px;")
        top_lay.addWidget(self.info_label, 1)

        export_btn = QPushButton("\u2714 Export")
        export_btn.setCursor(Qt.PointingHandCursor)
        export_btn.setStyleSheet("""
            QPushButton {
                background: #22c55e; color: white; border: none; border-radius: 6px;
                padding: 9px 18px; font-size: 13px; font-weight: 700;
            }
            QPushButton:hover { background: #16a34a; }
        """)
        export_btn.clicked.connect(self._apply_changes)
        top_lay.addWidget(export_btn)

        center_layout.addWidget(top_bar)

        # Image display
        self.image_display = ImageDisplay()
        center_layout.addWidget(self.image_display, 1)

        # Bottom action bar
        bottom_bar = QWidget()
        bottom_bar.setStyleSheet("background: #1e1f22;")
        bottom_lay = QHBoxLayout(bottom_bar)
        bottom_lay.setContentsMargins(14, 10, 14, 10)
        bottom_lay.setSpacing(8)

        def style_btn(color: str, hover: str):
            return (f"""
                QPushButton {{
                    background: {color}; color: white; border: none; border-radius: 6px;
                    padding: 9px 18px; font-size: 13px; font-weight: 600;
                }}
                QPushButton:hover {{ background: {hover}; }}
            """)

        self.prev_btn = QPushButton("\u25c0 Previous (A)")
        self.prev_btn.setCursor(Qt.PointingHandCursor)
        self.prev_btn.setStyleSheet(style_btn("#3a3d42", "#4a4e55"))
        self.prev_btn.clicked.connect(self._previous_image)
        bottom_lay.addWidget(self.prev_btn)

        keep_btn = QPushButton("\u2714 Keep (Enter)")
        keep_btn.setCursor(Qt.PointingHandCursor)
        keep_btn.setStyleSheet(style_btn("#22c55e", "#16a34a"))
        keep_btn.clicked.connect(self._keep_image)
        bottom_lay.addWidget(keep_btn)

        delete_btn = QPushButton("\u2715 Delete (X)")
        delete_btn.setCursor(Qt.PointingHandCursor)
        delete_btn.setStyleSheet(style_btn("#ef4444", "#dc2626"))
        delete_btn.clicked.connect(self._delete_image)
        bottom_lay.addWidget(delete_btn)

        move_btn = QPushButton("\u2192 Move (C)")
        move_btn.setCursor(Qt.PointingHandCursor)
        move_btn.setStyleSheet(style_btn("#f59e0b", "#d97706"))
        move_btn.clicked.connect(self._change_class)
        bottom_lay.addWidget(move_btn)

        undo_btn = QPushButton("\u21b6 Undo (Ctrl+Z)")
        undo_btn.setCursor(Qt.PointingHandCursor)
        undo_btn.setStyleSheet(style_btn("#3a3d42", "#4a4e55"))
        undo_btn.clicked.connect(self._undo)
        bottom_lay.addWidget(undo_btn)

        self.next_btn = QPushButton("Next (D) \u25b6")
        self.next_btn.setCursor(Qt.PointingHandCursor)
        self.next_btn.setStyleSheet(style_btn("#3a3d42", "#4a4e55"))
        self.next_btn.clicked.connect(self._next_image)
        bottom_lay.addWidget(self.next_btn)

        center_layout.addWidget(bottom_bar)

        # --- Gamification strip (slim, below action bar) ---
        game_bar = QWidget()
        game_bar.setStyleSheet("background: #161719;")
        game_lay = QHBoxLayout(game_bar)
        game_lay.setContentsMargins(16, 4, 16, 6)
        game_lay.setSpacing(18)

        self.streak_label = QLabel("\U0001f525 Streak: 0")
        self.streak_label.setStyleSheet("color: #fb923c; font-size: 13px; font-weight: 700;")
        game_lay.addWidget(self.streak_label)

        self.best_label = QLabel("\U0001f3c6 Best: 0")
        self.best_label.setStyleSheet("color: #fbbf24; font-size: 12px; font-weight: 600;")
        game_lay.addWidget(self.best_label)

        self.acc_label = QLabel("\U0001f4ca Model ACC: --")
        self.acc_label.setStyleSheet("color: #4ade80; font-size: 12px; font-weight: 600;")
        game_lay.addWidget(self.acc_label)

        self.score_label = QLabel("\U0001f3af Score: 0")
        self.score_label.setStyleSheet("color: #38bdf8; font-size: 12px; font-weight: 600;")
        game_lay.addWidget(self.score_label)

        self.level_label = QLabel("\U0001f396\ufe0f Data Tagger")
        self.level_label.setStyleSheet("color: #c084fc; font-size: 12px; font-weight: 600;")
        game_lay.addWidget(self.level_label)

        game_lay.addStretch()
        center_layout.addWidget(game_bar)

        root.addWidget(center_panel, 1)

        # === Right Panel: History Log ===
        right_panel = QWidget()
        right_panel.setFixedWidth(260)
        right_panel.setStyleSheet("background: #1a1b1e;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        history_label = QLabel("Action History")
        history_label.setStyleSheet("""
            color: #ffffff; font-size: 15px; font-weight: 700;
            padding: 14px 16px; background: #1a1b1e;
        """)
        right_layout.addWidget(history_label)

        self.history_list = HistoryListWidget()
        right_layout.addWidget(self.history_list)

        root.addWidget(right_panel)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Drop a folder or click 'Open Folder'")
    
    def _setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        # Navigation
        QShortcut(QKeySequence("A"), self, self._previous_image)
        QShortcut(QKeySequence("D"), self, self._next_image)
        QShortcut(QKeySequence(Qt.Key_Left), self, self._previous_image)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._next_image)
        
        # Actions
        QShortcut(QKeySequence(Qt.Key_Return), self, self._keep_image)
        QShortcut(QKeySequence(Qt.Key_Enter), self, self._keep_image)
        QShortcut(QKeySequence("X"), self, self._delete_image)
        QShortcut(QKeySequence("C"), self, self._change_class)
        
        # Undo
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo)
        
        # Zoom
        QShortcut(QKeySequence("+"), self, self.image_display.zoom_in)
        QShortcut(QKeySequence("="), self, self.image_display.zoom_in)
        QShortcut(QKeySequence("-"), self, self.image_display.zoom_out)
        QShortcut(QKeySequence("0"), self, self.image_display.reset_zoom)
        QShortcut(QKeySequence("F"), self, self.image_display.fit_to_view)
        
        # Save
        QShortcut(QKeySequence("Ctrl+S"), self, self._save_session)
    
    def _check_resume(self):
        """Check if there's a saved session to resume."""
        if self.session_file.exists():
            reply = QMessageBox.question(
                self,
                "Resume Session",
                "A previous session was found. Would you like to resume?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self._load_session()
    
    def _browse_folder(self):
        """Open folder browser dialog."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder",
            "",
            QFileDialog.ShowDirsOnly
        )
        if folder:
            self._load_folder(Path(folder))
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Accept folder drag & drop."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """Load folder dropped on the window."""
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_dir():
                self._load_folder(path)
                return
    
    def _load_folder(self, folder_path: Path):
        """Load images from the selected folder."""
        self.status_bar.showMessage(f"Loading images from {folder_path.name}...")
        QApplication.processEvents()
        
        if not self.state.load_from_directory(folder_path):
            QMessageBox.warning(
                self,
                "Error",
                f"No class folders with .bmp images found in:\n{folder_path}"
            )
            self.status_bar.showMessage("No images found")
            return
        
        self.current_filter_class = None
        self.class_list.set_classes(self.state.classes)
        self.class_list.select_all()
        
        self._update_ui()
        self._display_current_image()
        
        self.status_bar.showMessage(
            f"Loaded {len(self.state.images)} images from {len(self.state.classes)} classes"
        )
    
    def _on_class_selected(self, class_name: str):
        """Handle class card selection. class_name is a class or None for 'All'."""
        self.current_filter_class = class_name
        self._go_to_first_pending()
        self._update_stats()
        self._display_current_image()
    
    def _on_class_renamed(self, old_name: str, new_name: str):
        """Handle rename ('') or creation (old_name empty) of a class."""
        if not new_name:
            return
        
        if old_name == "":
            # New class creation
            if self.state.add_class(new_name):
                self.class_list.set_classes(self.state.classes)
                self._refresh_card_stats()
                self.status_bar.showMessage(f"Created class '{new_name}'", 3000)
            else:
                QMessageBox.warning(self, "Error", f"Could not create class '{new_name}'.")
            return
        
        # Rename existing class
        if self.state.rename_class(old_name, new_name):
            self.class_list.set_classes(self.state.classes)
            if self.current_filter_class == old_name:
                self.current_filter_class = new_name
            self._refresh_card_stats()
            self._display_current_image()
            self.status_bar.showMessage(f"Renamed '{old_name}' to '{new_name}'", 3000)
        else:
            QMessageBox.warning(self, "Error",
                f"Could not rename '{old_name}' to '{new_name}'.\n"
                "A class with that name may already exist.")
    
    def _refresh_card_stats(self):
        """Update stats on every class card."""
        for cls in self.state.classes:
            s = self.state.get_class_stats(cls)
            self.class_list.update_class_stats(cls, s["kept"], s["deleted"], s["moved"], s["pending"])
        total = self.state.get_class_stats(None)
        self.class_list.update_all_stats(total["kept"], total["deleted"], total["moved"], total["pending"])
    
    def _go_to_first_pending(self):
        """Navigate to first pending image in the current filter."""
        if not self.state.images:
            return
        
        if not self.current_filter_class:
            # Show all classes - find first pending from current position
            next_idx = self.state.get_next_pending_index(self.state.current_index + 1)
            if next_idx >= 0:
                self.state.current_index = next_idx
        else:
            # Filtered by class - find first pending of this class
            for i in range(len(self.state.images)):
                img = self.state.images[i]
                if img.class_name == self.current_filter_class and img.status == "pending":
                    self.state.current_index = i
                    break
            else:
                # No pending images in this class - stay at current or go to first
                if self.state.current_index >= len(self.state.images):
                    self.state.current_index = 0
    
    def _update_class_badge(self):
        """Update the prominent class badge display."""
        img = self.state.get_current_image()
        if img:
            self.class_badge.setText(f"📁 {img.class_name}")
            # Color code based on class index for visual distinction
            class_idx = self.state.classes.index(img.class_name) if img.class_name in self.state.classes else 0
            colors = ["#0078d4", "#2d8a3e", "#c4314b", "#986f0b", "#7a3b9e", "#008272"]
            color = colors[class_idx % len(colors)]
            self.class_badge.setStyleSheet(f"""
                QLabel {{
                    background-color: {color};
                    color: white;
                    font-size: 18px;
                    font-weight: bold;
                    padding: 10px 20px;
                    border-radius: 8px;
                    min-width: 120px;
                }}
            """)
        else:
            self.class_badge.setText("NO CLASS")
            self.class_badge.setStyleSheet("""
                QLabel {
                    background-color: #666;
                    color: white;
                    font-size: 18px;
                    font-weight: bold;
                    padding: 10px 20px;
                    border-radius: 8px;
                    min-width: 120px;
                }
            """)
    
    def _update_ui(self):
        """Update all UI elements based on current state."""
        self._update_stats()
        self._update_nav_buttons()
    
    def _update_stats(self):
        """Update statistics display (respecting class filter)."""
        self.state.update_stats()
        self._refresh_card_stats()
        
        if self.current_filter_class:
            filtered_imgs = [img for img in self.state.images if img.class_name == self.current_filter_class]
            total = len(filtered_imgs)
            pending = sum(1 for img in filtered_imgs if img.status == "pending")
            kept = sum(1 for img in filtered_imgs if img.status == "kept")
            deleted = sum(1 for img in filtered_imgs if img.status == "deleted")
            moved = sum(1 for img in filtered_imgs if img.status == "moved")
            filter_suffix = f" ({self.current_filter_class})"
        else:
            total = len(self.state.images)
            pending = self.state.stats.get("pending", 0)
            kept = self.state.stats.get("kept", 0)
            deleted = self.state.stats.get("deleted", 0)
            moved = self.state.stats.get("moved", 0)
            filter_suffix = " (All)"
        
        self.info_label.setText(self._get_info_text())
    
    def _get_info_text(self) -> str:
        """Get info text for current image."""
        img = self.state.get_current_image()
        if img:
            return (
                f"Image {self.state.current_index + 1} of {len(self.state.images)} | "
                f"Class: {img.class_name} | File: {img.path.name} | "
                f"Status: {img.status.upper()}"
            )
        return "No image loaded"
    
    def _update_nav_buttons(self):
        """Update navigation button states."""
        has_images = len(self.state.images) > 0
        self.prev_btn.setEnabled(has_images)
        self.next_btn.setEnabled(has_images)
    
    def _display_current_image(self):
        """Display the current image."""
        img = self.state.get_current_image()
        if img:
            if self.image_display.load_image(img.path):
                # Auto-fit on load
                QTimer.singleShot(100, self.image_display.fit_to_view)
            self.info_label.setText(self._get_info_text())
            self._update_class_badge()
        else:
            self.image_display.clear()
            self.info_label.setText("No image loaded")
            self._update_class_badge()
    
    def _previous_image(self):
        """Go to previous image (respecting class filter)."""
        if len(self.state.images) == 0:
            return
        
        if not self.current_filter_class:
            # All classes - cycle through all
            self.state.current_index = (self.state.current_index - 1) % len(self.state.images)
        else:
            # Filtered - cycle through only this class's images
            class_images = [i for i, img in enumerate(self.state.images) if img.class_name == self.current_filter_class]
            if class_images:
                current_pos = class_images.index(self.state.current_index) if self.state.current_index in class_images else 0
                self.state.current_index = class_images[(current_pos - 1) % len(class_images)]
        
        self._display_current_image()
        self._update_stats()
    
    def _next_image(self):
        """Go to next image (respecting class filter)."""
        if len(self.state.images) == 0:
            return
        
        if not self.current_filter_class:
            # All classes - cycle through all
            self.state.current_index = (self.state.current_index + 1) % len(self.state.images)
        else:
            # Filtered - cycle through only this class's images
            class_images = [i for i, img in enumerate(self.state.images) if img.class_name == self.current_filter_class]
            if class_images:
                current_pos = class_images.index(self.state.current_index) if self.state.current_index in class_images else 0
                self.state.current_index = class_images[(current_pos + 1) % len(class_images)]
        
        self._display_current_image()
        self._update_stats()
    
    def _keep_image(self):
        """Mark current image as kept."""
        if self.state.keep_image():
            self.history_list.add_history_entry(self.state.history[-1])
            self.game_log.append("keep")
            self._record_action("keep")
            self._update_stats()
            self._auto_advance()
            self._save_session()
    
    def _delete_image(self):
        """Mark current image for deletion."""
        if self.state.delete_image():
            self.history_list.add_history_entry(self.state.history[-1])
            self.game_log.append("delete")
            self._record_action("delete")
            self._update_stats()
            self._auto_advance()
            self._save_session()
    
    def _change_class(self):
        """Open dialog to change image class."""
        img = self.state.get_current_image()
        if not img or img.status != "pending":
            return
        
        dialog = ClassSelectionDialog(self.state.classes, img.class_name, self)
        if dialog.exec_() == QDialog.Accepted and dialog.selected_class:
            if self.state.move_image(dialog.selected_class):
                self.history_list.add_history_entry(self.state.history[-1])
                self.game_log.append("move")
                self._record_action("move")
                self._update_stats()
                self._auto_advance()
                self._save_session()
    
    def _auto_advance(self):
        """Automatically advance to next pending image (respecting class filter)."""
        cls = self.current_filter_class
        start = self.state.current_index + 1
        # Look forward
        for i in range(start, len(self.state.images)):
            img = self.state.images[i]
            if img.status == "pending" and (cls is None or img.class_name == cls):
                self.state.current_index = i
                self._display_current_image()
                self._update_stats()
                return
        # Look backward
        for i in range(0, start):
            img = self.state.images[i]
            if img.status == "pending" and (cls is None or img.class_name == cls):
                self.state.current_index = i
                self._display_current_image()
                self._update_stats()
                return
        msg = "All images processed!"
        if cls:
            msg = f"All images in '{cls}' processed!"
        self.status_bar.showMessage(msg, 4000)
    
    # ===================================================== GAMIFICATION
    _LEVELS = [
        (0,   "Data Tagger"),
        (10,  "Label Lord"),
        (25,  "Annotation Ace"),
        (50,  "Curator"),
        (100, "Dataset Sage"),
        (200, "Label Legend"),
        (400, "ML Zen Master"),
    ]

    _STREAK_MILESTONES = [5, 10, 25, 50, 100]
    _ACC_BADGES = [60, 75, 85, 90, 95, 99]

    def _level_for(self, total: int) -> Tuple[int, str]:
        """Return (level_index, title) for a given number of decisions."""
        for i in range(len(self._LEVELS) - 1, -1, -1):
            if total >= self._LEVELS[i][0]:
                return i, self._LEVELS[i][1]
        return 0, self._LEVELS[0][1]

    def _record_action(self, action: str):
        """Update streak/score/totals for an action, with milestone toasts."""
        changed = False
        if action == "keep":
            self.kept_total += 1
            self.streak += 1
            if self.streak > self.best_streak:
                self.best_streak = self.streak
                changed = True
            self.score += 10
            # streak moment
            if self.streak in self._STREAK_MILESTONES:
                self.status_bar.showMessage(
                    f"\U0001f525 {self.streak} keeps in a row! Model's on fire!", 3500)
            if self.best_streak in self._STREAK_MILESTONES and "best_streak" not in self.achievements:
                self.achievements.add("best_streak")
                self.status_bar.showMessage(f"\U0001f3c6 New best streak: {self.best_streak}!", 3500)
        elif action == "delete":
            self.deleted_total += 1
            self.streak = 0
            self.score += 2
        elif action == "move":
            self.moved_total += 1
            self.streak = 0
            self.score += 5

        # model accuracy badge (kept / processed)
        processed = self.kept_total + self.deleted_total + self.moved_total
        if processed >= 10:
            acc = self.kept_total / processed * 100
            for thresh in self._ACC_BADGES:
                if acc >= thresh and f"acc{thresh}" not in self.achievements:
                    self.achievements.add(f"acc{thresh}")
                    self.status_bar.showMessage(
                        f"\U0001f4ca Model accuracy reached {thresh:g}%! "
                        f"({acc:.1f}% current)", 3500)
                    changed = True

        # level-ups
        total = self.kept_total + self.deleted_total + self.moved_total
        new_idx, new_title = self._level_for(total)
        old_idx = self._game_level_idx if hasattr(self, "_game_level_idx") else 0
        if new_idx > old_idx:
            self.status_bar.showMessage(
                f"\U0001f396\ufe0f Level up! You are now a {new_title}!", 3500)
            changed = True
        self._game_level_idx = new_idx
        self._game_prev_total = total

        self._update_gamification()
        return changed

    def _unrecord_action(self, action: str):
        """Reverse an action's gamification effects (used by Undo)."""
        if action == "keep":
            self.kept_total = max(0, self.kept_total - 1)
            self.streak = max(0, self.streak - 1)
            self.score = max(0, self.score - 10)
        elif action == "delete":
            self.deleted_total = max(0, self.deleted_total - 1)
            self.score = max(0, self.score - 2)
        elif action == "move":
            self.moved_total = max(0, self.moved_total - 1)
            self.score = max(0, self.score - 5)

        total = self.kept_total + self.deleted_total + self.moved_total
        new_idx, new_title = self._level_for(total)
        if new_idx < self._game_level_idx:
            self.status_bar.showMessage(f"\U0001f396\ufe0f Down a level... no shame in it.", 2500)
        self._game_level_idx = new_idx
        self._update_gamification()

    def _update_gamification(self):
        """Refresh the gamification strip."""
        processed = self.kept_total + self.deleted_total + self.moved_total
        acc = (self.kept_total / processed * 100) if processed else None

        self.streak_label.setText(f"\U0001f525 Streak: {self.streak}")
        self.best_label.setText(f"\U0001f3c6 Best: {self.best_streak}")
        self.score_label.setText(f"\U0001f3af Score: {self.score}")

        level_idx, level_title = self._level_for(processed)
        self.level_label.setText(f"\U0001f396\ufe0f {level_title}")

        if acc is None:
            self.acc_label.setText("\U0001f4ca Model ACC: --")
        else:
            color = "#22c55e" if acc >= 85 else ("#eab308" if acc >= 60 else "#ef4444")
            self.acc_label.setText(f"\U0001f4ca Model ACC: {acc:.1f}%")
            self.acc_label.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: 600;")
    
    def _undo(self):
        """Undo the last action."""
        if self.state.undo_last():
            # Reverse gamification for the undone action
            if self.game_log:
                act = self.game_log.pop()
                self._unrecord_action(act)
            self._update_stats()
            self._display_current_image()
            # Update history list
            self.history_list.takeItem(0)
            self.status_bar.showMessage("Undo successful")
        else:
            self.status_bar.showMessage("Nothing to undo")
    
    def _save_session(self):
        """Save current session to file."""
        if self.state.source_root:
            self.state.save_progress(self.session_file)
            self.status_bar.showMessage("Session saved")
    
    def _load_session(self):
        """Load session from file."""
        if self.state.load_progress(self.session_file):
            self.current_filter_class = None
            self.class_list.set_classes(self.state.classes)
            self.class_list.select_all()
            
            # Rebuild gamification state from restored history
            self.streak = 0
            self.best_streak = 0
            self.score = 0
            self.kept_total = 0
            self.deleted_total = 0
            self.moved_total = 0
            self.game_log = []
            self.achievements = set()
            self._game_level_idx = 0
            for entry in self.state.history:
                act = {"keep": "keep", "delete": "delete", "move": "move"}.get(entry.action, "keep")
                self.game_log.append(act)
                self._record_action(act)
            
            self._update_ui()
            self._display_current_image()
            
            # Populate history list
            self.history_list.clear()
            for entry in reversed(self.state.history):
                self.history_list.add_history_entry(entry)
            
            self.status_bar.showMessage("Session restored")
        else:
            QMessageBox.warning(
                self,
                "Error",
                "Failed to load saved session"
            )
    
    def _apply_changes(self):
        """Apply all changes and export to output folder."""
        if not self.state.source_root:
            return
        
        pending = self.state.stats.get("pending", 0)
        if pending > 0:
            reply = QMessageBox.question(
                self,
                "Pending Images",
                f"There are still {pending} pending images. Continue anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        # Confirm action
        kept = self.state.stats.get("kept", 0)
        deleted = self.state.stats.get("deleted", 0)
        moved = self.state.stats.get("moved", 0)
        
        reply = QMessageBox.question(
            self,
            "Apply Changes",
            f"This will create the output folder with processed images:\n\n"
            f"  Kept: {kept} images (copied to same class)\n"
            f"  Deleted: {deleted} images (not copied)\n"
            f"  Moved: {moved} images (copied to new class)\n\n"
            f"Output location:\n{self.state.output_root}\n\n"
            f"Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            self.status_bar.showMessage("Applying changes...")
            QApplication.processEvents()
            
            k, d, m = self.state.apply_changes()
            
            QMessageBox.information(
                self,
                "Export Complete",
                f"Successfully exported:\n"
                f"  {k} kept images\n"
                f"  {d} deleted (skipped) images\n"
                f"  {m} moved images\n\n"
                f"Output saved to:\n{self.state.output_root}"
            )
            
            self.status_bar.showMessage("Export complete")
            
            # Clean up session file
            if self.session_file.exists():
                self.session_file.unlink()
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.state.history:
            reply = QMessageBox.question(
                self,
                "Save Session?",
                "Would you like to save your progress before closing?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self._save_session()
                event.accept()
            elif reply == QMessageBox.No:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(45, 45, 45))
    palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
    palette.setColor(QPalette.ToolTipBase, QColor(220, 220, 220))
    palette.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    palette.setColor(QPalette.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.Highlight, QColor(0, 120, 212))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = ImageProcessorWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
