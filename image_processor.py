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
    QDialog, QVBoxLayout, QDialogButtonBox, QFileDialog, QMessageBox,
    QScrollArea, QFrame, QStatusBar, QProgressBar, QShortcut, QComboBox
)
from PyQt5.QtCore import Qt, QTimer, QMimeData, pyqtSignal, QSize
from PyQt5.QtGui import (
    QPixmap, QImage, QKeySequence, QWheelEvent, QDragEnterEvent,
    QDropEvent, QFont, QColor, QPalette
)
from PIL import Image


# ============================================================================
# Data Classes and State Management
# ============================================================================

class ImageItem:
    """Represents a single image in the processing queue."""
    
    def __init__(self, path: Path, class_name: str):
        self.path = path
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
    
    def keep_image(self) -> bool:
        """Mark current image as kept."""
        img = self.get_current_image()
        if img and img.status == "pending":
            img.status = "kept"
            self.undo_stack.append((img, self.current_index, "pending"))
            self.history.append(HistoryEntry(img.path.name, "keep", img.class_name))
            self.update_stats()
            return True
        return False
    
    def delete_image(self) -> bool:
        """Mark current image for deletion."""
        img = self.get_current_image()
        if img and img.status == "pending":
            img.status = "deleted"
            self.undo_stack.append((img, self.current_index, "pending"))
            self.history.append(HistoryEntry(img.path.name, "delete", img.class_name))
            self.update_stats()
            return True
        return False
    
    def move_image(self, target_class: str) -> bool:
        """Mark current image to be moved to another class."""
        img = self.get_current_image()
        if img and img.status == "pending" and target_class != img.class_name:
            img.status = "moved"
            img.moved_to = target_class
            self.undo_stack.append((img, self.current_index, "pending"))
            self.history.append(HistoryEntry(img.path.name, "move", img.class_name, target_class))
            self.update_stats()
            return True
        return False
    
    def undo_last(self) -> bool:
        """Undo the last action."""
        if self.undo_stack:
            img, idx, prev_status = self.undo_stack.pop()
            img.status = prev_status
            img.moved_to = None
            if self.history:
                self.history.pop()
            self.current_index = idx
            self.update_stats()
            return True
        return False
    
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
        """Apply all changes and copy files to output directory. Returns (kept, deleted, moved)."""
        if not self.output_root:
            return 0, 0, 0
        
        kept_count = 0
        deleted_count = 0
        moved_count = 0
        
        for img in self.images:
            if img.status == "kept":
                # Copy to same class in output
                dest_dir = self.output_root / img.class_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img.path, dest_dir / img.path.name)
                kept_count += 1
                
            elif img.status == "deleted":
                # Don't copy - mark as deleted (or actually delete if you want)
                # For safety, we'll just not copy them to output
                deleted_count += 1
                
            elif img.status == "moved" and img.moved_to:
                # Copy to new class in output
                dest_dir = self.output_root / img.moved_to
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img.path, dest_dir / img.path.name)
                moved_count += 1
        
        return kept_count, deleted_count, moved_count


# ============================================================================
# Custom Widgets
# ============================================================================

class ImageDisplay(QScrollArea):
    """Custom scrollable image display with zoom capability and mouse-position zoom."""
    
    double_clicked = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.current_pixmap: Optional[QPixmap] = None
        self.mouse_pos = None  # Track mouse position for zooming
        
        self.setWidgetResizable(True)
        self.setAlignment(Qt.AlignCenter)
        self.setBackgroundRole(QPalette.Dark)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1e1e1e;")
        self.image_label.setMouseTracking(True)
        self.setWidget(self.image_label)
        
        self.setStyleSheet("""
            QScrollArea {
                border: 2px solid #444;
                background-color: #1e1e1e;
            }
        """)
    
    def _on_mouse_move(self, event):
        """Track mouse position for zooming."""
        self.mouse_pos = event.pos()
    
    def mouseMoveEvent(self, event):
        """Override to track mouse position."""
        self.mouse_pos = self.mapFromGlobal(event.globalPos())
        super().mouseMoveEvent(event)
        
    def load_image(self, path: Path) -> bool:
        """Load an image from file."""
        try:
            # Use PIL for better format support, then convert to QPixmap
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
            self.zoom_factor = 1.0
            self._update_display()
            return True
            
        except Exception as e:
            print(f"Error loading image: {e}")
            self.image_label.setText(f"Error loading image:\n{e}")
            self.current_pixmap = None
            return False
    
    def _update_display(self):
        """Update the displayed image with current zoom."""
        if self.current_pixmap:
            scaled = self.current_pixmap.scaled(
                self.current_pixmap.size() * self.zoom_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)
            self.image_label.resize(scaled.size())
    
    def fit_to_view(self):
        """Fit image to the view area."""
        if self.current_pixmap:
            viewport_size = self.viewport().size() - QSize(20, 20)
            img_size = self.current_pixmap.size()
            
            # Calculate scale to fit
            scale_x = viewport_size.width() / img_size.width()
            scale_y = viewport_size.height() / img_size.height()
            self.zoom_factor = min(scale_x, scale_y, 1.0)
            self._update_display()
    
    def zoom_in(self):
        """Zoom in by 25%."""
        self.zoom_factor = min(self.zoom_factor * 1.25, self.max_zoom)
        self._update_display()
    
    def zoom_out(self):
        """Zoom out by 25%."""
        self.zoom_factor = max(self.zoom_factor / 1.25, self.min_zoom)
        self._update_display()
    
    def reset_zoom(self):
        """Reset zoom to 100%."""
        self.zoom_factor = 1.0
        self._update_display()
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming."""
        if event.modifiers() == Qt.ControlModifier:
            # Zoom toward mouse position
            zoom_delta = 1.25 if event.angleDelta().y() > 0 else 0.8
            self._zoom_at_mouse(zoom_delta)
            event.accept()
        else:
            super().wheelEvent(event)
    
    def _zoom_at_mouse(self, zoom_factor: float):
        """Zoom toward current mouse position."""
        if not self.current_pixmap or not self.mouse_pos:
            return
        
        # Calculate the ratio of mouse position to image size
        current_scaled_size = self.current_pixmap.size() * self.zoom_factor
        if current_scaled_size.width() == 0 or current_scaled_size.height() == 0:
            return
        
        old_ratio_x = self.horizontalScrollBar().value() / max(1, current_scaled_size.width() - self.viewport().width())
        old_ratio_y = self.verticalScrollBar().value() / max(1, current_scaled_size.height() - self.viewport().height())
        
        # Apply zoom
        self.zoom_factor = min(max(self.zoom_factor * zoom_factor, self.min_zoom), self.max_zoom)
        
        # Update display
        self._update_display()
        
        # Adjust scrollbars to maintain zoom center
        new_scaled_size = self.current_pixmap.size() * self.zoom_factor
        if new_scaled_size.width() > self.viewport().width():
            new_x = int(old_ratio_x * (new_scaled_size.width() - self.viewport().width()))
            self.horizontalScrollBar().setValue(new_x)
        else:
            self.horizontalScrollBar().setValue(0)
            
        if new_scaled_size.height() > self.viewport().height():
            new_y = int(old_ratio_y * (new_scaled_size.height() - self.viewport().height()))
            self.verticalScrollBar().setValue(new_y)
        else:
            self.verticalScrollBar().setValue(0)
    
    def mouseDoubleClickEvent(self, event):
        """Handle double click to fit to view."""
        self.fit_to_view()
        self.double_clicked.emit()
    
    def clear(self):
        """Clear the displayed image."""
        self.image_label.clear()
        self.image_label.setText("No image loaded")
        self.current_pixmap = None


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
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        self.state = ProcessingState()
        self.session_file = Path("image_processor_session.json")
        self.current_filter_class = None  # None means "All Classes"
        
        self._setup_ui()
        self._setup_shortcuts()
        self._check_resume()
    
    def _setup_ui(self):
        """Set up the main UI layout."""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # === Left Panel: Controls & Info ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        
        # Drop area for folder selection
        self.drop_area = DropArea()
        left_layout.addWidget(self.drop_area)
        self.drop_area.folder_dropped.connect(self._load_folder)
        
        # Browse button
        browse_btn = QPushButton("📁 Browse for Folder")
        browse_btn.setStyleSheet("""
            QPushButton {
                padding: 10px;
                font-size: 14px;
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
            QPushButton:pressed {
                background-color: #0b6cb8;
            }
        """)
        browse_btn.clicked.connect(self._browse_folder)
        left_layout.addWidget(browse_btn)
        
        # Stats frame
        stats_frame = QFrame()
        stats_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        stats_layout = QVBoxLayout(stats_frame)
        
        self.stats_label = QLabel("Statistics:\nPending: 0\nKept: 0\nDeleted: 0\nMoved: 0")
        self.stats_label.setStyleSheet("color: #ccc; font-size: 12px;")
        stats_layout.addWidget(self.stats_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444;
                border-radius: 3px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
            }
        """)
        stats_layout.addWidget(self.progress_bar)
        
        left_layout.addWidget(stats_frame)
        
        # Keyboard shortcuts info
        shortcuts_label = QLabel(
            "Keyboard Shortcuts:\n"
            "A / D - Previous / Next image\n"
            "Enter - Keep image\n"
            "X - Delete image\n"
            "C - Change class\n"
            "Ctrl+Z - Undo\n"
            "F - Fit to view\n"
            "+ / - - Zoom in/out\n"
            "0 - Reset zoom"
        )
        shortcuts_label.setStyleSheet("""
            color: #888;
            font-size: 11px;
            background-color: #252525;
            padding: 10px;
            border-radius: 5px;
        """)
        left_layout.addWidget(shortcuts_label)
        
        # Apply changes button
        self.apply_btn = QPushButton("✅ Apply Changes & Export")
        self.apply_btn.setEnabled(False)
        self.apply_btn.setStyleSheet("""
            QPushButton {
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
                background-color: #2d8a3e;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #34a049;
            }
            QPushButton:disabled {
                background-color: #444;
                color: #666;
            }
        """)
        self.apply_btn.clicked.connect(self._apply_changes)
        left_layout.addWidget(self.apply_btn)
        
        left_layout.addStretch()
        splitter.addWidget(left_panel)
        
        # === Center Panel: Image Display ===
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(5)
        
        # Class filter and info bar (prominent class display)
        self.class_filter_combo = QComboBox()
        self.class_filter_combo.setStyleSheet("""
            QComboBox {
                padding: 8px 12px;
                font-size: 14px;
                background-color: #333;
                color: white;
                border: 1px solid #555;
                border-radius: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #888;
            }
        """)
        self.class_filter_combo.addItem("All Classes")
        self.class_filter_combo.currentIndexChanged.connect(self._on_class_filter_changed)
        
        # Class badge (large, prominent display)
        self.class_badge = QLabel("NO CLASS")
        self.class_badge.setStyleSheet("""
            QLabel {
                background-color: #0078d4;
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 8px;
                min-width: 120px;
            }
        """)
        self.class_badge.setAlignment(Qt.AlignCenter)
        
        # Image info label (smaller, below class)
        self.info_label = QLabel("No image loaded")
        self.info_label.setStyleSheet("""
            color: #aaa;
            font-size: 12px;
            padding: 6px;
            background-color: #252525;
            border-radius: 3px;
        """)
        
        # Top bar with filter and class display
        top_bar = QHBoxLayout()
        top_bar.addWidget(self.class_filter_combo)
        top_bar.addWidget(self.class_badge)
        top_bar.addWidget(self.info_label)
        top_bar.setSpacing(10)
        center_layout.addLayout(top_bar)
        
        # Image display
        self.image_display = ImageDisplay()
        center_layout.addWidget(self.image_display)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("◀ Previous (A)")
        self.prev_btn.setStyleSheet("""
            QPushButton {
                padding: 10px 20px;
                font-size: 13px;
                background-color: #444;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #555;
            }
        """)
        self.prev_btn.clicked.connect(self._previous_image)
        nav_layout.addWidget(self.prev_btn)
        
        nav_layout.addStretch()
        
        self.next_btn = QPushButton("Next (D) ▶")
        self.next_btn.setStyleSheet("""
            QPushButton {
                padding: 10px 20px;
                font-size: 13px;
                background-color: #444;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #555;
            }
        """)
        self.next_btn.clicked.connect(self._next_image)
        nav_layout.addWidget(self.next_btn)
        
        center_layout.addLayout(nav_layout)
        splitter.addWidget(center_panel)
        
        # === Right Panel: History Log ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        history_label = QLabel("Action History")
        history_label.setStyleSheet("color: #ccc; font-size: 14px; font-weight: bold;")
        right_layout.addWidget(history_label)
        
        self.history_list = HistoryListWidget()
        right_layout.addWidget(self.history_list)
        
        splitter.addWidget(right_panel)
        
        # Set splitter sizes (left: 200, center: 600, right: 200)
        splitter.setSizes([250, 700, 250])
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Drag & drop a folder to begin")
    
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
        
        # Populate class filter dropdown
        self.class_filter_combo.blockSignals(True)
        self.class_filter_combo.clear()
        self.class_filter_combo.addItem("All Classes")
        for cls in sorted(self.state.classes):
            self.class_filter_combo.addItem(cls)
        self.class_filter_combo.blockSignals(False)
        self.current_filter_class = None
        
        self._update_ui()
        self._display_current_image()
        self.apply_btn.setEnabled(True)
        
        self.status_bar.showMessage(
            f"Loaded {len(self.state.images)} images from {len(self.state.classes)} classes"
        )
    
    def _on_class_filter_changed(self, index: int):
        """Handle class filter change - update view to show only selected class images."""
        if index == 0:
            self.current_filter_class = None  # All classes
        else:
            self.current_filter_class = self.class_filter_combo.itemText(index)
        
        # Find first pending image in the filtered set
        self._go_to_first_pending()
        self._update_stats()
        self._display_current_image()
    
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
        
        # Get filtered images for stats display
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
        
        self.stats_label.setText(
            f"Statistics{filter_suffix}:\n"
            f"Pending: {pending}\n"
            f"Kept: {kept}\n"
            f"Deleted: {deleted}\n"
            f"Moved: {moved}"
        )
        
        if total > 0:
            processed = total - pending
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(processed)
            self.progress_bar.setFormat(f"{processed}/{total} ({processed*100//total}%)")
        
        self._update_class_badge()
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
        else:
            self.image_display.clear()
            self.info_label.setText("No image loaded")
    
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
            self._update_stats()
            self._auto_advance()
    
    def _delete_image(self):
        """Mark current image for deletion."""
        if self.state.delete_image():
            self.history_list.add_history_entry(self.state.history[-1])
            self._update_stats()
            self._auto_advance()
    
    def _change_class(self):
        """Open dialog to change image class."""
        img = self.state.get_current_image()
        if not img or img.status != "pending":
            return
        
        dialog = ClassSelectionDialog(self.state.classes, img.class_name, self)
        if dialog.exec_() == QDialog.Accepted and dialog.selected_class:
            if self.state.move_image(dialog.selected_class):
                self.history_list.add_history_entry(self.state.history[-1])
                self._update_stats()
                self._auto_advance()
    
    def _auto_advance(self):
        """Automatically advance to next pending image."""
        # Find next pending image
        next_idx = self.state.get_next_pending_index(self.state.current_index + 1)
        if next_idx >= 0:
            self.state.current_index = next_idx
            self._display_current_image()
            self._update_stats()
    
    def _undo(self):
        """Undo the last action."""
        if self.state.undo_last():
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
            self._update_ui()
            self._display_current_image()
            self.apply_btn.setEnabled(True)
            
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
