import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QFileDialog, QSlider, QListWidget, QListWidgetItem
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QIcon
from PyQt6.QtCore import Qt, QPoint, QSize

class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.image = None
        self.points = []
        self.dragging_idx = None
        self.radius = 7
        self.sapphire = QColor(255, 56, 0)
        self.orange = QColor(0, 165, 255)
        self.setMinimumSize(400, 400)

    def set_image(self, image, points=None):
        self.image = image.copy()
        h, w = self.image.shape[:2]
        if points is None or len(points) != 4:
            self.points = [[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]
        else:
            self.points = [
                [max(0, min(w-1, int(p[0]))), max(0, min(h-1, int(p[1])))]
                for p in points
            ]
        self.update()

    def get_points(self):
        return np.array(self.points, dtype=np.float32)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.image is not None:
            h, w = self.image.shape[:2]
            # Calcola il rettangolo di destinazione per mantenere l'aspect ratio
            widget_w, widget_h = self.width(), self.height()
            scale = min(widget_w / w, widget_h / h)
            disp_w, disp_h = int(w * scale), int(h * scale)
            offset_x = (widget_w - disp_w) // 2
            offset_y = (widget_h - disp_h) // 2
            qimg = QImage(self.image.data, w, h, self.image.strides[0], QImage.Format.Format_BGR888)
            pix = QPixmap.fromImage(qimg).scaled(disp_w, disp_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            painter = QPainter(self)
            painter.drawPixmap(offset_x, offset_y, pix)
            # Disegna solo il rettangolo
            pen = QPen(self.sapphire, 2)
            painter.setPen(pen)
            scaled_points = [
                QPoint(int(p[0] * scale) + offset_x, int(p[1] * scale) + offset_y)
                for p in self.points
            ]
            for i in range(4):
                painter.drawLine(scaled_points[i], scaled_points[(i+1)%4])
            # Punti
            for i, p in enumerate(scaled_points):
                color = self.orange if i == 0 else self.sapphire
                painter.setPen(QPen(color, 2))
                painter.setBrush(color)
                painter.drawEllipse(p, self.radius, self.radius)
            painter.end()

    def mousePressEvent(self, event):
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        widget_w, widget_h = self.width(), self.height()
        scale = min(widget_w / w, widget_h / h)
        disp_w, disp_h = int(w * scale), int(h * scale)
        offset_x = (widget_w - disp_w) // 2
        offset_y = (widget_h - disp_h) // 2
        x = (event.position().x() - offset_x) / scale
        y = (event.position().y() - offset_y) / scale
        if x < 0 or y < 0 or x > w-1 or y > h-1:
            return
        for i, p in enumerate(self.points):
            if np.linalg.norm([x - p[0], y - p[1]]) < self.radius*2:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.dragging_idx = i
                elif event.button() == Qt.MouseButton.RightButton:
                    # Reset al vertice originale
                    if i == 0:
                        self.points[i] = [0, 0]
                    elif i == 1:
                        self.points[i] = [w-1, 0]
                    elif i == 2:
                        self.points[i] = [w-1, h-1]
                    elif i == 3:
                        self.points[i] = [0, h-1]
                    self.update()
                return

    def mouseMoveEvent(self, event):
        if self.image is None or self.dragging_idx is None:
            return
        h, w = self.image.shape[:2]
        widget_w, widget_h = self.width(), self.height()
        scale = min(widget_w / w, widget_h / h)
        disp_w, disp_h = int(w * scale), int(h * scale)
        offset_x = (widget_w - disp_w) // 2
        offset_y = (widget_h - disp_h) // 2
        x = (event.position().x() - offset_x) / scale
        y = (event.position().y() - offset_y) / scale
        x = max(0, min(w-1, int(x)))
        y = max(0, min(h-1, int(y)))
        self.points[self.dragging_idx] = [x, y]
        self.update()

    def mouseReleaseEvent(self, event):
        self.dragging_idx = None

class MainWindow(QMainWindow):
    def __init__(self, image_dir):
        super().__init__()
        self.setWindowTitle("OCR Receipt Annotator")
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.image_files.sort()
        self.current_idx = 0
        self.perimeters = {}  # filename -> 4 points
        self.preview_size = 60  # dimensione preview regolabile
        self.processing = {}  # filename -> bool (in analisi)
        self.spinner_movie = None
        spinner_path = os.path.join(os.path.dirname(__file__), 'spinner.gif')
        if os.path.exists(spinner_path):
            from PyQt6.QtGui import QMovie
            self.spinner_movie = QMovie(spinner_path)
        self.init_ui()
        self.load_image()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        # Preview verticale immagini
        class PreviewWidget(QWidget):
            def __init__(self, parent):
                super().__init__(parent)
                self.parent = parent
                self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            def wheelEvent(self, event):
                modifiers = QApplication.keyboardModifiers()
                if (modifiers & Qt.KeyboardModifier.ControlModifier) or (modifiers & Qt.KeyboardModifier.MetaModifier):
                    delta = event.angleDelta().y()
                    if delta > 0:
                        new_size = min(self.parent.preview_size + 8, 120)
                    else:
                        new_size = max(self.parent.preview_size - 8, 32)
                    self.parent.size_slider.setValue(new_size)
                else:
                    super().wheelEvent(event)
        self.preview_list = QListWidget()
        self.preview_list.setMaximumWidth(180)
        self.preview_list.setIconSize(QSize(self.preview_size, self.preview_size))
        self.preview_list.setMinimumWidth(140)
        self.preview_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.preview_list.setSpacing(8)
        self.preview_list.setStyleSheet("QListWidget::item { margin-bottom: 8px; }")
        self.preview_items = []
        for fname in self.image_files:
            img_path = os.path.join(self.image_dir, fname)
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                qimg = QImage(img.data, w, h, img.strides[0], QImage.Format.Format_BGR888)
                pix = QPixmap.fromImage(qimg).scaled(self.preview_size, self.preview_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                item = QListWidgetItem(QIcon(pix), fname)
                self.preview_list.addItem(item)
                self.preview_items.append(item)
        self.preview_list.currentRowChanged.connect(self.on_preview_selected)
        # Slider per regolare la dimensione delle preview
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setMinimum(32)
        self.size_slider.setMaximum(120)
        self.size_slider.setValue(self.preview_size)
        self.size_slider.valueChanged.connect(self.update_preview_size)
        preview_layout = QVBoxLayout()
        preview_layout.addWidget(self.preview_list)
        preview_layout.addWidget(self.size_slider)
        self.preview_widget = PreviewWidget(self)
        self.preview_widget.setLayout(preview_layout)
        main_layout.addWidget(self.preview_widget, 0)
        # Sinistra: immagine
        self.img_label = ImageLabel()
        main_layout.addWidget(self.img_label, 2)
        # Destra: placeholder
        right_panel = QVBoxLayout()
        self.ocr_label = QLabel("<b>Analisi OCR/LLM</b>\n(qui verrà mostrato il risultato)")
        self.ocr_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        right_panel.addWidget(self.ocr_label)
        right_panel.addStretch(1)
        main_layout.addLayout(right_panel, 1)
        # Sotto: barra navigazione (Start Analyze, Start Analyze All)
        nav_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("Start Analyze")
        self.analyze_btn.clicked.connect(self.start_analyze)
        self.analyze_all_btn = QPushButton("Start Analyze All")
        self.analyze_all_btn.clicked.connect(self.start_analyze_all)
        nav_layout.addStretch(1)
        nav_layout.addWidget(self.analyze_btn)
        nav_layout.addWidget(self.analyze_all_btn)
        # Layout finale
        layout = QVBoxLayout()
        layout.addLayout(main_layout)
        layout.addLayout(nav_layout)
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
        self.resize(1000, 700)

    def update_preview_size(self, value):
        self.preview_size = value
        self.preview_list.setIconSize(QSize(self.preview_size, self.preview_size))
        # Aggiorna le icone
        for idx, fname in enumerate(self.image_files):
            img_path = os.path.join(self.image_dir, fname)
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                qimg = QImage(img.data, w, h, img.strides[0], QImage.Format.Format_BGR888)
                pix = QPixmap.fromImage(qimg).scaled(self.preview_size, self.preview_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.preview_items[idx].setIcon(QIcon(pix))
        self.preview_list.update()

    def set_processing(self, fname, processing=True):
        self.processing[fname] = processing
        idx = self.image_files.index(fname)
        if processing and self.spinner_movie:
            label = QLabel()
            label.setFixedSize(self.preview_size, self.preview_size)
            label.setStyleSheet("background: transparent;")
            label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            label.setMovie(self.spinner_movie)
            self.spinner_movie.start()
            self.preview_list.setItemWidget(self.preview_items[idx], label)
        else:
            img_path = os.path.join(self.image_dir, fname)
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                qimg = QImage(img.data, w, h, img.strides[0], QImage.Format.Format_BGR888)
                pix = QPixmap.fromImage(qimg).scaled(self.preview_size, self.preview_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.preview_items[idx].setIcon(QIcon(pix))
                self.preview_list.setItemWidget(self.preview_items[idx], None)
        self.preview_list.update()

    def load_image(self):
        if not self.image_files:
            self.img_label.setText("Nessuna immagine trovata")
            return
        fname = self.image_files[self.current_idx]
        img_path = os.path.join(self.image_dir, fname)
        img = cv2.imread(img_path)
        points = self.perimeters.get(fname)
        self.img_label.set_image(img, points)
        self.setWindowTitle(f"OCR Receipt Annotator - {fname}")
        self.preview_list.setCurrentRow(self.current_idx)

    def save_current_perimeter(self):
        fname = self.image_files[self.current_idx]
        self.perimeters[fname] = self.img_label.get_points().tolist()

    def on_preview_selected(self, row):
        if row < 0 or row >= len(self.image_files):
            return
        self.save_current_perimeter()
        self.current_idx = row
        self.load_image()
        # Mostra il risultato OCR/LLM relativo a questo file, se presente
        if hasattr(self, 'ocr_results'):
            fname = self.image_files[self.current_idx]
            if fname in self.ocr_results:
                self.ocr_label.setText(self.ocr_results[fname])
            else:
                self.ocr_label.setText("<b>Analisi OCR/LLM</b>\n(qui verrà mostrato il risultato)")
        else:
            self.ocr_label.setText("<b>Analisi OCR/LLM</b>\n(qui verrà mostrato il risultato)")

    def prev_image(self):
        pass  # Disabilitato

    def next_image(self):
        pass  # Disabilitato

    def start_analyze(self, idx=None):
        from PyQt6.QtCore import QTimer
        if idx is None:
            idx = self.current_idx
        self.save_current_perimeter()
        fname = self.image_files[idx]
        coords = self.perimeters.get(fname)
        # Salva info analisi per ogni file
        if not hasattr(self, 'ocr_results'):
            self.ocr_results = {}
        self.ocr_results[fname] = f"<b>Analisi OCR/LLM</b><br>Richiesta analisi per:<br>{fname}<br>Vertici:<br>{coords}"
        if idx == self.current_idx:
            self.ocr_label.setText(self.ocr_results[fname])
        print(f"Richiesta analisi per: {fname} - Vertici: {coords}")
        self.set_processing(fname, True)
        QTimer.singleShot(2000, lambda: self.set_processing(fname, False))

    def start_analyze_all(self):
        from PyQt6.QtCore import QTimer
        if not hasattr(self, 'ocr_results'):
            self.ocr_results = {}
        for idx, fname in enumerate(self.image_files):
            self.save_current_perimeter()
            self.set_processing(fname, True)
            coords = self.perimeters.get(fname)
            self.ocr_results[fname] = f"<b>Analisi OCR/LLM</b><br>Richiesta analisi per:<br>{fname}<br>Vertici:<br>{coords}"
            QTimer.singleShot(2000 + idx*500, lambda f=fname: self.set_processing(f, False))
        # Aggiorna la UI con il risultato del file corrente
        curr_fname = self.image_files[self.current_idx]
        self.ocr_label.setText(self.ocr_results[curr_fname])
        print("Richiesta analisi per tutti i file.")


if __name__ == "__main__":
    import signal
    from PyQt6.QtCore import QTimer
    import threading
    def handle_sigint(sig, frame):
        print("\nTerminazione richiesta dall'utente (Ctrl+C). Uscita...")
        QTimer.singleShot(0, QApplication.quit)
    signal.signal(signal.SIGINT, handle_sigint)
    app = QApplication(sys.argv)
    # Timer per forzare la gestione dei segnali anche se la finestra non è in focus
    def keep_alive():
        # Timer che ogni 200ms chiama processEvents per permettere la gestione di SIGINT
        timer = QTimer()
        timer.timeout.connect(lambda: None)
        timer.start(200)
        return timer
    _ka_timer = keep_alive()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="test_receipt", help="Directory immagini")
    args = parser.parse_args()
    window = MainWindow(args.dir)
    window.show()
    sys.exit(app.exec())

