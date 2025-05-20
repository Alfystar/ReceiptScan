"""
Modulo che contiene la classe PreviewWidget per la visualizzazione delle anteprime dei file.
"""

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import QWidget, QApplication


class PreviewWidget(QWidget):
    """Widget per la visualizzazione dell'anteprima dei file."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        """Gestisce l'evento della rotellina del mouse per lo zoom delle anteprime."""
        modifiers = QApplication.keyboardModifiers()
        if (modifiers & Qt.KeyboardModifier.ControlModifier) or (modifiers & Qt.KeyboardModifier.MetaModifier):
            delta = event.angleDelta().y()
            if delta > 0:
                new_size = min(self.parent.preview_size + 8, 120)
            else:
                new_size = max(self.parent.preview_size - 8, 32)
            # Aggiorna direttamente la dimensione dell'anteprima senza usare lo slider
            self.parent.preview_size = new_size
            self.parent.preview_list.setIconSize(QSize(new_size, new_size))
            self.parent.preview_size_changed.emit(new_size)
        else:
            super().wheelEvent(event)
