from PyQt6.QtCore import QDate
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QDateEdit, QDoubleSpinBox, QTextEdit, QFrame


class TransactionDetailsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # self.setFixedHeight(230)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10) # Spazio tra gli elementi

        # Titolo
        title = QLabel("<b>Dettagli Transazione</b>")
        title.setStyleSheet("font-size: 13px; margin-bottom: 2px; margin-top: 12px;")
        layout.addWidget(title)

        # Riga: Data | divisore | Totale
        row1 = QHBoxLayout()
        row1.setSpacing(0)
        # Data
        date_col = QVBoxLayout()
        date_label = QLabel("Data")
        date_label.setStyleSheet("margin-bottom: 2px;")
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setDisplayFormat("dd/MM/yyyy")

        # Stile migliorato per il QDateEdit e il calendario popup
        self.date_edit.setStyleSheet("""
            QDateEdit { 
                background-color: white; 
                color: #212121; 
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                padding: 2px;
            }
            
            QCalendarWidget {
                min-width: 300px;
                min-height: 200px;
            }
            
            QCalendarWidget QWidget {
                background-color: white;
                color: #212121;
            }
            
            QCalendarWidget QToolButton {
                background-color: #FFFFFF;
                color: #212121;
                border: none;
                border-radius: 4px;
                padding: 2px;
                margin: 2px;
            }
            
            QCalendarWidget QToolButton:hover {
                background-color: #E0E0E0;
            }
            
            QCalendarWidget QMenu {
                background-color: white;
                color: #212121;
            }
            
            QCalendarWidget QSpinBox {
                background-color: white;
                color: #212121;
                border: 1px solid #BDBDBD;
                border-radius: 4px;
            }
            
            QCalendarWidget QAbstractItemView {
                background-color: white;
                color: #212121;
                selection-background-color: #2979FF;
                selection-color: white;
            }
            
            QCalendarWidget QAbstractItemView:enabled {
                color: #212121;
            }
            
            QCalendarWidget QAbstractItemView:disabled {
                color: #9E9E9E;
            }
        """)

        date_col.addWidget(date_label)
        date_col.addWidget(self.date_edit)
        row1.addLayout(date_col)

        # Spazio tra Data e Totale
        row1.addSpacing(24)

        # Divisore verticale
        divider1 = QFrame()
        divider1.setFrameShape(QFrame.Shape.VLine)
        divider1.setFrameShadow(QFrame.Shadow.Sunken)
        divider1.setStyleSheet("color: #bbb; margin: 0 12px;")
        row1.addWidget(divider1)

        # Spazio tra divisore e Totale
        row1.addSpacing(24)

        # Totale
        total_col = QVBoxLayout()
        total_label = QLabel("Totale")
        total_label.setStyleSheet("margin-bottom: 2px;")
        self.total_edit = QDoubleSpinBox()
        self.total_edit.setMaximum(1000000)
        self.total_edit.setPrefix("€ ")
        self.total_edit.setStyleSheet("""
            QDoubleSpinBox { 
                background-color: white; 
                color: #212121; 
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                padding: 2px;
            } 
            
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { 
                width: 18px; 
                height: 18px; 
                margin: 0; 
                padding: 0; 
                background-color: white;
            }
            
            QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow { 
                width: 14px; 
                height: 14px; 
            }
        """)
        total_col.addWidget(total_label)
        total_col.addWidget(self.total_edit)
        row1.addLayout(total_col)
        layout.addLayout(row1)

        # Riga: Tipo pagamento | divisore | Valuta
        row2 = QHBoxLayout()
        row2.setSpacing(0)
        # Tipo pagamento
        payment_col = QVBoxLayout()
        payment_label = QLabel("Tipo di pagamento")
        payment_label.setStyleSheet("margin-bottom: 2px;")
        self.payment_type = QComboBox()
        self.payment_type.addItems(["Contanti", "Carta", "Bancomat", "Altro"])
        self.payment_type.setStyleSheet("""
            QComboBox {
                background-color: white;
                color: #212121;
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                padding: 2px;
            }
            
            QComboBox QAbstractItemView {
                background-color: white;
                color: #212121;
                border: 1px solid #BDBDBD;
                selection-background-color: #2979FF;
                selection-color: white;
            }
        """)
        payment_col.addWidget(payment_label)
        payment_col.addWidget(self.payment_type)
        row2.addLayout(payment_col)

        # Spazio tra Tipo pagamento e Valuta
        row2.addSpacing(24)

        # Divisore verticale
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.VLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        divider.setStyleSheet("color: #bbb; margin: 0 12px;")
        row2.addWidget(divider)

        # Spazio tra divisore e Valuta
        row2.addSpacing(24)

        # Valuta
        currency_col = QVBoxLayout()
        currency_label = QLabel("Valuta")
        currency_label.setStyleSheet("margin-bottom: 2px;")
        self.currency_combo = QComboBox()
        self.currency_combo.addItems(["EUR", "USD", "GBP", "CHF", "JPY", "Altro"])
        self.currency_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                color: #212121;
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                padding: 2px;
            }
            
            QComboBox QAbstractItemView {
                background-color: white;
                color: #212121;
                border: 1px solid #BDBDBD;
                selection-background-color: #2979FF;
                selection-color: white;
            }
        """)
        currency_col.addWidget(currency_label)
        currency_col.addWidget(self.currency_combo)
        row2.addLayout(currency_col)
        layout.addLayout(row2)

        # Descrizione (2-3 righe)
        desc_label = QLabel("Descrizione")
        desc_label.setStyleSheet("margin-top: 6px; margin-bottom: 2px;")
        self.description_edit = QTextEdit()
        self.description_edit.setMinimumHeight(40)
        self.description_edit.setMaximumHeight(120)
        self.description_edit.setStyleSheet("""
            QTextEdit {
                background-color: white;
                color: #212121;
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                padding: 2px;
            }
        """)
        layout.addWidget(desc_label)
        layout.addWidget(self.description_edit)

        layout.addStretch(1)
        self.setStyleSheet("""
            QWidget {
                background-color: transparent;
                border: none;
            }
            QLabel {
                font-size: 12px;
            }
        """)

    def get_details(self):
        return {
            "date": self.date_edit.date().toString("yyyy-MM-dd"),
            "total": self.total_edit.value(),
            "payment_type": self.payment_type.currentText(),
            "description": self.description_edit.toPlainText(),
            "currency": self.currency_combo.currentText(),
        }

    def set_details(self, details):
        if "date" in details:
            self.date_edit.setDate(QDate.fromString(details["date"], "yyyy-MM-dd"))
        if "total" in details:
            self.total_edit.setValue(float(details["total"]))
        if "payment_type" in details:
            idx = self.payment_type.findText(details["payment_type"])
            if idx >= 0:
                self.payment_type.setCurrentIndex(idx)
        if "description" in details:
            self.description_edit.setPlainText(details["description"])
        if "currency" in details:
            idx = self.currency_combo.findText(details["currency"])
            if idx >= 0:
                self.currency_combo.setCurrentIndex(idx)
            else:
                self.currency_combo.setCurrentIndex(0)
