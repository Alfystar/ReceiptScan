from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QDateEdit, QDoubleSpinBox, QTextEdit
from PyQt6.QtCore import QDate

class TransactionDetailsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Titolo
        title = QLabel("<b>Dettagli Transazione</b>")
        title.setStyleSheet("font-size: 13px;")
        layout.addWidget(title)

        # Riga: Data e Totale
        row = QHBoxLayout()
        row.setSpacing(8)
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setDisplayFormat("dd/MM/yyyy")
        row.addWidget(QLabel("Data"))
        row.addWidget(self.date_edit)
        self.total_edit = QDoubleSpinBox()
        self.total_edit.setMaximum(1000000)
        self.total_edit.setPrefix("€ ")
        row.addWidget(QLabel("Totale"))
        row.addWidget(self.total_edit)
        layout.addLayout(row)

        # Tipo di pagamento
        self.payment_type = QComboBox()
        self.payment_type.addItems(["Contanti", "Carta", "Bancomat", "Altro"])
        layout.addWidget(QLabel("Tipo di pagamento"))
        layout.addWidget(self.payment_type)

        # Valuta (menu a tendina)
        self.currency_combo = QComboBox()
        self.currency_combo.addItems(["EUR", "USD", "GBP", "CHF", "JPY", "Altro"])
        layout.addWidget(QLabel("Valuta"))
        layout.addWidget(self.currency_combo)

        # Descrizione (2-3 righe)
        self.description_edit = QTextEdit()
        self.description_edit.setFixedHeight(48)
        layout.addWidget(QLabel("Descrizione"))
        layout.addWidget(self.description_edit)

        layout.addStretch(1)
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                border-radius: 6px;
                border: 1px solid #ddd;
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
