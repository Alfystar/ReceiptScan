from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QComboBox, QDateEdit, QDoubleSpinBox
from PyQt6.QtCore import QDate

class TransactionDetailsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 0)
        layout.setSpacing(8)

        # Titolo
        title = QLabel("<b>Dettagli Transazione</b>")
        title.setStyleSheet("font-size: 14px;")
        layout.addWidget(title)

        # Data
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())
        layout.addWidget(QLabel("Data"))
        layout.addWidget(self.date_edit)

        # Ammontare
        self.amount_edit = QDoubleSpinBox()
        self.amount_edit.setMaximum(1000000)
        self.amount_edit.setPrefix("€ ")
        layout.addWidget(QLabel("Ammontare"))
        layout.addWidget(self.amount_edit)

        # Tipo di pagamento
        self.payment_type = QComboBox()
        self.payment_type.addItems(["Contanti", "Carta", "Bancomat", "Altro"])
        layout.addWidget(QLabel("Tipo di pagamento"))
        layout.addWidget(self.payment_type)

        # Totale
        self.total_edit = QDoubleSpinBox()
        self.total_edit.setMaximum(1000000)
        self.total_edit.setPrefix("€ ")
        layout.addWidget(QLabel("Totale"))
        layout.addWidget(self.total_edit)

        # Descrizione
        self.description_edit = QLineEdit()
        layout.addWidget(QLabel("Descrizione"))
        layout.addWidget(self.description_edit)

        # Valuta
        self.currency_edit = QLineEdit()
        self.currency_edit.setPlaceholderText("EUR, USD, ...")
        layout.addWidget(QLabel("Valuta"))
        layout.addWidget(self.currency_edit)

        layout.addStretch(1)

    def get_details(self):
        return {
            "date": self.date_edit.date().toString("yyyy-MM-dd"),
            "amount": self.amount_edit.value(),
            "payment_type": self.payment_type.currentText(),
            "total": self.total_edit.value(),
            "description": self.description_edit.text(),
            "currency": self.currency_edit.text(),
        }

    def set_details(self, details):
        if "date" in details:
            self.date_edit.setDate(QDate.fromString(details["date"], "yyyy-MM-dd"))
        if "amount" in details:
            self.amount_edit.setValue(float(details["amount"]))
        if "payment_type" in details:
            idx = self.payment_type.findText(details["payment_type"])
            if idx >= 0:
                self.payment_type.setCurrentIndex(idx)
        if "total" in details:
            self.total_edit.setValue(float(details["total"]))
        if "description" in details:
            self.description_edit.setText(details["description"])
        if "currency" in details:
            self.currency_edit.setText(details["currency"])

