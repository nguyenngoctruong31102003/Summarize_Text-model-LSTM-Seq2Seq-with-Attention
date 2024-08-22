import sys
from PyQt5 import QtCore, QtWidgets
from summary_text import TextSummarizer, build_Model

# Tạo lựa chọn cho người dùng tùy biến
class InitialWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Chọn Lựa Chọn")
        self.setGeometry(400, 400, 550, 360)

        self.centralwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralwidget)
        
        self.TextLabel = QtWidgets.QLabel("Lựa Chọn Chức Năng Bạn Cần!", self.centralwidget)
        self.TextLabel.setGeometry(QtCore.QRect(100, 30, 400, 50))
        self.setLabelFont(self.TextLabel)
        self.TextLabel.setStyleSheet("color: #FF0000;")

        self.trainButton = QtWidgets.QPushButton("Lựa Chọn 1: Train Mô Hình", self.centralwidget)
        self.trainButton.setGeometry(QtCore.QRect(150, 120, 250, 50))
        self.trainButton.clicked.connect(self.trainModel)
        self.setButtonFont(self.trainButton)

        self.skipButton = QtWidgets.QPushButton("Lựa Chọn 2: Summarize Text", self.centralwidget)
        self.skipButton.setGeometry(QtCore.QRect(150, 190, 250, 50))
        self.skipButton.clicked.connect(self.runGUI)
        self.setButtonFont(self.skipButton)
        
        # Nút Thoát Khỏi chưong trình
        self.exitButton_1 = QtWidgets.QPushButton("Exit", self.centralwidget)
        self.exitButton_1.setGeometry(QtCore.QRect(150, 260, 121, 51))
        self.exitButton_1.clicked.connect(self.close)
        self.setButtonFont(self.exitButton_1)
        self.exitButton_1.setStyleSheet("color: #FF3300;")
        
        self.authorLabel = QtWidgets.QLabel("By Nguyễn Ngọc Trường", self.centralwidget)
        self.authorLabel.setGeometry(QtCore.QRect(350, 320, 200, 41))
        self.authorLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.setauthorFont(self.authorLabel)
        self.authorLabel.setStyleSheet("color: #00CC00;")
    
    def setLabelFont(self, label):
        font = label.font()
        font.setPointSize(16)
        label.setFont(font)
    
    def setauthorFont(self, label):
        font = label.font()
        font.setPointSize(10)
        label.setFont(font)

    def setButtonFont(self, button):
        font = button.font()
        font.setPointSize(11)
        button.setFont(font)
    
    def trainModel(self):
        self.close()
        w2v_path = 'Model/vi.vec' 
        data_dir = 'Data/Train' 
        stopwords_path = 'Data/vietnamese-stopwords.txt'
        tokenizer_path = 'Model/tokenizer.pkl'
        model_builder = build_Model(w2v_path, stopwords_path, data_dir)
        model_builder.train_model(model_path='Model/seq2seq_model.h5', tokenizer_path=tokenizer_path)  # Huấn luyện mô hình

    def runGUI(self):
        self.close()
        self.mainWindow = MainWindow()
        self.mainWindow.show()

# Lớp Chính để xử lý việc tóm tắt văn bản
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.summarizer = TextSummarizer('Model/vi.vec','Data/vietnamese-stopwords.txt')
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Summary Tool")
        self.setGeometry(100, 100, 990, 720)  # Đặt kích thước cửa sổ chính

        self.centralwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralwidget)

        # Tạo các nút và nhãn
        # Nút Open File 
        self.openFileButton = QtWidgets.QPushButton("Open File", self.centralwidget)
        self.openFileButton.setGeometry(QtCore.QRect(840, 180, 121, 51))
        self.openFileButton.clicked.connect(self.openFile)
        self.openFileButton.setStyleSheet("color: #0033FF;")
        self.setButtonFont(self.openFileButton)
        
        # Nút Xử lý để tóm tắt văn bản
        self.processingButton = QtWidgets.QPushButton("Processing", self.centralwidget)
        self.processingButton.setGeometry(QtCore.QRect(840, 260, 121, 51))
        self.processingButton.clicked.connect(self.processText)
        self.processingButton.setStyleSheet("color: #009900;")
        self.setButtonFont(self.processingButton)
        
        # Hộp chọn số lượng câu trong bản tóm tắt
        self.TextLabel_SpinBox = QtWidgets.QLabel("Số Câu Tóm Tắt =>", self.centralwidget)
        self.TextLabel_SpinBox.setGeometry(QtCore.QRect(660, 350, 200, 31)) # Vị trí TextLabel_SpinBox
        self.setTextLabel_SpinBox(self.TextLabel_SpinBox)
        self.sentenceSpinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.sentenceSpinBox.setGeometry(QtCore.QRect(840, 345, 121, 40)) # Vị trị SpinBox
        self.sentenceSpinBox.setMinimum(5)
        self.sentenceSpinBox.setMaximum(25)  # giá trị câu tóm tắt tối đa 
        self.sentenceSpinBox.setValue(5)  # Giá trị câu tóm tắt mặc định
        
        # Nút Thoát Khỏi chưong trình
        self.exitButton = QtWidgets.QPushButton("Exit", self.centralwidget)
        self.exitButton.setGeometry(QtCore.QRect(840, 440, 121, 51))
        self.exitButton.clicked.connect(self.close)
        self.setButtonFont(self.exitButton)
        self.exitButton.setStyleSheet("color: #FF3300;")

        self.originalTextLabel = QtWidgets.QLabel("Văn Bản Gốc", self.centralwidget)
        self.originalTextLabel.setGeometry(QtCore.QRect(80, 30, 220, 31))
        self.originalTextLabel.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.originalTextLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.originalTextLabel.setStyleSheet("color: #FF9900;")
        self.setLabelFont(self.originalTextLabel)

        self.originalTextBox = QtWidgets.QTextEdit(self.centralwidget)
        self.originalTextBox.setGeometry(QtCore.QRect(80, 70, 711, 271))
        self.setTextBoxFont(self.originalTextBox)

        self.processedTextLabel = QtWidgets.QLabel("Văn Bản Đã Tóm Tắt", self.centralwidget)
        self.processedTextLabel.setGeometry(QtCore.QRect(80, 350, 220, 31))
        self.processedTextLabel.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.processedTextLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.processedTextLabel.setStyleSheet("color: #FF9900;")
        self.setLabelFont(self.processedTextLabel)

        self.processedTextBox = QtWidgets.QTextEdit(self.centralwidget)
        self.processedTextBox.setGeometry(QtCore.QRect(80, 390, 711, 271))
        self.setTextBoxFont(self.processedTextBox)
        
        # Tính toán số từ (Word_count) đã giảm khi tóm tắt
        # Tạo QLabel để hiển thị thông tin về số từ đã giảm
        self.reductionLabel = QtWidgets.QLabel("", self.centralwidget)
        self.reductionLabel.setGeometry(QtCore.QRect(80, 670, 400, 31))
        self.reductionLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.reductionLabel.setStyleSheet("color: #339900;")
        self.setLabelFont(self.reductionLabel)

        
        self.authorLabel = QtWidgets.QLabel("By Nguyễn Ngọc Trường", self.centralwidget)
        self.authorLabel.setGeometry(QtCore.QRect(800, 670, 200, 41))
        self.authorLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.setauthorFont(self.authorLabel)
        self.authorLabel.setStyleSheet("color: #00CC00;")

    def setButtonFont(self, button):
        font = button.font()
        font.setPointSize(12)
        button.setFont(font)
    
    def setTextLabel_SpinBox(self, button):
        font = button.font()
        font.setPointSize(10)
        button.setFont(font)

    def setLabelFont(self, label):
        font = label.font()
        font.setPointSize(12)
        label.setFont(font)
    
    def setauthorFont(self, label):
        font = label.font()
        font.setPointSize(10)
        label.setFont(font)

    def setTextBoxFont(self, text_box):
        font = text_box.font()
        font.setPointSize(12)
        text_box.setFont(font)
        
    def openFile(self):
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Text File", "", "Text Files (*.txt);;All Files (*)", options=options)
        if fileName:
            try:
                with open(fileName, 'r', encoding='utf-8') as file:
                    text = file.read()
                    self.originalTextBox.setText(text)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Warning!!!", f"Không thể mở tệp:\n{str(e)}")
            
    def processText(self):
        original_text = self.originalTextBox.toPlainText()
        if not original_text.strip():
            QtWidgets.QMessageBox.warning(self, "Warning", "Không được để trống văn bản. Vui lòng điền nội dung!")
        else:
            num_sentences = self.sentenceSpinBox.value()
            
            # Đường dẫn tới mô hình và tokenizer
            model_path = 'Model/seq2seq_model.h5'
            tokenizer_path = 'Model/tokenizer.pkl'
            
            # Gọi hàm summarize_text với các tham số đầy đủ
            summarized_text = self.summarizer.summarize_text(original_text, model_path, tokenizer_path, num_sentences)
            self.processedTextBox.setText(summarized_text)
            
            # Tính toán số từ
            original_word_count = len(original_text.split())
            summarized_word_count = len(summarized_text.split())
            
            # Tính phần trăm từ giảm
            reduction_percent = ((original_word_count - summarized_word_count) / original_word_count) * 100
            
            # Cập nhật nội dung của QLabel
            reduction_info = f"Text reduced by {reduction_percent:.2f}% ({original_word_count} to {summarized_word_count} words)"
            self.reductionLabel.setText(reduction_info)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    initialWindow = InitialWindow()
    initialWindow.show()
    sys.exit(app.exec_())