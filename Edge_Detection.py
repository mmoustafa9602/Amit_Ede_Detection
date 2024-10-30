import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
from PyQt5 import Qt, uic, QtWidgets


class Proj(QtWidgets.QDialog):
    def __init__(self) -> None:
        """Initialize"""
        # Loading UI form
        super(Proj, self).__init__()

        # self.setWindowTitle('Image Viewer')
        uic.loadUi('C:\mahmoud\My Work\Gasco\Machine learning and AI\computer vision\QT5 Projects/Edge_Detection_Pyqt.ui',
                   self)
        self.layout = QVBoxLayout()

        self.originview = QLabel(self.imglabel)
        self.originview.setFixedSize(291, 350)
        self.Show_Img.clicked.connect(self.openFileDialog)
       

        self.Resview = QLabel(self.Reslabel_1)
        self.Resview.setFixedSize(291, 350)
        self.Show_Res_Img_Canny.clicked.connect(self.applyCannyEdge)
        self.Show_Res_Img_perwit.clicked.connect(self.applyperwitEdge)
        self.image  = None


    def openFileDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                  "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.gif);;All Files (*)",
                                                  options=options)
        if fileName:
            self.image  = cv2.imread(fileName)
            self.displayImage(self.image, self.originview)

    def   applyCannyEdge(self):
        if self.image is None:
            print('No image loaded' )  
        else:
             gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
             gray_image_F = cv2.medianBlur(gray_image, 5)
             canny_edges = cv2.Canny(gray_image_F,int(self.Lth.value()), int(self.Uth.value()))
             self.displayImage(canny_edges, self.Resview)
    def   applyperwitEdge(self):
        if self.image is None:
            print('No image loaded' )  
        else:
             gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
             gray_image_F = cv2.medianBlur(gray_image, 5)
             prewitt_kernel_x = np.array([[ -1,  0,  1],
                                                  [ -1,  0,  1],
                                                  [ -1,  0,  1]])
             prewitt_kernel_y = np.array([[  1,  1,  1],
                                              [  0,  0,  0],
                                              [ -1, -1, -1]])
             # Apply the Prewitt operator (convolution)
             edges_x = cv2.filter2D(gray_image_F, -1, prewitt_kernel_x)  # Horizontal edges
             edges_y = cv2.filter2D(gray_image_F, -1, prewitt_kernel_y)  # Vertical edges
             # Combine the two gradients (X and Y) to get the edge magnitude
             perwit_edges = cv2.addWeighted(edges_x, 0.5, edges_y, 0.5, 0)
             self.displayImage(perwit_edges, self.Resview)

    

        

    def displayImage(self, img, label):
        if len(img.shape)==2:
            height,width =img.shape
            bytesPerLine =width
            qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
            label.setPixmap(QPixmap.fromImage(qImg))
        else:
            height, width, channel = img.shape
            bytesPerLine = 3 * width
            qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            label.setPixmap(QPixmap.fromImage(qImg)


)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = Proj()
    viewer.show()
    sys.exit(app.exec_())

