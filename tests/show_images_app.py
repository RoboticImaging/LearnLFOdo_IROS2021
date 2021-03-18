# This script launches a GUI to visualize results
#
# Author and Maintainer: Tejaswi Digumarti (tejaswi.digumarti@sydney.edu.au)

import sys
import os

import cv2
import numpy as np

from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QApplication
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tests.MainWindow import Ui_MainWindow


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=2, height=2, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        # make adjustments to remove whitespace
        self.axes.set_axis_off()
        self.axes.margins(0, 0)

        super(MplCanvas, self).__init__(fig)
        self.setParent(parent)
        self.plot_ref = None
        self.fig = fig
        self.cb = None

    def update_figure(self, figure_array, cmap=None):
        if self.plot_ref is None:
            if self.cb is not None:
                self.cb.remove()
            if cmap is None:
                plot_refs = self.axes.imshow(figure_array)
                self.fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, hspace=0, wspace=0)
            else:
                plot_refs = self.axes.imshow(figure_array, cmap=cmap,
                                             vmin=np.min(figure_array), vmax=np.max(figure_array))
                self.fig.subplots_adjust(left=0.1, right=0.9, bottom=0.01, top=0.99, hspace=0, wspace=0)

                divider = make_axes_locatable(self.axes)
                cax = divider.append_axes("right", size="5%", pad=0.05)

                self.cb = self.fig.colorbar(plot_refs, cax=cax)
            self.plot_ref = plot_refs
        else:
            self.plot_ref.set_data(figure_array)
            if self.cb is not None:
                self.plot_ref.set_clim(vmin=np.min(figure_array), vmax=np.max(figure_array))

        self.draw()

    def reset_ref(self):
        self.plot_ref = None

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # setup UI from the UI file
        self.setupUi(self)
        # initialize default values of paths
        # self.root_folder = "/home/dtejaswi/Desktop/joseph_daniel/ral/"
        self.root_folder = "/home/dtejaswi/tensorboard_hpc/artemis_test_b16"
        self.lineEdit_root.setText(self.root_folder)
        # initialize default value sequence name
        self.seq_name = "seq16_epoch_40"
        self.lineEdit_name.setText(self.seq_name)
        # initialize the encoding and mode
        self.enc = self.buttonGroup_enc.checkedButton().text()
        self.mode = self.buttonGroup_mode.checkedButton().text()
        # update complete path to images
        self.img_root = None
        self.update_img_root()
        # initialize image ID to the second image (1) - index starts at 0
        self.img_id = 1
        self.update_label_img_id()
        self.lineEdit_curr_id.setText(str(self.img_id))
        self.prev_valid_image_id = 1

        # setup matplotlib canvases for all the widgets
        # --> input images
        self.canvas_img_tminus = self.add_mpl_widget(parent=self.widget_img_tminus)
        self.canvas_img_t = self.add_mpl_widget(parent=self.widget_img_t)
        self.canvas_img_tplus = self.add_mpl_widget(parent=self.widget_img_tplus)
        # --> disparity and depth predictions of img_t
        self.canvas_disp_t = self.add_mpl_widget(parent=self.widget_disp_t)
        self.canvas_depth_t = self.add_mpl_widget(parent=self.widget_depth_t)
        # --> warps
        self.canvas_img_1 = self.add_mpl_widget(parent=self.widget_img_1)
        self.canvas_img_2 = self.add_mpl_widget(parent=self.widget_img_2)
        self.canvas_img_3 = self.add_mpl_widget(parent=self.widget_img_3)
        self.canvas_img_4 = self.add_mpl_widget(parent=self.widget_img_4)
        self.canvas_img_5 = self.add_mpl_widget(parent=self.widget_img_5)
        self.canvas_img_6 = self.add_mpl_widget(parent=self.widget_img_6)

        # draw the plots
        self.show()

        # define all the combo boxes
        self.image_types = ["ref_minus", "ref_plus", "tgt", "warp_minus", "warp_plus", "diff_minus", "diff_plus"]
        self.comboBox_ctrl_frame1_img_1.addItems(self.image_types)
        self.comboBox_ctrl_frame2_img_1.addItems(self.image_types)
        self.comboBox_ctrl_frame1_img_2.addItems(self.image_types)
        self.comboBox_ctrl_frame2_img_2.addItems(self.image_types)
        self.comboBox_ctrl_frame1_img_3.addItems(self.image_types)
        self.comboBox_ctrl_frame2_img_3.addItems(self.image_types)
        self.comboBox_ctrl_frame1_img_4.addItems(self.image_types)
        self.comboBox_ctrl_frame2_img_4.addItems(self.image_types)
        self.comboBox_ctrl_frame1_img_5.addItems(self.image_types)
        self.comboBox_ctrl_frame2_img_5.addItems(self.image_types)
        self.comboBox_ctrl_frame1_img_6.addItems(self.image_types)
        self.comboBox_ctrl_frame2_img_6.addItems(self.image_types)
        # initialize comboboxes
        self.comboBox_ctrl_frame1_img_1.setCurrentIndex(0)
        self.comboBox_ctrl_frame1_img_2.setCurrentIndex(1)
        self.comboBox_ctrl_frame1_img_3.setCurrentIndex(3)
        self.comboBox_ctrl_frame1_img_4.setCurrentIndex(4)
        self.comboBox_ctrl_frame1_img_5.setCurrentIndex(5)
        self.comboBox_ctrl_frame1_img_6.setCurrentIndex(6)
        self.comboBox_ctrl_frame2_img_1.setCurrentIndex(2)
        self.comboBox_ctrl_frame2_img_2.setCurrentIndex(2)
        self.comboBox_ctrl_frame2_img_3.setCurrentIndex(2)
        self.comboBox_ctrl_frame2_img_4.setCurrentIndex(2)
        self.comboBox_ctrl_frame2_img_5.setCurrentIndex(5)
        self.comboBox_ctrl_frame2_img_6.setCurrentIndex(6)

        # set all toggle states to True
        self.toggle_state_img_1 = True
        self.toggle_state_img_2 = True
        self.toggle_state_img_3 = True
        self.toggle_state_img_4 = True
        self.toggle_state_img_5 = True
        self.toggle_state_img_6 = True

        # initialize list for group toggle
        self.group_toggle_list = []

        # Connect signals to slots
        self.lineEdit_root.editingFinished.connect(self.update_root)
        self.lineEdit_name.editingFinished.connect(self.update_seq_name)
        self.buttonGroup_mode.buttonClicked.connect(self.update_mode)
        self.buttonGroup_enc.buttonClicked.connect(self.update_encoding)
        self.pushButton_plot.clicked.connect(self.update_plots)
        self.pushButton_prev.clicked.connect(self.decrease_img_id)
        self.pushButton_next.clicked.connect(self.increase_img_id)
        self.lineEdit_curr_id.editingFinished.connect(self.set_image_with_id)

        self.pushButton_ctrl_tgl_img_1.clicked.connect(self.toggle_img_1)
        self.pushButton_ctrl_tgl_img_2.clicked.connect(self.toggle_img_2)
        self.pushButton_ctrl_tgl_img_3.clicked.connect(self.toggle_img_3)
        self.pushButton_ctrl_tgl_img_4.clicked.connect(self.toggle_img_4)
        self.pushButton_ctrl_tgl_img_5.clicked.connect(self.toggle_img_5)
        self.pushButton_ctrl_tgl_img_6.clicked.connect(self.toggle_img_6)
        self.pushButton_tgl_group.clicked.connect(self.toggle_multiple)

        self.checkBox_ctrl_grp_tgl_img_1.clicked.connect(self.update_toggle_group_1)
        self.checkBox_ctrl_grp_tgl_img_2.clicked.connect(self.update_toggle_group_2)
        self.checkBox_ctrl_grp_tgl_img_3.clicked.connect(self.update_toggle_group_3)
        self.checkBox_ctrl_grp_tgl_img_4.clicked.connect(self.update_toggle_group_4)
        self.checkBox_ctrl_grp_tgl_img_5.clicked.connect(self.update_toggle_group_5)
        self.checkBox_ctrl_grp_tgl_img_6.clicked.connect(self.update_toggle_group_6)

        self.checkBox_ctrl_static_img_1.clicked.connect(self.update_static_mode_1)
        self.checkBox_ctrl_static_img_2.clicked.connect(self.update_static_mode_2)
        self.checkBox_ctrl_static_img_3.clicked.connect(self.update_static_mode_3)
        self.checkBox_ctrl_static_img_4.clicked.connect(self.update_static_mode_4)
        self.checkBox_ctrl_static_img_5.clicked.connect(self.update_static_mode_5)
        self.checkBox_ctrl_static_img_6.clicked.connect(self.update_static_mode_6)

    @staticmethod
    def add_mpl_widget(parent):
        """
        Adds a matplotlib canvas and a toolbar into the chosen parent widget
        :param parent: The widget into which the matplotlib canvas and toolbar have to be added
        :type parent: QWidget
        :return: Matplotlib canvas
        :rtype: MplCanvas
        """
        # set layout of the widget where
        parent.setLayout(QVBoxLayout())
        # the part where the image is drawn
        canvas = MplCanvas(parent=parent)
        # navigation toolbar for the image
        toolbar = NavigationToolbar2QT(canvas, parent=parent)
        parent.layout().addWidget(toolbar)
        parent.layout().addWidget(canvas)
        return canvas

    def update_img_root(self):
        if self.mode == "multiwarp-5" and self.enc == "monocular":
            self.statusbar.showMessage("Monocular is only valid in singlewarp", msecs=0)
            self.mode = "singlewarp"
            self.radioButton_singlewarp.click()

        self.img_root = os.path.join(self.root_folder, self.mode,
                                     self.enc, "results", self.seq_name)
        self.statusbar.showMessage("Img root: {}".format(self.img_root), msecs=0)

    def update_root(self):
        self.root_folder = self.lineEdit_root.text()
        self.pushButton_plot.setFocus()
        self.update_img_root()

    def update_seq_name(self):
        self.seq_name = self.lineEdit_name.text()
        self.update_img_root()

    def update_mode(self):
        self.mode = self.buttonGroup_mode.checkedButton().text()
        if self.mode == "multiwarp-5":
            self.radioButton_mono.setEnabled(False)
        else:
            self.radioButton_mono.setEnabled(True)
        self.canvas_depth_t.reset_ref()
        self.canvas_disp_t.reset_ref()
        self.update_plots()

    def update_encoding(self):
        self.enc = self.buttonGroup_enc.checkedButton().text()
        self.update_plots()

    def update_label_img_id(self):
        self.label_curr_id.setText(str(self.img_id))

    def update_lineEdit_img_id(self):
        self.lineEdit_curr_id.setText(str(self.img_id))

    def increase_img_id(self):
        self.prev_valid_image_id = self.img_id

        self.img_id += 1
        # if self.img_id > 67:    # TODO Hardcoded for now. Remove later
        #     self.img_id = 67
        self.update_label_img_id()
        self.update_lineEdit_img_id()
        self.update_plots()

    def decrease_img_id(self):
        self.prev_valid_image_id = self.img_id

        self.img_id -= 1
        # if self.img_id < 1:
        #     self.img_id = 1
        self.update_label_img_id()
        self.update_lineEdit_img_id()
        self.update_plots()

    def set_image_paths(self):
        # update image plots
        file_t = "{:06d}.png".format(self.img_id)
        file_tminus = "{:06d}.png".format(self.img_id - 1)
        file_tplus = "{:06d}.png".format(self.img_id + 1)

        path_tplus = os.path.join(self.img_root, file_tplus)
        path_tminus = os.path.join(self.img_root, file_tminus)
        path_t = os.path.join(self.img_root, file_t)

        return path_tminus, path_t, path_tplus

    def update_plots(self):
        self.update_img_root()

        path_tminus, path_t, path_tplus = self.set_image_paths()

        if not os.path.exists(path_tplus) or not os.path.exists(path_tminus):
            self.statusbar.showMessage("Reached extremity. Resetting to previous image ID", msecs=0)
            self.img_id = self.prev_valid_image_id
            self.update_label_img_id()
            self.update_lineEdit_img_id()
            path_tminus, path_t, path_tplus = self.set_image_paths()

        img_t = cv2.imread(path_t)
        img_tminus = cv2.imread(path_tminus)
        img_tplus = cv2.imread(path_tplus)

        self.canvas_img_tminus.update_figure(img_tminus)
        self.canvas_img_t.update_figure(img_t)
        self.canvas_img_tplus.update_figure(img_tplus)

        file_disp_t = "{:06d}.npy".format(self.img_id)
        disp_t, depth_t = self.load_disp_depth(os.path.join(self.img_root, "disp", file_disp_t))

        self.canvas_disp_t.update_figure(disp_t, cmap=matplotlib.cm.get_cmap("viridis").reversed())
        self.canvas_depth_t.update_figure(depth_t, cmap=matplotlib.cm.get_cmap("viridis").reversed())

        # update warp plots
        self.plot_warp_images()

    def plot_warp_images(self):
        self.update_canvas(self.canvas_img_1, self.checkBox_ctrl_static_img_1, self.toggle_state_img_1,
                           self.comboBox_ctrl_frame1_img_1, self.comboBox_ctrl_frame2_img_1, self.label_img_1)

        self.update_canvas(self.canvas_img_2, self.checkBox_ctrl_static_img_2, self.toggle_state_img_2,
                           self.comboBox_ctrl_frame1_img_2, self.comboBox_ctrl_frame2_img_2, self.label_img_2)

        self.update_canvas(self.canvas_img_3, self.checkBox_ctrl_static_img_3, self.toggle_state_img_3,
                           self.comboBox_ctrl_frame1_img_3, self.comboBox_ctrl_frame2_img_3, self.label_img_3)

        self.update_canvas(self.canvas_img_4, self.checkBox_ctrl_static_img_4, self.toggle_state_img_4,
                           self.comboBox_ctrl_frame1_img_4, self.comboBox_ctrl_frame2_img_4, self.label_img_4)

        self.update_canvas(self.canvas_img_5, self.checkBox_ctrl_static_img_5, self.toggle_state_img_5,
                           self.comboBox_ctrl_frame1_img_5, self.comboBox_ctrl_frame2_img_5, self.label_img_5)

        self.update_canvas(self.canvas_img_6, self.checkBox_ctrl_static_img_6, self.toggle_state_img_6,
                           self.comboBox_ctrl_frame1_img_6, self.comboBox_ctrl_frame2_img_6, self.label_img_6)

    def update_canvas(self, canvas, static_state, toggle_state, frame_1, frame_2, label_title):
        """
        Updates the matplotlib canvas based on the state of the controls and the toggle state
        :param canvas: Matplotlib canvas for drawing images
        :type canvas: MplCanvas
        :param static_state: Checkbox indicating if this canvas is static or not
        :type static_state: QCheckBox
        :param toggle_state: Boolean indicating the toggle state for this canvas
        :type toggle_state: bool
        :param frame_1: ComboBox with frame to use for static and toggle_state False
        :type frame_1: QComboBox
        :param frame_2: ComboBox with frame to use for toggle_state True if static_state is unchecked
        :type frame_2: QComboBox
        :param label_title: Label that is the title of the canvas
        :type label_title: QLabel
        :return: Nothing
        :rtype: None
        """
        if static_state.isChecked() or (not static_state.isChecked() and toggle_state):
            frame = frame_1
        else:
            frame = frame_2
        cm = None
        if frame.currentText() == self.image_types[0]:
            # reference t-1
            img = cv2.imread(os.path.join(self.img_root, "{:06d}.png".format(self.img_id - 1)))
            title = "reference t-1"
        elif frame.currentText() == self.image_types[1]:
            # reference t+1
            img = cv2.imread(os.path.join(self.img_root, "{:06d}.png".format(self.img_id + 1)))
            title = "reference t+1"
        elif frame.currentText() == self.image_types[2]:
            # target
            img = cv2.imread(os.path.join(self.img_root, "{:06d}.png".format(self.img_id)))
            title = "target t"
        elif frame.currentText() == self.image_types[3]:
            # warped t-1
            img = cv2.imread(os.path.join(self.img_root, "warps", "{:06d}_0.png".format(self.img_id)))
            title = "reference t-1 warped to t"
        elif frame.currentText() == self.image_types[4]:
            # warped t+1
            img = cv2.imread(os.path.join(self.img_root, "warps", "{:06d}_0.png".format(self.img_id)))
            title = "reference t+1 warped to t"
        elif frame.currentText() == self.image_types[5]:
            print(self.img_id)
            # diff t-1
            print(os.path.join(self.img_root, "diffs", "{:06d}_0.npy".format(self.img_id)))
            img = np.load(os.path.join(self.img_root, "diffs", "{:06d}_0.npy".format(self.img_id)))
            print(np.min(img), np.max(img))
            img = abs(img)
            title = "absolute difference |(warped reference t-1) - (target t)|"
            cm = matplotlib.cm.get_cmap("viridis")
        else:
            # diff t+1
            img = np.load(os.path.join(self.img_root, "diffs", "{:06d}_0.npy".format(self.img_id)))
            img = abs(img)
            title = "absolute difference |(warped reference t+1) - (target t)|"
            cm = matplotlib.cm.get_cmap("viridis")
        canvas.update_figure(img, cmap=cm)
        label_title.setText(title)

    @staticmethod
    def load_disp_depth(path, border_crop=5):
        """
        Reads a .npy file containing disparity output and converts it into a depth image output
        :param path: path to the .npy file
        :type path: str
        :param border_crop: number of pixels by which to crop the border
        :type border_crop: int
        :return: depth image where the pixel values are in metres
        :rtype: numpy array
        """
        img_disp = np.load(path)
        img_depth = 1.0 / img_disp
        h, w = img_disp.shape
        img_disp = img_disp[border_crop:h - border_crop, border_crop:w - border_crop]
        img_depth = img_depth[border_crop:h - border_crop, border_crop:w - border_crop]
        return img_disp, img_depth

    def keyPressEvent(self, event) -> None:
        """
        Handles Key presses
        :param event: The key press event
        :type event: QKeyEvent
        :return: Nothing
        :rtype: None
        """
        if event.key() == Qt.Key_Comma:
            self.decrease_img_id()
        elif event.key() == Qt.Key_Period:
            self.increase_img_id()
        elif event.key() == Qt.Key_T:
            self.toggle_multiple()
        elif event.key() == Qt.Key_1:
            if not self.checkBox_ctrl_static_img_1.isChecked():
                self.toggle_img_1()
        elif event.key() == Qt.Key_2:
            if not self.checkBox_ctrl_static_img_2.isChecked():
                self.toggle_img_2()
        elif event.key() == Qt.Key_3:
            if not self.checkBox_ctrl_static_img_3.isChecked():
                self.toggle_img_3()
        elif event.key() == Qt.Key_4:
            if not self.checkBox_ctrl_static_img_4.isChecked():
                self.toggle_img_4()
        elif event.key() == Qt.Key_5:
            if not self.checkBox_ctrl_static_img_5.isChecked():
                self.toggle_img_5()
        elif event.key() == Qt.Key_6:
            if not self.checkBox_ctrl_static_img_6.isChecked():
                self.toggle_img_6()

    def toggle_img_1(self):
        self.toggle_state_img_1 = not self.toggle_state_img_1
        self.update_canvas(self.canvas_img_1, self.checkBox_ctrl_static_img_1, self.toggle_state_img_1,
                           self.comboBox_ctrl_frame1_img_1, self.comboBox_ctrl_frame2_img_1, self.label_img_1)

    def toggle_img_2(self):
        self.toggle_state_img_2 = not self.toggle_state_img_2
        self.update_canvas(self.canvas_img_2, self.checkBox_ctrl_static_img_2, self.toggle_state_img_2,
                           self.comboBox_ctrl_frame1_img_2, self.comboBox_ctrl_frame2_img_2, self.label_img_2)

    def toggle_img_3(self):
        self.toggle_state_img_3 = not self.toggle_state_img_3
        self.update_canvas(self.canvas_img_3, self.checkBox_ctrl_static_img_3, self.toggle_state_img_3,
                           self.comboBox_ctrl_frame1_img_3, self.comboBox_ctrl_frame2_img_3, self.label_img_3)

    def toggle_img_4(self):
        self.toggle_state_img_4 = not self.toggle_state_img_4
        self.update_canvas(self.canvas_img_4, self.checkBox_ctrl_static_img_4, self.toggle_state_img_4,
                           self.comboBox_ctrl_frame1_img_4, self.comboBox_ctrl_frame2_img_4, self.label_img_4)

    def toggle_img_5(self):
        self.toggle_state_img_5 = not self.toggle_state_img_5
        self.update_canvas(self.canvas_img_5, self.checkBox_ctrl_static_img_5, self.toggle_state_img_5,
                           self.comboBox_ctrl_frame1_img_5, self.comboBox_ctrl_frame2_img_5, self.label_img_5)

    def toggle_img_6(self):
        self.toggle_state_img_6 = not self.toggle_state_img_6
        self.update_canvas(self.canvas_img_6, self.checkBox_ctrl_static_img_6, self.toggle_state_img_6,
                           self.comboBox_ctrl_frame1_img_6, self.comboBox_ctrl_frame2_img_6, self.label_img_6)

    def update_toggle_group_1(self):
        self.update_toggle_group("1", self.checkBox_ctrl_grp_tgl_img_1.isChecked())

    def update_toggle_group_2(self):
        self.update_toggle_group("2", self.checkBox_ctrl_grp_tgl_img_2.isChecked())

    def update_toggle_group_3(self):
        self.update_toggle_group("3", self.checkBox_ctrl_grp_tgl_img_3.isChecked())

    def update_toggle_group_4(self):
        self.update_toggle_group("4", self.checkBox_ctrl_grp_tgl_img_4.isChecked())

    def update_toggle_group_5(self):
        self.update_toggle_group("5", self.checkBox_ctrl_grp_tgl_img_5.isChecked())

    def update_toggle_group_6(self):
        self.update_toggle_group("6", self.checkBox_ctrl_grp_tgl_img_6.isChecked())

    def update_toggle_group(self, value, to_add):
        if to_add:
            if value not in self.group_toggle_list:
                self.group_toggle_list.append(value)
        else:
            if value in self.group_toggle_list:
                self.group_toggle_list.remove(value)

    def toggle_multiple(self):
        if "1" in self.group_toggle_list:
            self.toggle_img_1()
        if "2" in self.group_toggle_list:
            self.toggle_img_2()
        if "3" in self.group_toggle_list:
            self.toggle_img_3()
        if "4" in self.group_toggle_list:
            self.toggle_img_4()
        if "5" in self.group_toggle_list:
            self.toggle_img_5()
        if "6" in self.group_toggle_list:
            self.toggle_img_6()

    def update_static_mode_1(self):
        if self.checkBox_ctrl_static_img_1.isChecked():
            # disable buttons
            self.checkBox_ctrl_grp_tgl_img_1.setEnabled(False)
            self.pushButton_ctrl_tgl_img_1.setEnabled(False)
            # remove from toggle list
            self.update_toggle_group("1", False)
            # reset to initial toggle state
            self.toggle_state_img_1 = True
        else:
            # enable buttons
            self.checkBox_ctrl_grp_tgl_img_1.setEnabled(True)
            self.pushButton_ctrl_tgl_img_1.setEnabled(True)

    def update_static_mode_2(self):
        if self.checkBox_ctrl_static_img_2.isChecked():
            # disable buttons
            self.checkBox_ctrl_grp_tgl_img_2.setEnabled(False)
            self.pushButton_ctrl_tgl_img_2.setEnabled(False)
            # remove from toggle list
            self.update_toggle_group("2", False)
            # reset to initial toggle state
            self.toggle_state_img_2 = True
        else:
            # enable buttons
            self.checkBox_ctrl_grp_tgl_img_2.setEnabled(True)
            self.pushButton_ctrl_tgl_img_2.setEnabled(True)

    def update_static_mode_3(self):
        if self.checkBox_ctrl_static_img_3.isChecked():
            # disable buttons
            self.checkBox_ctrl_grp_tgl_img_3.setEnabled(False)
            self.pushButton_ctrl_tgl_img_3.setEnabled(False)
            # remove from toggle list
            self.update_toggle_group("3", False)
            # reset to initial toggle state
            self.toggle_state_img_3 = True
        else:
            # enable buttons
            self.checkBox_ctrl_grp_tgl_img_3.setEnabled(True)
            self.pushButton_ctrl_tgl_img_3.setEnabled(True)

    def update_static_mode_4(self):
        if self.checkBox_ctrl_static_img_4.isChecked():
            # disable buttons
            self.checkBox_ctrl_grp_tgl_img_4.setEnabled(False)
            self.pushButton_ctrl_tgl_img_4.setEnabled(False)
            # remove from toggle list
            self.update_toggle_group("4", False)
            # reset to initial toggle state
            self.toggle_state_img_4 = True
        else:
            # enable buttons
            self.checkBox_ctrl_grp_tgl_img_4.setEnabled(True)
            self.pushButton_ctrl_tgl_img_4.setEnabled(True)

    def update_static_mode_5(self):
        if self.checkBox_ctrl_static_img_5.isChecked():
            # disable buttons
            self.checkBox_ctrl_grp_tgl_img_5.setEnabled(False)
            self.pushButton_ctrl_tgl_img_5.setEnabled(False)
            # remove from toggle list
            self.update_toggle_group("5", False)
            # reset to initial toggle state
            self.toggle_state_img_5 = True
        else:
            # enable buttons
            self.checkBox_ctrl_grp_tgl_img_5.setEnabled(True)
            self.pushButton_ctrl_tgl_img_5.setEnabled(True)

    def update_static_mode_6(self):
        if self.checkBox_ctrl_static_img_6.isChecked():
            # disable buttons
            self.checkBox_ctrl_grp_tgl_img_6.setEnabled(False)
            self.pushButton_ctrl_tgl_img_6.setEnabled(False)
            # remove from toggle list
            self.update_toggle_group("6", False)
            # reset to initial toggle state
            self.toggle_state_img_6 = True
        else:
            # enable buttons
            self.checkBox_ctrl_grp_tgl_img_6.setEnabled(True)
            self.pushButton_ctrl_tgl_img_6.setEnabled(True)

    def set_image_with_id(self):
        self.img_id = int(self.lineEdit_curr_id.text())
        self.update_label_img_id()
        self.update_plots()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    # window.show()   # Important

    app.exec_()
