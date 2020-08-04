import numpy as np
import cv2

import source.helpers as helpers
import tensorflow as tf


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, _, logs={}):
        if logs.get('accuracy') > 0.995:
            self.model.stop_training = True


def process_image(path: str):
    # Load image
    try:
        img = cv2.imread(path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 11, 3)
        img = 255 - img
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        contours, h = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        img = img[y:y + h, x:x + w]
        img = cv2.resize(img, (288, 288), interpolation=cv2.INTER_AREA)
        img2 = img.copy()
        largest = helpers.largest4SideContour(img2)
        app = helpers.approx(largest)
        if app is None:
            return img
        corners = helpers.get_rectangle_corners(app)
        img = helpers.warp_perspective(corners, img)
        return img
    except Exception:
        return None

def ground_truth(path: str):
    with(open(path, 'r')) as fin:
        lines = fin.readlines()
        truth = []

        # First line contains the details of phone
        # Second line contains size of source image
        # Skipping both these lines
        for line in lines[2:]:
            for value in line.split():
                truth.append(int(value))

        return np.array(truth)


def clean(cell):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, kernel)
    cell = 255 * (cell / 130)
    return cell


def centerDigit(digit):
    digit = helpers.centerX(digit)
    digit = helpers.centerY(digit)
    return digit


def get_cells(sudoku, size=288):
    sudoku = sudoku / 255.
    cells = []
    cell_shape = size // 9
    for i in range(9):
        i = cell_shape * i
        for j in range(9):
            j = j * cell_shape
            cell = sudoku[i: i+cell_shape, j: j+cell_shape]
            cell = clean(cell)
            cell = centerDigit(cell)
            cells.append(cell)

    return np.array(cells)


def compare_outputs(pred, truth):
    errors = 0
    for i in range(len(pred)):
        if pred[i] != truth[i]:
            errors += 1
            break

    if errors:
        return 0
    return 1


def get_sudoku(path: str, model):
    sudoku = process_image(path)

    if sudoku is None:
        raise Exception

    cells = get_cells(sudoku, 288)
    cells = np.reshape(cells, (cells.shape[0], 32, 32, 1))

    # for cell in cells:
    #   cell = cell.reshape((32, 32))
    #   plt.imshow(cell)
    #   plt.show()
    #   v_pred = model.predict(cell.reshape((1, 32, 32, 1)))
    #   print('Predicted: ',np.argmax(v_pred, axis=1))
    #   input("Continue")

    v_pred = model.predict(cells)
    v_pred = np.argmax(v_pred, axis=1)
    sudoku_ext = np.reshape(v_pred, (9, 9))
    return sudoku_ext
