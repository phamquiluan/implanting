"""
Author : Luan Pham
Email  : phamquiluan@gmail.com
Date   : 26 Feb 2022
"""
import cv2
import numpy as np

def color_quantization(image, K):
    Z = image.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    return res2


def show(img, name="disp", width=1000):
    """
    name: name of window, should be name of img
    img: source of img, should in type ndarray
    """
    cv2.namedWindow(name, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(name, width, 1000)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)
    

def main():
    image = cv2.imread("/home/lulu/Downloads/274511155_1680515062297432_6796541277625137443_n.jpg")

    quantized_image = color_quantization(image, K=6)

    bin_image = auto_canny(quantized_image)

    # crop bounding box
    cnts = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts, _ = cnts if len(cnts) == 2 else cnts[1:]

    max_cnt = max(cnts, key=lambda x:cv2.contourArea(x))
    x, y, w, h = cv2.boundingRect(max_cnt)
    x, y, w, h = x - 2, y - 2, w + 4, h + 4
    

    cropped_image = image[y : y + h, x : x + w]
    bin_image = bin_image[y : y + h, x : x + w]
    quantized_image = quantized_image[y : y + h, x : x + w]
    cv2.imwrite("output/01_bin_image.png", bin_image)
    cv2.imwrite("output/02_quantized_image.png", quantized_image)

    
    # remove characters
    bin_image = cv2.dilate(bin_image, np.ones((3, 3)))
    cnts = cv2.findContours(bin_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnts, _ = cnts if len(cnts) == 2 else cnts[1:]

    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        
        if y + h < bin_image.shape[0] // 10:
            cv2.drawContours(bin_image, [cnt], -1, 0, -1)

    cv2.imwrite("output/03_bin_image_no_char.png", bin_image)


    # extract vertical component
    vline = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, np.ones((40, 1)))
    cnts = cv2.findContours(vline, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnts, _ = cnts if len(cnts) == 2 else cnts[1:]
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < vline.shape[0] // 4:
            cv2.drawContours(vline, [cnt], -1, 0, -1)


    cv2.imwrite("output/04_vertical_line.png", vline)
    
    # extract horizontal component
    hline = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, np.ones((1, 40)))
    cv2.imwrite("output/05_horizontal_line.png", hline)

    # extract curve component

    curve_image = bin_image - vline - cv2.dilate(hline, np.ones((3, 3)))
    # curve_image = bin_image - cv2.dilate(vline, np.ones((3, 3))) - cv2.dilate(hline, np.ones((3, 3)))
    curve_image = cv2.morphologyEx(curve_image, cv2.MORPH_CLOSE, np.ones((5, 5)))
    # show(curve_image)
    cv2.imwrite("output/06_curve_line.png", curve_image)


    # find sep position by vert line, x
    cnts = cv2.findContours(vline, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnts, _ = cnts if len(cnts) == 2 else cnts[1:]
    color_curve_image = cv2.cvtColor(curve_image, cv2.COLOR_GRAY2BGR)
    seps = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.line(color_curve_image, (x, 0), (x, color_curve_image.shape[0]), (0, 255, 0), 2)
        seps.append(x)
    
    seps = sorted(list(set(seps)))

    cv2.imwrite("output/07_curve_line_with_sep.png", color_curve_image)

    total = np.sum(curve_image)  # 11758
    print(seps)
    for idx, sep in enumerate(seps[1:]):
        r_image = curve_image[:, seps[idx]:sep]
        print(np.sum(r_image) / total * 100)

    # find percentage


if __name__ == "__main__":
    main()

