import cv2
import numpy as np
import math

def resize(img):
    # Resize
    height, width = img.shape[:2]
    scale_ratio = 600.0 / width
    resize = cv2.resize(img, (int(scale_ratio * width), int(scale_ratio * height)),
                        interpolation=cv2.INTER_CUBIC)
    return resize

def color_filter(image):
    """ Make color filter for given image
    Args:
        image (image) Image to get its color filter
    Returns:
        (image,image) Tuple of images, borders highlighted and color filter mask
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_limit = 56  # Valor mínimo 20%
    s_limit = 56  # Saturación máxima 20%
    lower_white = np.array([0, 0, v_limit], dtype=np.uint8)
    upper_white = np.array([255, s_limit, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    mask_dilated = cv2.dilate(mask, None, iterations=2)
    return cv2.bitwise_and(image, image, mask=mask_dilated), mask

def rotate(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = edge_detection(gray)
    edges = cv2.bitwise_and(edges, edges, mask=mask)
    img_aux = img.copy()
    # Ref. http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)
    if lines is not None:
        for rho, theta in lines[0]:
            pass
        width, height = img_aux.shape[:2]
        # Ref. http://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/
        M = cv2.getRotationMatrix2D((height / 2, width / 2), (math.degrees(theta)), 1.0)
        img_aux = cv2.warpAffine(img, M, (height, width))
    return img_aux

def edge_detection(image):
    """ Extracts edges from image
    Args:
        image (gray scale image) Image to rotate
    Returns:
        (image) Image with borders highlighted
    """
    # Ref. http://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
    # Ref. http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html#why-laplacian-is-a-high-pass-filter
    # Eliminar ruido
    # Sal y pimienta
    blur = cv2.medianBlur(image, 7)
    # Filtro bilateral (que preserva bordes)
    smooth = cv2.bilateralFilter(blur, 11, 17, 17)
    # Mejorar contraste
    clahe = cv2.createCLAHE(clipLimit=2)
    enh = clahe.apply(smooth)
    # Obtener gradientes en ambos ejes
    grad_x = cv2.Sobel(enh, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_y = cv2.Sobel(enh, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    # Restar gradientes para obtener bordes en ambos ejes
    gradient = cv2.subtract(grad_x, grad_y)
    # Obtener valores absolutos para obtener imagen válida [0,255]
    edges = cv2.convertScaleAbs(gradient)
    # Segmentar uso umbralización para eliminar ruido de detección de bordes
    # Ref. http://docs.opencv.org/3.2.0/d7/d4d/tutorial_py_thresholding.html
    smoothed_edges = cv2.GaussianBlur(edges, (3, 3), 0)
    # Ref. http://docs.opencv.org/2.4/doc/tutorials/imgproc/threshold/threshold.html
    (_, thresh) = cv2.threshold(smoothed_edges, 100, 255, cv2.THRESH_BINARY)
    res = thresh
    return res


def connected_components(edges, mask=None, size_correction=0):
    """ Returns connected components optimized for qrcodes detection
    Args:
        edges (gray scale image) Image to rotate
        mask (binary image) Mask to apply after processing
        size_correction (int) Distance from where the make was made
    Returns:
        (image) Image with connected components differentiated by intensity
    """
    strut_x = size_correction
    strut_y = size_correction
    if mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=mask)
    # Operaciones morfológicas: closing & erosion
    # Ref. http://docs.opencv.org/2.4/doc/tutorials/imgproc/opening_closing_hats/opening_closing_hats.html
    closing_mask = cv2.getStructuringElement(cv2.MORPH_RECT, (strut_x, strut_y))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, closing_mask)
    # Ref. http://docs.opencv.org/2.4/doc/tutorials/imgproc/erosion_dilatation/erosion_dilatation.html
    eroded = cv2.erode(closed, None, iterations=15)
    dilated = cv2.dilate(eroded, None, iterations=18)
    # Detectar componentes conectados
    # Ref. http://aishack.in/tutorials/labelling-connected-components-example/
    connected = cv2.connectedComponents(dilated)
    # Asignar a cada uno de las componentes un valor diferenciador
    components = np.uint8((connected[1] * 255) / connected[0])
    res = components
    return res


def qrcode_detection(connected_component, original_img):
    """ Returns image with qrcode centered
    Args:
        connected_component (gray scale image) Connected components image
        original_img (image) Image to extract original qrcode
    Returns:
        (image, image) Tuple of images qrcode centered image and original image with qrcode highlighted
    """
    qrcode_img = None
    # Ref. http://docs.opencv.org/3.2.0/d4/d73/tutorial_py_contours_begin.html
    (_, contours, _) = cv2.findContours(connected_component.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Solo contorno con mayor área
    c = sorted(contours, key=cv2.contourArea, reverse=True)[0] 
    rect = cv2.minAreaRect(c)
    rotation = rect[2]
    # Definir límites de contorno
    # Ref. http://opencvpython.blogspot.com.es/2012/06/contours-2-brotherhood.html
    c_start_x, c_start_y, c_wide, c_height = cv2.boundingRect(c)
    # Recortar código 
    qrcode_img = np.copy(original_img[c_start_y:c_start_y + c_height, c_start_x:c_start_x + c_wide])
    # Rotar código
    width, height = qrcode_img.shape[:2]
    M = cv2.getRotationMatrix2D((height / 2, width / 2), (rotation), 1.0)
    qrcode_img = cv2.warpAffine(qrcode_img, M, (height, width))
    # Dibujar contorno
    # Ref. http://docs.opencv.org/2.4.2/modules/core/doc/drawing_functions.html#drawcontours
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    qrcode_highlighted = cv2.drawContours(original_img.copy(),[box],0,(255,255,255),2)
    res = qrcode_img
    return res, qrcode_highlighted


def qrcode_postprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 11, 17, 17)
    clahe = cv2.createCLAHE(clipLimit=2)
    contrast = clahe.apply(smooth)
    _, th = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    res = th
    return res

def qrcode_orientationdetection(img, img_name):
    
    (_, contours, hierarchy) = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_highlighted = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

    qr_candidates = []    
    for c in range(len(contours)):
        if hierarchy[0][c][2] == -1 and hierarchy[0][c][3] != -1:
            candidate = []
            candidate.append(c)
            childCount = 1
            nextParent = hierarchy[0][c][3]
            while nextParent != -1 and childCount < 3:
                childCount += 1
                cc = nextParent
                candidate.append(cc)
                nextParent = hierarchy[0][cc][3]
            if childCount == 3:
                qr_candidates.append(candidate)
        
    passed = []
        
    for cand in qr_candidates:
        previous_area = 0
        check_passed = True
        for c in range(len(cand)):
            area = cv2.contourArea(contours[cand[c]])
            if previous_area != 0 and (area/previous_area > 4 or area/previous_area < 1.5):
                check_passed = False
                break
            previous_area = area
            
        if check_passed is True:
            passed.append(cand)
            for c in range(len(cand)):
                rect = cv2.minAreaRect(contours[cand[c]])
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                contours_highlighted = cv2.drawContours(contours_highlighted,[box],0,(0,127*c,255/(c+1)),1)
    
    if len(passed) == 3:
        cv2.imshow(img_name, resize(contours_highlighted))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return contours, passed
    else:
        return
    
def qrcode_reorientate(img, contours, blocks):
    block_centers = []
    for b in blocks:
        (x,y),radius = cv2.minEnclosingCircle(contours[b[0]])
        block_centers.append((x,y))
    
    a = np.array((block_centers[0][0], block_centers[0][1]))
    b = np.array((block_centers[1][0], block_centers[1][1]))
    c = np.array((block_centers[2][0], block_centers[2][1]))
    
    distAB = np.linalg.norm(a-b)
    distAC = np.linalg.norm(a-c)
    distBC = np.linalg.norm(b-c)
    
    if distAB > distAC and distAB > distBC:
        pivot = c
        center = (a+b)/2
    if distAC > distAB and distAC > distBC:
        pivot = b
        center = (a+c)/2
    if distBC > distAB and distBC > distAC:
        pivot = a
        center = (b+c)/2

    if pivot[0] - center[0] > 0:
        x_orientation = 1
    else:
        x_orientation = -1
    if pivot[1] - center[1] > 0:
        y_orientation = 1
    else:
        y_orientation = -1
            
    if x_orientation == -1:
        if y_orientation == -1:
            return img
        else:
            angle = 270
    else:
        if y_orientation == -1:
            angle = 90
        else:
            angle = 180
                
    width, height = img.shape
    M = cv2.getRotationMatrix2D((height / 2, width / 2), (angle), 1.0)
    res = cv2.warpAffine(img, M, (height, width))
    return res
    