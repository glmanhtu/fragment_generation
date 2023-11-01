import numpy as np, cv2, math

def getEllipseKernel(a, b, theta):
    "create a kernel mask shaped as an ellipse with a as major axis (on x-axis), b as minor axis (on y-axis), and theta as orientation"
    w = max(a, b);
    h = max(a, b);
    if(~w%2):
        w+=1;
    if(~h%2):
        h+=1;

    ret = np.zeros((h, w), np.uint8);
    
    cx = int(w/2)
    cy = int(h/2)

    cv2.ellipse(ret, (cx, cy), (int(a/2), int(b/2)), -theta, 0, 360, 1, -1);
    return ret;

def steerableFilterALCM(im, a, b, theta):
    k = getEllipseKernel(a, b, theta)
    test = cv2.filter2D(im, cv2.CV_64F, k)
    cv2.normalize(test, test, 255, 0, cv2.NORM_MINMAX)
    test = np.uint8(test)
    return test;

def newWindow(name, w, h):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h);

'''
newWindow("test", 800, 600)

im = cv2.imread("./img/1191_a_r_IR.jpg")
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
t = steerableFilterALCM(gray, 100, 20, 0)
cv2.normalize(t, t, 255, 0, cv2.NORM_MINMAX)
t = np.uint8(t)
min = cv2.minMaxLoc(t)
print(min, max)
#tg = cv2.cvtColor(t,cv2.COLOR_BGR2GRAY)
print(t.shape, t.dtype)
ret, t2 = cv2.threshold(t, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow("test", t2)
cv2.waitKey(0)

'''
