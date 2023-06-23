import torch 
import numpy as np
import cv2
from align_image.align_image import perspective_img
class loadYolo:
    def __init__(self, path_model, path_image):
        self.model = self.load_model(path_model)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.path_model = path_model
        self.path_image = path_image

    def load_model(self,path_model):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path = path_model)
        return model
    
    def score_image(self,image):
        self.model.to(self.device)
        image = np.transpose(image,(2, 0, 1))
        results = self.model(image)
        results = results.pandas().xyxy[0].values
        labels, cord = results[:,-1],results[:,:-1]
        return labels, cord
    
    def class_to_label(self, x):
        return self.classes[int(x)]
    
    def __call__(self):
        img = cv2.imread(self.path_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_orig = img
        labels, cord = self.score_image(img)
        tl, tr, bl,br = 0,0,0,0
        for i in range(cord.shape[0]):
            xmin, ymin, xmax, ymax = cord[i][0], cord[i][1],cord[i][2], cord[i][3]
            x_center = int((xmin + xmax)/2)
            y_center = int((ymin + ymax)/2)
            conf = cord[i][4]
            # cv2.circle(img_orig, (x_center, y_center),10, (0,255,0),2)
            name_class = int(cord[i][5])
            if conf > 0.5:
                if name_class == 0 and bl == 0:
                    bl = [x_center, y_center]
                if name_class == 3 and tr == 0:
                    tr = [x_center, y_center]
                if name_class == 2 and tl == 0:
                    tl = [x_center, y_center]
                if name_class == 1 and br == 0:
                    br = [x_center, y_center]    
        
        keypoints = np.zeros((4, 2), dtype = "float32")
        keypoints[0],keypoints[1], keypoints[2],keypoints[3] = tl, tr, br,bl

        wrap = perspective_img(img_orig, keypoints)
        wrap = cv2.cvtColor(wrap, cv2.COLOR_BGR2RGB)
        cv2.imwrite('output.png', wrap)

if __name__ == '__main__':
    model = loadYolo('last.pt', 'CMT_test.jpg')
    model()
