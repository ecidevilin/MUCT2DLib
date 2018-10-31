import dlib
from skimage import io
import os
rf = open('muct76-opencv.csv', "r")
wf = open("jpg/training_with_face_landmarks.xml", "w")

detector = dlib.get_frontal_face_detector()
lines = rf.readlines()
wf.writelines("<?xml version='1.0' encoding='ISO-8859-1'?>\n<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>\n<dataset>\n<name>Training faces</name>\n<comment>http://www.milbo.org/muct</comment>\n<images>\n")
toTrain = [31, 36, 37, 45, 67]

for i in range(1, len(lines)):
    mirror = lines[i].startswith("ir")
    cols = lines[i].split(",")
    file = cols[0]
    wf.write("  <image file='{name}.jpg'>\n".format(name=file))
    if mirror:
        if not os.path.exists("jpg/{name}.jpg".format(name=file)):
            img = io.imread("jpg/{name}.jpg".format(name='i'+file[2:]))
            img = img[:, ::-1]
            img = io.imsave("jpg/{name}.jpg".format(name=file),img)
            img = io.imread("jpg/{name}.jpg".format(name=file))
        else:
            img = io.imread("jpg/{name}.jpg".format(name=file))
    else:
        img = io.imread("jpg/{name}.jpg".format(name=file))
    xmax = 0
    ymax = 0
    xmin = 65535
    ymin = 65535
    for j in range(1, int(len(cols) / 2)):
        numx = int(round(float(cols[j * 2])))
        numy = int(round(float(cols[j * 2 + 1])))
        xmax = max(xmax, numx)
        xmin = min(xmin, numx)
        ymax = max(ymax, numy)
        ymin = min(ymin, numy)
    dets = detector(img, 1)
    if (len(dets) == 0):
        print(cols[0])
    else:
        ymin = min(ymin, dets[0].top())
        xmin = min(xmin, dets[0].left())
        ymax = max(ymax, dets[0].bottom())
        xmax = max(xmax, dets[0].right())
    wf.write("    <box top='{top}' left='{left}' width='{width}' height='{height}'>\n".format(top=ymin,left=xmin,width=xmax-xmin+1,height=ymax-ymin+1))
    for j in range(0, len(toTrain)):
        numx = int(round(float(cols[toTrain[j] * 2 + 2])))
        numy = int(round(float(cols[toTrain[j] * 2 + 3])))
        wf.write("      <part name='{name:0>2d}' x='{x}' y='{y}'/>\n".format(name=j, x=numx, y=numy))
    # for j in range(1, int(len(cols) / 2)):
    #     numx = int(round(float(cols[j * 2])))
    #     numy = int(round(float(cols[j * 2 + 1])))
    #     wf.write("      <part name='{name:0>2d}' x='{x}' y='{y}'/>\n".format(name=j-1,x=numx,y=numy))
    wf.write("    </box>\n")
    wf.write("  </image>\n")
wf.writelines("</images>\n</dataset>\n")
wf.flush()
rf.close()
wf.close()

print("Done")
os.system("python train_shape_predictor.py jpg")
