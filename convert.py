
rf = open('muct76-opencv.csv', "r")
wf = open("training_with_face_landmarks.xml", "w")

lines = rf.readlines()
wf.writelines("<?xml version='1.0' encoding='ISO-8859-1'?>\n<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>\n<dataset>\n<name>Training faces</name>\n<comment>http://www.milbo.org/muct</comment>\n<images>\n")
for i in range(1, len(lines)):
    if (lines[i].startswith("ir")):
        continue
    cols = lines[i].split(",")
    wf.write("  <image file='{name}.jpg'>\n".format(name=cols[0]))
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
    wf.write("    <box top='{top}' left='{left}' width='{width}' height='{height}'>\n".format(top=ymin,left=xmin,width=xmax-xmin+1,height=ymax-ymin+1))
    for j in range(1, int(len(cols) / 2)):
        numx = int(round(float(cols[j * 2])))
        numy = int(round(float(cols[j * 2 + 1])))
        wf.write("      <part name='{name:0>2d}' x='{x}' y='{y}'/>\n".format(name=j-1,x=numx,y=numy))
    wf.write("    </box>\n")
    wf.write("  </image>\n")
wf.writelines("</images>\n</dataset>\n")
wf.flush()
rf.close()
wf.close()

print("Done")
