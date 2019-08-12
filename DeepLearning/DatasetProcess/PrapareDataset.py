from lxml import etree
import glob
from tqdm import tqdm
import os.path as osp
import os
import shutil

train_annotations_path = r"F:\Download\dataset\auto\voc\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations"
train_path = r"F:\Download\dataset\auto\voc\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages"
test_path = r"F:\Download\dataset\auto\voc\VOC2012test\VOCdevkit\VOC2012\JPEGImages"
test_annotations_path = r"F:\Download\dataset\auto\voc\VOC2012test\VOCdevkit\VOC2012\Annotations"


class ParDataset:
    class_list = []     # 用来查询class_index使用
    def __init__(self, train_annotations_path, trainjpg_path, testjpg_path, test_annotations_path, file_type=".xml"):
        self.train_annotations_path = train_annotations_path
        self.file_type = file_type
        self.trainjpg_path = trainjpg_path
        self.testjpg_path = testjpg_path
        self.test_annotations_path = test_annotations_path

    # 获取所有文件名的list
    def creat_annotationslist(self):
        annotations_list = glob.glob(self.train_annotations_path + "/*" + self.file_type)
        return annotations_list

    # 得到所有jpg
    def crear_jpglist(self):
        jpglist = glob.glob(self.trainjpg_path + "/*" + ".jpg")
        return jpglist

    def crear_testannlist(self):
        testann_list = glob.glob(self.test_annotations_path + "/*" + ".jpg")
        return testann_list

    def crear_testjpglist(self):
        testjpglist = glob.glob(self.testjpg_path + "/*" + ".jpg")
        return testjpglist

    # 得到xml的根
    def get_xml_root(self, file_path):
        tree = etree.parse(file_path)
        root = tree.getroot()
        return root

    def get_middlename(self, file_path):
        dirname, basename = osp.split(file_path)
        middlename, filetype = osp.splitext(basename)
        return middlename

    def get_size(self, file_path):
        root = self.get_xml_root(file_path)
        size = root.xpath("size")[0]
        width = size.xpath("width")[0].text
        height = size.xpath("height")[0].text
        # print("width:{0}, height{1}".format(width, height))
        return width, height

    def get_object(self, file_path):
        namelist = []
        xminlist = []
        yminlist = []
        xmaxlist = []
        ymaxlist = []
        root = self.get_xml_root(file_path)
        for obj in root.xpath("object"):
            name = obj.xpath("name")[0].text
            bndbox = obj.xpath("bndbox")[0]
            xmin = bndbox.xpath("xmin")[0].text
            ymin = bndbox.xpath("ymin")[0].text
            xmax = bndbox.xpath("xmax")[0].text
            ymax = bndbox.xpath("ymax")[0].text
            # print("name: {0}, xmin:{1}, ymin:{2}, xmax:{3}, ymax:{4}".format(name, xmin, ymin, xmax, ymax))
            namelist.append(name)
            xminlist.append(xmin)
            yminlist.append(ymin)
            xmaxlist.append(xmax)
            ymaxlist.append(ymax)
        # print(namelist, xminlist, yminlist, xmaxlist, ymaxlist)
        return namelist, xminlist, yminlist, xmaxlist, ymaxlist

    def get_Coconame(self):
        namelist = []
        ann_list = self.creat_annotationslist()
        for file in tqdm(ann_list):
            root = self.get_xml_root(file)
            for objs in root.xpath("object"):
                for name in objs.xpath("name"):
                    if name.text not in namelist:
                        namelist.append(name.text)
        self.class_list = namelist
        CocoFile = open("./files/coco.name", "w")
        for name in namelist:
            CocoFile.writelines(name)
            CocoFile.write("\n")

    def get_traintxt(self):
        if osp.exists("./files/train.txt"):
            os.remove("./files/train.txt")
        image_index = 0
        class_index = 0
        jpgfile_list = self.crear_jpglist()
        for jpgfile in tqdm(jpgfile_list):
            jpg_path = jpgfile
            middlename = self.get_middlename(jpgfile)
            annotations_name = self.train_annotations_path + "\\" + middlename + ".xml"
            width, height = self.get_size(annotations_name)
            output = str(image_index) + " " + jpg_path + " " + width + " " + height + " "
            traintxt = open("./files/train.txt", "a")
            traintxt.writelines(output)
            namelist, xminlist, yminlist, xmaxlist, ymaxlist = self.get_object(annotations_name)
            length = len(namelist)
            for i in range(length):
                name = namelist[i]
                xmin = xminlist[i]
                ymin = yminlist[i]
                xmax = xmaxlist[i]
                ymax = ymaxlist[i]
                for j in range(len(self.class_list)):
                    if name == self.class_list[j]:
                        class_index = j
                        continue
                obj1 = str(class_index) + " " + xmin + " " + ymin + " " + xmax + " " + ymax + " "
                traintxt.writelines(obj1)
            traintxt.writelines("\r")
            traintxt.close()
            image_index += 1

    def get_valtxt(self):
        if osp.exists("./files/val.txt"):
            os.remove("./files/val.txt")
        image_index = 0
        class_index = 0
        jpgfile_list = self.crear_testjpglist()
        for jpgfile in tqdm(jpgfile_list):
            jpg_path = jpgfile
            middlename = self.get_middlename(jpgfile)
            annotations_name = self.test_annotations_path + "\\" + middlename + ".xml"
            width, height = self.get_size(annotations_name)
            output = str(image_index) + " " + jpg_path + " " + width + " " + height + " "
            valtxt = open("./files/val.txt", "a")
            valtxt.writelines(output)
            namelist, xminlist, yminlist, xmaxlist, ymaxlist = self.get_object(annotations_name)
            length = len(namelist)
            for i in range(length):
                name = namelist[i]
                xmin = xminlist[i]
                ymin = yminlist[i]
                xmax = xmaxlist[i]
                ymax = ymaxlist[i]
                for j in range(len(self.class_list)):
                    if name == self.class_list[j]:
                        class_index = j
                        continue
                obj1 = str(class_index) + " " + xmin + " " + ymin + " " + xmax + " " + ymax + " "
                valtxt.writelines(obj1)
            valtxt.writelines("\r")
            valtxt.close()
            image_index += 1

    def check_val_jpg(self):
        count = 0
        jpgfile_list = self.crear_testjpglist()
        for jpgfile in tqdm(jpgfile_list):
            midname = self.get_middlename(jpgfile)
            annotations_file = self.test_annotations_path + "\\" + midname + ".xml"
            if not osp.exists(annotations_file):
                tempdir = self.testjpg_path + "\\temp"
                if not osp.exists(tempdir):
                    os.mkdir(tempdir)
                new_file = tempdir + "\\" + midname + ".jpg"
                shutil.copyfile(jpgfile, new_file)
                os.remove(jpgfile)


if __name__ == '__main__':
    pd = ParDataset(train_annotations_path, train_path, test_path, test_annotations_path)
    pd.check_val_jpg()
    pd.get_Coconame()
    pd.get_traintxt()
    pd.get_valtxt()



