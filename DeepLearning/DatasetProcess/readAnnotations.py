from lxml import etree
import glob
import os.path as osp
from tqdm import tqdm

file = r"F:\Download\dataset\auto\voc\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations"
class ReadAnnotations:
    def __init__(self, annotations_path, file_type=".xml"):
        # annotations_path 标注文件地址
        # file_type 标注文件类型 默认.xml
        self.file_paths = annotations_path
        self.filetype = file_type

    # 遍历文件夹，创建文件名list
    def create_filelist(self):
        file_list = glob.glob(self.file_paths +"/*" + self.filetype)
        return file_list

    # 获取文件名 输入：file_path 完整地址(包含驱动器名那种)
    def get_middlename(self, file_path):
        dirname, basename = osp.split(file_path)
        middlename, filetype = osp.splitext(basename)
        return middlename

    # 得到xml的根
    def get_xml_root(self, file_path):
        tree = etree.parse(file_path)
        root = tree.getroot()
        return root

    # 得到文件相应的大小 返回值 width, height
    def get_size(self, file_path):
        root = self.get_xml_root(file_path)
        size = root.xpath("size")[0]
        width = size.xpath("width")[0].text
        height = size.xpath("height")[0].text
        print("width:{0}, height{1}".format(width, height))
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

    def output_object(self):
        files_list = self.create_filelist()
        for file in tqdm(files_list):
            self.get_object(file)

    def output_size(self):
        files_list = self.create_filelist()
        for file in tqdm(files_list):
            self.get_size(file)

if __name__ == '__main__':
    ra = ReadAnnotations(file)
    ra.output_object()


