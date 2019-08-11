from lxml import etree
import glob
from tqdm import tqdm

path = r"F:\Download\dataset\auto\voc\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations"

class ClassStatistics:

    # 初始化输入文件地址和文件类型
    def __init__(self, annotations_path, file_type):
        self.annotations_path = annotations_path
        self.file_type = file_type

    # 获取所有文件名的list
    def creat_annotationslist(self):
        annotations_list = glob.glob(self.annotations_path + "/*" + self.file_type)
        return annotations_list

    # 得到xml的根
    def get_xml_root(self, file_path):
        tree = etree.parse(file_path)
        root = tree.getroot()
        return root

    # 先遍历所有标注文件，获取所有的类别list
    def get_class_list(self):
        namelist = []
        ann_list = self.creat_annotationslist()
        for file in tqdm(ann_list):
            root = self.get_xml_root(file)
            for objs in root.xpath("object"):
                for name in objs.xpath("name"):
                    if name.text not in namelist:
                        namelist.append(name.text)
        return namelist

    # 进行分类数量统计
    def class_statistics(self):
        class_list = self.get_class_list()      # 所有类别的list
        numberlist = [0] * len(class_list)   # 根class_list对应的计数用的list 初始化创建全零数组
        ann_list = self.creat_annotationslist()
        for file in tqdm(ann_list):
            root = self.get_xml_root(file)
            for objs in root.xpath("object"):           # 从根里获取object的子节点
                for name in objs.xpath("name"):             # 从object里获取name的子节点
                    for i in range(len(class_list)):            # 如果name.text和classlist中的第i个名称一样，就加入计数list的第i个元素中
                        if name.text == class_list[i]:
                           numberlist[i] += 1
        print(numberlist)
        for i in range(len(class_list)):    # 输出类别对应数量
            print(class_list[i], " : ", numberlist[i])

if __name__ == '__main__':
    cs1 = ClassStatistics(path, ".xml")
    cs1.class_statistics()


