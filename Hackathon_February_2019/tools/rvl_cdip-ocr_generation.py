import tqdm
from lxml import etree
from multiprocessing import Pool
import os

base_path = "/home/akoelsch/Downloads/rvl-cdip/"


def target_function(line):
    path, _ = line.split()
    xml_file = base_path + "xml/iitcdip." + path[8] + "." + path[10] + ".xml"
    docid, filename = path.rsplit("/", 2)[1:]

    parser = etree.XMLParser(recover=True, huge_tree=True)
    tree = etree.parse(xml_file, parser=parser)

    record = tree.findall("./record/[docid='" + docid + "']")
    if len(record) == 1:
        record = record[0]
    else:
        print("record", record, path)

    ocr = record.findall("*ot")
    text = ""
    if len(ocr) == 1:
        text = ocr[0].text.split("pgNbr", 1)[0]
    elif len(ocr) == 0:
        pass
    else:
        print("ocr", ocr, path)

    ocr_dir = base_path + "ocr/" + os.path.dirname(path)
    os.makedirs(ocr_dir, exist_ok=True)
    out = open(os.path.join(ocr_dir, filename[:-4] + ".txt"), "w")
    out.write(text)
    out.close()


for partition in ["train", "val", "test"]:
    gt_file = open(base_path + "labels/" + partition + ".txt").readlines()
    pool = Pool(processes=24)
    for _ in tqdm.tqdm(pool.imap_unordered(target_function, gt_file), total=len(gt_file)):
        pass
