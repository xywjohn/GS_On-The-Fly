import os
import shutil
from tqdm import tqdm
from PIL import Image
import glob
import os

def copy_images(src_path, dst_path):
    shutil.copy2(src_path, dst_path)

def resize_image_pil(image_path, new_width=1600):
    img = Image.open(image_path)
    if float(img.width) > 1600:
        w_percent = new_width / float(img.width)
        new_height = int((float(img.height) * w_percent))  # 计算等比例高度
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)  # 高质量降采样
        return img_resized, True
    else:
        return img, False

DatasetName = ['WHU']
SourceImageDataset = r"/SourceImageDataset"
On_The_Fly_Dataset = r"/On_The_Fly_Dataset"

for dataset in DatasetName:
    SourceImagesDir = os.path.join(SourceImageDataset, r"{}/images".format(dataset))
    jpg_files = glob.glob(os.path.join(SourceImagesDir, "*.JPG"))
    endimage = jpg_files[0].split(".")[-1]
    progress_bar = tqdm(range(0, len(jpg_files)), desc="Resize progress {}".format(dataset))
    for jpg in jpg_files:
        img, IsResize = resize_image_pil(jpg)
        if IsResize:
            os.makedirs(SourceImagesDir + "_Resize", exist_ok=True)
            img.save(os.path.join(SourceImagesDir + "_Resize", jpg.split("\\")[-1]))
        progress_bar.update(1)
    progress_bar.close()

    MainDir = os.path.join(On_The_Fly_Dataset, dataset)

    if IsResize:
        OriginSourceImagesDir = SourceImagesDir
        SourceImagesDir = SourceImagesDir + "_Resize"

    print(MainDir)
    subfolders = [f for f in os.listdir(MainDir) if os.path.isdir(os.path.join(MainDir, f))]
    progress_bar = tqdm(range(0, len(subfolders)), desc="Copy progress {}".format(dataset))

    for i in range(len(subfolders)):
        if os.path.exists(os.path.join(MainDir, r"{}/sparse/0/imagesNames.txt".format(subfolders[i]))):
            ImagesNamesTXT = open(MainDir + r"/{}/sparse/0/imagesNames.txt".format(subfolders[i]))
            ImagesNamesList = ImagesNamesTXT.readline().split(",")
            os.makedirs(MainDir + r"/{}/images".format(subfolders[i]), exist_ok=True)
            for image in ImagesNamesList:
                image = image.split("\n")[0]
                copy_images(SourceImagesDir + "/" + image + ".{}".format(endimage), MainDir + r"/{}/images/".format(subfolders[i]) + image + ".{}".format(endimage))
        progress_bar.update(1)
    progress_bar.close()