import os
import matplotlib.pyplot as plt
import random
import torch
import numpy as np
from random import randint

from torch import no_grad

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, Scene_Second, GaussianModel
from fused_ssim import fused_ssim
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from plyfile import PlyData, PlyElement
from PIL import Image
import torchvision.transforms as transforms
from scipy.spatial import cKDTree
import math
import time
import cv2
import subprocess
import lpips
from skimage.metrics import structural_similarity as ssim2
import copy
from scene.cameras import Camera
import open3d as o3d

def Get_lpips(rgbs: torch.Tensor, target_rgbs: torch.Tensor):
    gt = target_rgbs.unsqueeze(0) * 2 - 1
    pred = rgbs.unsqueeze(0) * 2 - 1

    lpips_vgg = lpips.LPIPS(net='vgg').eval().to(rgbs.device)
    lpips_vgg_i = lpips_vgg(gt, pred, normalize=True)

    lpips_alex = lpips.LPIPS(net='alex').eval().to(rgbs.device)
    lpips_alex_i = lpips_alex(gt, pred, normalize=True)

    lpips_squeeze = lpips.LPIPS(net='squeeze').eval().to(rgbs.device)
    lpips_squeeze_i = lpips_squeeze(gt, pred, normalize=True)

    return {'vgg': lpips_vgg_i.item(), 'alex': lpips_alex_i.item(), 'squeeze': lpips_squeeze_i.item()}

def get_gpu_memory_usage(gpu=0):
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader', f'--id={gpu}'], stdout=subprocess.PIPE)
    memory_used = int(result.stdout.decode('utf-8').strip())
    return memory_used

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def RenderTargetCam(TargetCam, gaussians, pipe, bg, iteration, ImageOutputDir, UseRenderTarget = False):
    if UseRenderTarget:
        render_pkg = render(TargetCam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]
        ImageOutput = (image - image.min()) / (image.max() - image.min())
        ImageOutput = ImageOutput.permute(1, 2, 0).cpu().numpy()
        ImageOutput = (ImageOutput * 255).astype(np.uint8)
        ImageOutput = Image.fromarray(ImageOutput)

        ImageOutput.save(os.path.join(ImageOutputDir, f"PredictionImages{iteration}.jpg"))

# 基于深度的损失
def pearson_depth_loss(depth_src, depth_target):
    #co = pearson(depth_src.reshape(-1), depth_target.reshape(-1))

    src = depth_src - depth_src.mean()
    target = depth_target - depth_target.mean()

    src = src / (src.std() + 1e-6)
    target = target / (target.std() + 1e-6)

    co = (src * target).mean()
    assert not torch.any(torch.isnan(co))
    return 1 - co

def local_pearson_loss(depth_src, depth_target, box_p, p_corr):
        # Randomly select patch, top left corner of the patch (x_0,y_0) has to be 0 <= x_0 <= max_h, 0 <= y_0 <= max_w
        num_box_h = math.floor(depth_src.shape[0]/box_p)
        num_box_w = math.floor(depth_src.shape[1]/box_p)
        max_h = depth_src.shape[0] - box_p
        max_w = depth_src.shape[1] - box_p
        _loss = torch.tensor(0.0,device='cuda')
        n_corr = int(p_corr * num_box_h * num_box_w)
        x_0 = torch.randint(0, max_h, size=(n_corr,), device = 'cuda')
        y_0 = torch.randint(0, max_w, size=(n_corr,), device = 'cuda')
        x_1 = x_0 + box_p
        y_1 = y_0 + box_p
        _loss = torch.tensor(0.0,device='cuda')
        for i in range(len(x_0)):
            _loss += pearson_depth_loss(depth_src[x_0[i]:x_1[i],y_0[i]:y_1[i]].reshape(-1), depth_target[x_0[i]:x_1[i],y_0[i]:y_1[i]].reshape(-1))
        return _loss/n_corr

def Soft_Hard_loss(SoftDepth, HardDepth):
    delta_Depth = SoftDepth - HardDepth
    norm = torch.sqrt(torch.sum(delta_Depth ** 2))

    return norm

# 训练前的准备工作
def prepare_output_and_logger(args):
    # 如果args.model_path为空
    if not args.model_path:
        # 接下来的部分的意思是，如果不存在一开始认为指定的模型输出路径，则会在默认的输出文件夹output中随机生成一个以一串随机字符组成的名字的文件夹，
        # 并将该文件夹作为模型输出文件夹
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # 创建对应的输出文件夹
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)

    # 在文件夹中创建并打开一个二进制文件cfg_args，并在里面输出参数配置其的所有内容
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 创建一个Tensorboard writer对象
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def prepare_output_and_logger_second(args):
    # 如果args.model_path为空
    if not args.model_path_second:
        # 接下来的部分的意思是，如果不存在一开始认为指定的模型输出路径，则会在默认的输出文件夹output中随机生成一个以一串随机字符组成的名字的文件夹，
        # 并将该文件夹作为模型输出文件夹
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path_second = os.path.join("./output/", unique_str[0:10])

    # 创建对应的输出文件夹
    print("Output folder: {}".format(args.model_path_second))
    os.makedirs(args.model_path_second, exist_ok=True)

    # 在文件夹中创建并打开一个二进制文件cfg_args，并在里面输出参数配置其的所有内容
    with open(os.path.join(args.model_path_second, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 创建一个Tensorboard writer对象
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path_second)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

# 输出训练日志
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

# 打印模型的评价结果
def EvaluateModel(iteration, l1_loss, scene, renderFunc, renderArgs, Diary, IsFinal=False, EvaluateDiary=None):
    # Report test and samples of training set
    torch.cuda.empty_cache()
    validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()}, {'name': 'train', 'cameras': scene.getTrainCameras()})
                            # {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}

    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpips_test = 0.0

            PSNRs = []
            for idx, viewpoint in enumerate(config['cameras']):
                image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                l1_test += l1_loss(image, gt_image).mean().double()

                PSNR = psnr(image, gt_image).mean().double()
                psnr_test += PSNR
                PSNRs.append(PSNR)

                if IsFinal and False:
                    lpips_test += Get_lpips(image, gt_image)['vgg']

                    # 自动选择合适的 win_size，避免小图像报错
                    np_image = image.detach().cpu().numpy().astype(np.float32)
                    np_gt_image = gt_image.detach().cpu().numpy().astype(np.float32)

                    min_dim = min(np_image.shape[:2])
                    win_size = min(7, min_dim) if min_dim >= 7 else min_dim

                    ssim_test += ssim2(np_gt_image, np_image, channel_axis=-1, data_range=1, win_size=win_size)

            psnr_test /= len(config['cameras'])
            ssim_test /= len(config['cameras'])
            lpips_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])
            print("\n[ITER {}] Evaluating {}: L1 {} Gaussians {} PSNR {}".format(iteration, config['name'], l1_test, scene.gaussians.get_xyz.shape[0], psnr_test))
            Diary.write("[ITER {}] Evaluating {}: L1 {} Gaussians {} PSNR {}\n".format(iteration, config['name'], l1_test, scene.gaussians.get_xyz.shape[0], psnr_test))
            if IsFinal and False:
                Diary.write("[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}\n".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))

            if EvaluateDiary is not None and len(PSNRs) != 0:
                EvaluateDiary.write("-----------------------------------------------------------\n")
                EvaluateDiary.write(f"Iterations: {iteration}\n")
                for i in range(len(PSNRs)):
                    EvaluateDiary.write(f"{config['cameras'][i].image_name}: {PSNRs[i]}\n")

# 对所有训练影像都进行一次预测并保存结果
def RenderAllImagesAndEvaluate(args, ALL_viewpoint_stack, gaussians, pipe, bg, opt, AlreadyIterations, FirstTrainingIterations, first_iter, iteration, NewlyAddedImages=[]):
    _, ALL_Images_Loss, ALL_Image_PSNR, All_PredictedImages, ALL_Image_Visibility_Filter, ALL_Depth_Map, All_Depth_Map_Hard = GetImageWeightsFromDynamicMonitoring(ALL_viewpoint_stack, gaussians, pipe, bg, opt)
    progress_bar = tqdm(len(ALL_viewpoint_stack), desc="Rendering progress")
    for i in range(len(ALL_viewpoint_stack)):
        ImageOutputDir = os.path.join(args.Model_Path_Dir, "OutputImages")
        ImageOutputDir = os.path.join(ImageOutputDir, ALL_viewpoint_stack[i].image_name)
        PSNR_File_Path = os.path.join(ImageOutputDir, "PSNR.txt")

        os.makedirs(ImageOutputDir, exist_ok=True)
        if (AlreadyIterations == FirstTrainingIterations) or (ALL_viewpoint_stack[i].image_name in NewlyAddedImages):
            PSNR_File = open(PSNR_File_Path, 'w')
        else:
            PSNR_File = open(PSNR_File_Path, 'a')

        VF = ALL_Image_Visibility_Filter[i]
        Visible_Gaussian_Num = VF.shape[0]
        PSNR_File.write(str(iteration - first_iter + AlreadyIterations + 1) + f": {ALL_Image_PSNR[i]}, Visible_Gaussians_Num: {Visible_Gaussian_Num}\n")
        PSNR_File.close()

        image = All_PredictedImages[i]
        ImageOutput = (image - image.min()) / (image.max() - image.min())
        ImageOutput = ImageOutput.permute(1, 2, 0).cpu().numpy()
        ImageOutput = (ImageOutput * 255).astype(np.uint8)
        ImageOutput = Image.fromarray(ImageOutput)

        ImageOutput.save(os.path.join(ImageOutputDir, f"PredictionImages{iteration - first_iter + AlreadyIterations + 1}.jpg"))

        Depth_Map = ALL_Depth_Map[i]

        ImageOutput = (Depth_Map - 0) / (Depth_Map.max() - 0)
        ImageOutput = ImageOutput.cpu().numpy()
        ImageOutput = (ImageOutput * 255).astype(np.uint8)
        ImageOutput = Image.fromarray(ImageOutput, mode="L")

        ImageOutput.save(os.path.join(ImageOutputDir, f"Depth_Map{iteration - first_iter + AlreadyIterations + 1}.jpg"))

        Depth_Map_Hard = All_Depth_Map_Hard[i]

        ImageOutput = (Depth_Map_Hard - 0) / (Depth_Map_Hard.max() - 0)
        ImageOutput = ImageOutput.cpu().numpy()
        ImageOutput = (ImageOutput * 255).astype(np.uint8)
        ImageOutput = Image.fromarray(ImageOutput, mode="L")

        ImageOutput.save(os.path.join(ImageOutputDir, f"Depth_Map_Hard{iteration - first_iter + AlreadyIterations + 1}.jpg"))

        Deta_Depth_Map = (Depth_Map_Hard - Depth_Map) / Depth_Map
        Deta_Depth_Map = torch.where(Deta_Depth_Map < 0, -Deta_Depth_Map, Deta_Depth_Map)

        ImageOutput = (Deta_Depth_Map - 0) / (Deta_Depth_Map.max() - 0)
        ImageOutput = ImageOutput.cpu().numpy()
        ImageOutput = (ImageOutput * 255).astype(np.uint8)
        ImageOutput = Image.fromarray(ImageOutput, mode="L")

        ImageOutput.save(os.path.join(ImageOutputDir, f"Deta_Depth_Map{iteration - first_iter + AlreadyIterations + 1}.jpg"))

        result_np = Deta_Depth_Map.cpu().numpy().flatten()

        # 绘制直方图
        plt.hist(result_np, bins=40, color='blue', edgecolor='black', alpha=0.7)
        plt.title('Deta_Depth_Map')
        plt.xlabel('Deta_Depth')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 保存图像到文件
        plt.savefig(os.path.join(ImageOutputDir, f"Histogram_Deta_Depth_Map{iteration - first_iter + AlreadyIterations + 1}.jpg"), dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形，释放内存

        progress_bar.update(1)

# 对于影像的中间渲染结果，对其PSNR进行统计并绘制成折线图
def DrawImagesPSNR(OutputImagesDir, FirstSceneIterations, MergeSceneIterations):
    ImagesNames = [name for name in os.listdir(OutputImagesDir) if os.path.isdir(os.path.join(OutputImagesDir, name)) and name != "Final"]
    X = []
    PSNRs = []
    GaussiansNums = []
    for name in ImagesNames:
        PSNRFile = open(os.path.join(OutputImagesDir, name, "PSNR.txt"))

        x = []
        PSNR = []
        GaussiansNum = []
        tempstr = PSNRFile.readline()
        while(tempstr != ""):
            this_x = int((int(tempstr.split(":")[0]) - FirstSceneIterations) / MergeSceneIterations)
            this_PSNR = float(tempstr.split(" ")[1].split(",")[0])
            this_GaussiansNum = int(tempstr.split(" ")[-1])

            if this_x not in x:
                x.append(this_x)
                GaussiansNum.append(this_GaussiansNum)
                PSNR.append(this_PSNR)
            else:
                GaussiansNum[x.index(this_x)] = this_GaussiansNum
                PSNR[x.index(this_x)] = this_PSNR
            tempstr = PSNRFile.readline()

        X.append(x)
        PSNRs.append(PSNR)
        GaussiansNums.append(GaussiansNum)

        # 绘制当前影像的PSNR变化
        plt.figure()
        random_color = (random.random(), random.random(), random.random())
        plt.plot(x, PSNR, color=random_color, linestyle="-")
        plt.xlabel("X")
        plt.ylabel("PSNR")
        plt.title(f"{name} PSNR")

        plt.savefig(os.path.join(OutputImagesDir, name, f"{name}_PSNR.jpg"))

        plt.close()

        # 绘制当前影像的GaussiansNum变化
        plt.figure()
        random_color = (random.random(), random.random(), random.random())
        plt.plot(x, GaussiansNum, color=random_color, linestyle="-")
        plt.xlabel("X")
        plt.ylabel("GaussiansNum")
        plt.title(f"{name} GaussiansNum")

        plt.savefig(os.path.join(OutputImagesDir, name, f"{name}_GaussiansNum.jpg"))

        plt.close()

    # 绘制多条折线
    plt.figure()
    for i in range(0, len(ImagesNames), 1):
        random_color = (random.random(), random.random(), random.random())
        plt.plot(X[i], PSNRs[i], color=random_color, linestyle="-")


    # 添加图例、标题和标签
    plt.xlabel("X")
    plt.ylabel("PSNR")
    plt.title("Images PSNR")

    plt.savefig(os.path.join(OutputImagesDir, "ImagesPSNRs.jpg"))

    plt.close()

    # 绘制多条折线
    plt.figure()
    for i in range(0, len(ImagesNames), 1):
        random_color = (random.random(), random.random(), random.random())
        plt.plot(X[i], GaussiansNums[i], color=random_color, linestyle="-")

    # 添加图例、标题和标签
    plt.xlabel("X")
    plt.ylabel("GaussiansNum")
    plt.title("Images GaussiansNum")

    plt.savefig(os.path.join(OutputImagesDir, "ImagesGaussiansNum.jpg"))

    plt.close()

# 对模型的PSNR进行统计并绘制成折线图
def DrawModelPSNR(DiaryOutputDir, FirstSceneIterations):
    SimpleDiaryPath = os.path.join(DiaryOutputDir, "SimpleDiary.txt")
    SimpleDiary = open(SimpleDiaryPath)

    for i in range(9):
        SimpleDiary.readline()

    its = []
    PSNRs = []
    Gaussians = []
    for i in range(int(FirstSceneIterations / 500)):
        tempstr = SimpleDiary.readline()
        its.append(int(tempstr.split(" ")[1].split("]")[0]))
        PSNRs.append(float(tempstr.split(" ")[-1].split('\n')[0]))
        Gaussians.append(int(tempstr.split(" ")[7]))

    for i in range(7):
        SimpleDiary.readline()

    tempstr = SimpleDiary.readline()
    while tempstr != "":
        if tempstr.startswith("[ITER") and tempstr.split(" ")[-1] != "Gaussians\n":
            its.append(int(tempstr.split(" ")[1].split("]")[0]))
            PSNRs.append(float(tempstr.split(" ")[-1].split('\n')[0]))
            Gaussians.append(int(tempstr.split(" ")[7]))
            tempstr = SimpleDiary.readline()
        else:
            tempstr = SimpleDiary.readline()

    plt.plot(its, PSNRs, color="red", linestyle="-")

    plt.xlabel("Iterations")
    plt.ylabel("PSNR")
    plt.title("Model PSNR")

    plt.savefig(os.path.join(DiaryOutputDir, "ModelPSNR.jpg"))

    plt.close()

    plt.plot(its, Gaussians, color="blue", linestyle="-")

    plt.xlabel("Iterations")
    plt.ylabel("Gaussians")
    plt.title("Model Gaussians")

    plt.savefig(os.path.join(DiaryOutputDir, "ModelGaussians.jpg"))

    plt.close()

# 对于新加入的影像，在训练完成后对该影像进行一次输出
def RenderNewlyAddedImages(ALL_viewpoint_stack, gaussians, dataset, opt, pipe, OutputImagesDir, ProgressiveTrainingTime, State):
    # 初始化背景颜色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 设置背景颜色，一般是白色
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    viewpoint_cam_copy = ALL_viewpoint_stack[-1]
    render_pkg_copy = render(viewpoint_cam_copy, gaussians, pipe, bg)
    image_copy, viewspace_point_tensor_copy, visibility_filter_copy, radii_copy = render_pkg_copy["render"], \
                                                                                  render_pkg_copy["viewspace_points"], \
                                                                                  render_pkg_copy["visibility_filter"], \
                                                                                  render_pkg_copy["radii"]
    image = image_copy
    ImageOutput = (image - image.min()) / (image.max() - image.min())
    ImageOutput = ImageOutput.permute(1, 2, 0).cpu().numpy()
    ImageOutput = (ImageOutput * 255).astype(np.uint8)
    ImageOutput = Image.fromarray(ImageOutput)

    os.makedirs(OutputImagesDir, exist_ok=True)

    ImageOutput.save(os.path.join(OutputImagesDir, f"PredictionImages_{ProgressiveTrainingTime}_{State}.jpg"))

# 生成Demo
def GetTrainingDemo(image_folder, output_video):
    # 读取文件夹中的影像文件，并按文件名排序
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    imageIndex = [(int(ImageName.split("_")[1]), int(ImageName.split("_")[2].split(".")[0])) for ImageName in images]
    SortedImages = [" " for i in range(len(images))]
    for i in range(int(len(images))):
        SortedImages[imageIndex[i][0] * 3 + imageIndex[i][1] - 1] = images[i]
    images = SortedImages

    # 获取影像的尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # 定义视频编码器和输出格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式，例如'XVID'或'mp4v'
    fps = 5  # 每秒帧数
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 将每帧添加到视频中
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # 释放资源
    video.release()
    cv2.destroyAllWindows()

    print(f"训练视频已保存到 {output_video}")

def GetResultDemo(OutputImagesDir, FirstScene, args):
    # 初始化背景颜色
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 设置背景颜色，一般是白色
    bg = torch.rand((3), device="cuda") if args.random_background else background

    viewpoint_stack = FirstScene.getTrainCameras().copy()

    os.makedirs(OutputImagesDir, exist_ok=True)

    video = None

    for viewpoint_cam in viewpoint_stack:
        render_pkg_copy = render(viewpoint_cam, FirstScene.gaussians, args, bg)
        image_copy, viewspace_point_tensor_copy, visibility_filter_copy, radii_copy = render_pkg_copy["render"], \
            render_pkg_copy["viewspace_points"], \
            render_pkg_copy["visibility_filter"], \
            render_pkg_copy["radii"]
        image = image_copy
        ImageOutput = (image - image.min()) / (image.max() - image.min())
        ImageOutput = ImageOutput.permute(1, 2, 0).cpu().numpy()
        ImageOutput = (ImageOutput * 255).astype(np.uint8)
        ImageOutput = Image.fromarray(ImageOutput)

        ImageOutput.save(os.path.join(OutputImagesDir, f"{viewpoint_cam.image_name}.jpg"))

        if video == None:
            # 获取影像的尺寸
            first_image_path = os.path.join(OutputImagesDir, f"{viewpoint_cam.image_name}.jpg")
            frame = cv2.imread(first_image_path)
            height, width, layers = frame.shape

            # 定义视频编码器和输出格式
            output_video = os.path.join(OutputImagesDir, "Demo.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式，例如'XVID'或'mp4v'
            fps = 5  # 每秒帧数
            video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

            video.write(frame)
        else:
            img_path = os.path.join(OutputImagesDir, f"{viewpoint_cam.image_name}.jpg")
            frame = cv2.imread(img_path)
            video.write(frame)

    # 释放资源
    video.release()
    cv2.destroyAllWindows()

    print(f"最终结果视频已保存到 {output_video}")

# 获取较为简洁的输出日志
def GetSimpleDiary(DiaryOutputDir):
    DiaryPath = os.path.join(DiaryOutputDir, "Diary.txt")
    SimpleDiaryPath = os.path.join(DiaryOutputDir, "SimpleDiary.txt")

    Diary = open(DiaryPath)
    SimpleDiary = open(SimpleDiaryPath, "w")

    SimpleDiary.write(Diary.readline())
    SimpleDiary.write(Diary.readline())
    SimpleDiary.write(Diary.readline())

    tempstr = Diary.readline()

    while not tempstr.startswith("***"):
        SimpleDiary.write(tempstr)
        tempstr = Diary.readline()

    while tempstr != "":
        if tempstr.startswith("***"):
            SimpleDiary.write('\n' + tempstr)
            tempstr = Diary.readline()
        elif tempstr.startswith(("New scene points Num", "Second Scene Initialization Time Cost",
                                 "Newly Added Images Name", "Merge Scene Time Cost", "[ITER",
                                 "Merge Scene Training Time Cost",
                                 "Global Optimization Time Cost", "---")):
            SimpleDiary.write(tempstr)
            tempstr = Diary.readline()
        elif tempstr.startswith("Training complete."):
            SimpleDiary.write(tempstr.split('\n')[0])
            break
        else:
            tempstr = Diary.readline()

    SimpleDiary.close()

# 每一次更新影像堆栈之前，输出每一张影像被选中的次数，并标记出新加入的影像
def GetImageChosenTimesByWeights(Images_Weights, NewlyAddedImages, ALL_viewpoint_stack, MergeTrainingIterations, Diary):
    ImageChosenTimes = []
    IsNew = []
    viewpoint_stack = []
    Sum_Weight = sum(Images_Weights)
    NewCamera = None
    WorstCamera = None
    WorstCameraIndex = -1

    for i in range(len(ALL_viewpoint_stack)):
        # 判断是否是新加入的影像
        if (ALL_viewpoint_stack[i].image_name in NewlyAddedImages):
            IsNew.append(True)
            NewCamera = ALL_viewpoint_stack[i]
        else:
            IsNew.append(False)

        # 计算需要训练的次数
        vs_times = round(MergeTrainingIterations * Images_Weights[i] / Sum_Weight)
        ImageChosenTimes.append(vs_times)

        if vs_times == max(ImageChosenTimes):
            WorstCamera = ALL_viewpoint_stack[i]
            WorstCameraIndex = i

        # 将多个当前影像对应的Camera类加入到堆栈中
        for j in range(vs_times):
            viewpoint_stack.append(ALL_viewpoint_stack[i])

    if len(NewlyAddedImages) == 0:
        NewCamera = WorstCamera

    # 如果viewpoint_stack中的Camera数量没有达到MergeTrainingIterations，则剩下的全部补充为新加入的影像
    RepeatAddNew = 0
    while len(viewpoint_stack) < MergeTrainingIterations:
        viewpoint_stack.append(NewCamera)
        RepeatAddNew = RepeatAddNew + 1

    if len(NewlyAddedImages) == 0:
        ImageChosenTimes[WorstCameraIndex] = ImageChosenTimes[WorstCameraIndex] + RepeatAddNew

    # 打印结果
    for i in range(len(IsNew)):
        if IsNew[i]:
            Diary.write(f"{ALL_viewpoint_stack[i].image_name} => New, Weight={Images_Weights[i]}, ChosenTimes={ImageChosenTimes[i] + RepeatAddNew}, Repeat={RepeatAddNew}\n")
        else:
            Diary.write(f"{ALL_viewpoint_stack[i].image_name} => Weight={Images_Weights[i]}, ChosenTimes={ImageChosenTimes[i]}\n")
    '''
    for i in range(len(IsNew)):
        if IsNew[i]:
            print(ALL_viewpoint_stack[i].image_name, "=>", "New", f"Weight={Images_Weights[i]}", f"ChosenTimes={ImageChosenTimes[i] + RepeatAddNew}", f"Repeat={RepeatAddNew}")
        else:
            print(ALL_viewpoint_stack[i].image_name, "=>", f"Weight={Images_Weights[i]}", f"ChosenTimes={ImageChosenTimes[i]}")
    '''

    # 返回训练影像的选取结果
    return viewpoint_stack

def OnlyTrainMaxWeightsImages(Images_Weights, NewlyAddedImages, ALL_viewpoint_stack, MergeTrainingIterations, Diary):
    if len(Images_Weights) <= 20:
        return ALL_viewpoint_stack.copy()
    else:
        # 找到16张权值最大的影像
        MaxWeights = []
        MaxWeightImages = []
        MaxWeightIndexes = []
        for i in range(len(Images_Weights)):
            if len(MaxWeightImages) < 16:
                MaxWeights.append(Images_Weights[i])
                MaxWeightImages.append(ALL_viewpoint_stack[i])
                MaxWeightIndexes.append(i)
            else:
                if min(MaxWeights) < Images_Weights[i]:
                    cidx = MaxWeights.index(min(MaxWeights))
                    MaxWeights[cidx] = Images_Weights[i]
                    MaxWeightImages[cidx] = ALL_viewpoint_stack[i]
                    MaxWeightIndexes[cidx] = i

        # 从剩余的影像中随机挑出4张
        Allindex = [i for i in range(len(Images_Weights))]
        diff_index = [i for i in Allindex if i not in MaxWeightIndexes]
        SampleIndex = random.sample(diff_index, 4)
        for i in SampleIndex:
            MaxWeights.append(Images_Weights[i])
            MaxWeightImages.append(ALL_viewpoint_stack[i])
            MaxWeightIndexes.append(i)

        viewpoint_stack = []
        for i in range(len(MaxWeights)):
            for _ in range(5):
                viewpoint_stack.append(MaxWeightImages[i])

        # 打印结果
        for i in range(len(ALL_viewpoint_stack)):
            if i in MaxWeightIndexes:
                Diary.write(f"{ALL_viewpoint_stack[i].image_name} => Weight={Images_Weights[i]}, ChosenTimes={5}\n")
            else:
                Diary.write(f"{ALL_viewpoint_stack[i].image_name} => Weight={Images_Weights[i]}, ChosenTimes={0}\n")

        return viewpoint_stack

# 读入影像匹配关系矩阵
def ReadInImageMatchMatrix(MatrixPath, ImagesNamePath, scene):
    # 获取现在scene.train_cameras[resolution_scale]中影像的名称
    viewpoint_stack = scene.getTrainCameras().copy()

    # 读取影像匹配关系矩阵
    image_tensor = [[]]
    if (MatrixPath.split(".")[-1] == "png"):
        image = Image.open(MatrixPath)

        if image.mode != 'L':
            gray_image = image.convert('L')

            # 保存灰度图像
            gray_image.save(MatrixPath.split('.')[0] + "_Gray.png")
        else:
            gray_image = image

        # 定义转换器: 将图像转换为 tensor 格式
        transform_to_tensor = transforms.ToTensor()

        # 将图像转换为 tensor 格式
        image_tensor = transform_to_tensor(gray_image).tolist()
    elif (MatrixPath.split(".")[-1] == "txt"):
        MatrixFile = open(MatrixPath)
        MaxNum = 0
        for i in range(len(viewpoint_stack)):
            tempstr = MatrixFile.readline().split(",")
            MatrixLine = []
            for j in range(len(tempstr)):
                MatrixLine.append(int(tempstr[j]))
            if (MaxNum < max(MatrixLine)):
                MaxNum = max(MatrixLine)
            image_tensor[0].append(MatrixLine)

        for i in range(len(viewpoint_stack)):
            for j in range(len(viewpoint_stack)):
                image_tensor[0][i][j] = math.log(image_tensor[0][i][j] + 1) / math.log(MaxNum + 1)

    # 获取匹配矩阵中每一行对应的影像名称
    ImagesNameFile = open(ImagesNamePath)
    ImagesNames = []
    # ImagesNo = []
    Images = ImagesNameFile.readline().split(",")
    for i in range(len(Images)):
        ImagesNames.append(Images[i].split("\n")[0])
        # ImagesNo.append(Images[i].split(" ")[0])

    # 根据CamsList中影像的顺序，将image_tensor中每一行的顺序进行改变，使最终的矩阵的每一行与CamsList相对应
    Indexes = []
    Matrix = [[-1 for i in range(len(viewpoint_stack))] for i in range(len(viewpoint_stack))]
    for i in range(len(viewpoint_stack)):
        img_name = viewpoint_stack[i].image_name
        Indexes.append(ImagesNames.index(img_name))

    for i in range(len(viewpoint_stack)):
        for j in range(len(viewpoint_stack)):
            Matrix[i][j] = image_tensor[0][Indexes[i]][Indexes[j]]

    # 返回结果
    return Matrix

# 根据影像匹配关系确定影像权重
def GetImagesWeightsFromMatrix(MatrixPath, ImagesNamePath, scene, dataset):
    # 读入影像匹配关系矩阵
    ImageMatchMatrix = ReadInImageMatchMatrix(MatrixPath, ImagesNamePath, scene)

    # 读入旧模型对应的影像和新模型对应的影像
    files = os.listdir(dataset.source_path + r"/images")
    OldImagesList = [f for f in files if f.lower().endswith('.jpg') or f.lower().endswith('.png')]
    files = os.listdir(dataset.source_path_second + r"/images")
    NewImagesList = [f for f in files if f.lower().endswith('.jpg') or f.lower().endswith('.png')]

    # 找出新加入的影像的名字和原本就有的影像名字，同时确定新加入的影像在ALL_viewpoint_stack中对应的是哪几个Camera类，并返回索引值
    OldImages = []
    NewlyAddedImages = []
    NewlyAddedImagesIndex = []
    ALL_viewpoint_stack = scene.getTrainCameras().copy()
    for new_img_name in NewImagesList:
        if new_img_name not in OldImagesList:
            NewlyAddedImages.append(new_img_name.split('.')[0])
            for i in range(len(ALL_viewpoint_stack)):
                if ALL_viewpoint_stack[i].image_name == NewlyAddedImages[-1]:
                    NewlyAddedImagesIndex.append(i)
                    break
        else:
            OldImages.append(new_img_name.split('.')[0])
    print(NewlyAddedImages)
    # print("NewlyAddedImagesIndex:", NewlyAddedImagesIndex)

    # 为每一张影像赋予一个权值，代表后续影像训练中这张影像的重视程度，权值取值为[0, 1]
    # 新加入的影像权值直接赋值为1，其余影像根据重叠度决定权值
    Images_Weights = []
    Calculated = [] # 这个列表用于存储哪一些影像的权值不是0，不是0的标记为True
    ZeroWeightExist = False
    for i in range(len(ALL_viewpoint_stack)):
        if ALL_viewpoint_stack[i].image_name in NewlyAddedImages:
            ThisImageWeight = 1
            # print("ThisImageWeight=", ThisImageWeight)
        else:
            MatchDegree = 0
            for j in range(len(NewlyAddedImagesIndex)):
                MatchDegree = MatchDegree + ImageMatchMatrix[i][NewlyAddedImagesIndex[j]]
            ThisImageWeight = MatchDegree / len(NewlyAddedImages)
            # print("MatchDegree=", MatchDegree, "ThisImageWeight=", ThisImageWeight)

        # 若当前影像权值为0，则记录ZeroWeightExist为True
        if ThisImageWeight == 0:
            Calculated.append(False)
            ZeroWeightExist = True
        else:
            Calculated.append(True)

        Images_Weights.append(ThisImageWeight)

    # 如果存在影像的权值为0，则需要经过一定的方法为其赋予一个不为0的权值，具体参考詹老师提出的动态局部平差中设计的定权方法
    CalculateTime = 1
    while ZeroWeightExist and CalculateTime <= 4:
        ZeroWeightExist = False
        for i in range(len(Calculated)):
            if not Calculated[i]:
                MatchDegree = 0
                MatchTime = 0
                for j in range(len(Calculated)):
                    if Calculated[j] and ImageMatchMatrix[i][j] != 0:
                        MatchDegree = MatchDegree + ImageMatchMatrix[i][j] * Images_Weights[j]
                        MatchTime = MatchTime + 1
                ThisImageWeight = MatchDegree / MatchTime if MatchTime != 0 else 0
                Images_Weights[i] = ThisImageWeight

                if ThisImageWeight == 0:
                    ZeroWeightExist = True
                else:
                    Calculated[i] = True

        CalculateTime = CalculateTime + 1

    # 返回结果
    return Images_Weights, OldImages, NewlyAddedImages, ALL_viewpoint_stack, ImageMatchMatrix

# 根据动态监测结果来确定影像权重
def GetImageWeightsFromDynamicMonitoring(ALL_viewpoint_stack, gaussians, pipe, bg, opt):
    # 计算所有影像此时的PSNR和LOSS
    All_Predicted_Images = []
    ALL_Images_Loss = []
    ALL_Image_PSNR = []
    ALL_Image_Visibility_Filter = []
    ALL_Depth_Map = []
    All_Depth_Map_Hard = []
    for i in range(len(ALL_viewpoint_stack)):
        viewpoint_cam_copy = ALL_viewpoint_stack[i]
        render_pkg_copy = render(viewpoint_cam_copy, gaussians, pipe, bg)
        image_copy, viewspace_point_tensor_copy, visibility_filter_copy, radii_copy = render_pkg_copy["render"], \
                                                                                      render_pkg_copy["viewspace_points"], \
                                                                                      render_pkg_copy["visibility_filter"], \
                                                                                      render_pkg_copy["radii"]
        Depth_Map = render_pkg_copy["depth_map"]
        Depth_Map_Hard = render_pkg_copy["depth_map_hard"]

        ALL_Depth_Map.append(Depth_Map)
        All_Depth_Map_Hard.append(Depth_Map_Hard)
        ALL_Image_Visibility_Filter.append(visibility_filter_copy)
        All_Predicted_Images.append(image_copy)
        gt_image_copy = viewpoint_cam_copy.original_image.cuda()
        Ll1_copy = l1_loss(image_copy, gt_image_copy)
        loss_copy = (1.0 - opt.lambda_dssim) * Ll1_copy + opt.lambda_dssim * (1.0 - ssim(image_copy, gt_image_copy))
        PSNR_copy = psnr(image_copy, gt_image_copy)
        ALL_Images_Loss.append(loss_copy.cpu())
        ALL_Image_PSNR.append(PSNR_copy.cpu().sum().item() / 3)

    # 根据PSNR和LOSS来确定影像权重
    Weights = []
    MinPSNR = min(ALL_Image_PSNR)
    for i in range(len(ALL_viewpoint_stack)):
        Weights.append(MinPSNR / ALL_Image_PSNR[i])

    # 返回结果
    return Weights, ALL_Images_Loss, ALL_Image_PSNR, All_Predicted_Images, ALL_Image_Visibility_Filter, ALL_Depth_Map, All_Depth_Map_Hard

# 第一个场景的高斯模型点云训练
def training_firstGaussian(dataset, opt, pipe, testing_iterations, checkpoint, debug_from, FirstTrainingIterations,
                           ImagesAlreadyBeTrainedIterations, Diary, NoDebug=False, EvaluateDiary=None):
    Diary.write('\n')
    DebugTime = 0

    # 设置初始训练次数
    first_iter = 0

    # 主要进行一系列的模型以及其它文件的输出准备（例如构建输出文件夹等），并返回一个Tensorboard writer对象
    tb_writer = prepare_output_and_logger(dataset)

    # 初始化3D Gaussian Splatting模型，主要是一些模型属性的初始化和神经网络中一些激活函数的初始化
    gaussians = GaussianModel(dataset.sh_degree)

    # 初始化一个场景
    scene = Scene(dataset, gaussians, load_iteration=None, shuffle=True, resolution_scales=[1.0])

    if pipe.UseDepthLoss:
        scene.ReadInDepthMapAll(dataset.depth_map_path, dataset.data_device, dataset.depth_scale)

    # 进行一些训练上的初始化设置
    gaussians.training_setup(opt)

    # 若checkpoint存在，将checkpoint中的信息和数据导入到模型之中
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 初始化背景颜色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 这是两个计时器，专门用于记录训练开始和结束时的时刻
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    TargetCam = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(0, FirstTrainingIterations), desc="Training progress", initial=first_iter)
    first_iter += 1
    for iteration in range(first_iter, FirstTrainingIterations + 1):
        # 下面这一段应该是在连接服务器并将训练状态实时反馈在服务器上
        '''
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(FirstTrainingIterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        '''

        # 记录开始训练的时刻
        iter_start.record()

        # 根据现在的训练次数来更新学习率（使用学习率的指数下降）
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # 每训练1000次，重新将球谐函数的阶数增加到设定的最大阶数
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        # 当摄影机视点栈堆为空时：
        if not viewpoint_stack:
            # 将Scene类中所有的self.train_cameras中指定缩放比例的训练影像及其相关信息存入到摄影机视点栈堆中
            viewpoint_stack = scene.getTrainCameras().copy()

        if TargetCam == None:
            RenderCam = copy.deepcopy(viewpoint_stack[0])

            RenderCam.R[0][0] = 0.99943693
            RenderCam.R[0][1] = 0.02387958
            RenderCam.R[0][2] = -0.02357095
            RenderCam.R[1][0] = -0.02322576
            RenderCam.R[1][1] = 0.99934829
            RenderCam.R[1][2] = 0.02763265
            RenderCam.R[2][0] = 0.02421545
            RenderCam.R[2][1] = -0.02706963
            RenderCam.R[2][2] = 0.99934021

            RenderCam.T[0] = -18.03593356  # +向左，-向右
            RenderCam.T[1] = 1.58552898    # +向上，-向下
            RenderCam.T[2] = 26.30952862   # +向后，-向前

            TargetCam = Camera(colmap_id=-2, R=RenderCam.R, T=RenderCam.T,
                    FoVx=RenderCam.FoVx, FoVy=RenderCam.FoVy,
                    image=RenderCam.original_image, gt_alpha_mask=None,
                    image_name='Target', uid=-2, data_device=args.data_device)

        # 随机从摄影机视点栈堆中任意取出一个影像以及其视点信息
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # 检查viewpoint_cam对应的影像是否在ImagesAlreadyBeTrainedIterations中出现过，若没有出现过，则需要将这张影像加进去并将已训练的次数设置为0
        if viewpoint_cam.image_name not in ImagesAlreadyBeTrainedIterations:
            ImagesAlreadyBeTrainedIterations[viewpoint_cam.image_name] = 0

        # 影像渲染
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # 设置背景颜色，一般是白色
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # 进行影像渲染，渲染指定视点的影像
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
                                                                  render_pkg["visibility_filter"], render_pkg["radii"]

        # load_distribution中每一个值代表的是这一张影像上对每一个像素进行渲染时使用到的高斯求数量
        if not pipe.NotGetLoad:
            load_distribution = render_pkg["load_distribution"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # PGSR Scale loss
        if pipe.UseScaleLoss:
            scale_loss_weight = 100.0
            visibility_filter_bool = render_pkg["visibility_filter_bool"]
            if visibility_filter_bool.sum() > 0:
                scale = gaussians.get_scaling[visibility_filter_bool]
                sorted_scale, _ = torch.sort(scale, dim=-1)
                min_scale_loss = sorted_scale[..., 0]
                loss += scale_loss_weight * min_scale_loss.mean()

        # 以下是负载均衡的损失函数
        if not pipe.NotGetLoad:
            load_loss = torch.std(load_distribution)  # 计算标准差
            loss_adj = math.floor(math.log10(load_loss)) - math.floor(math.log10(loss))
            load_loss = load_loss / math.pow(10, loss_adj + 1.0)

        # 深度图
        if pipe.UseDepthLoss:
            box_p = 128
            p_corr = 0.5
            lambda_SoftHard = 0.00001
            lambda_pearson = 0.05
            lambda_local_pearson = 0.15

            Depth_Map = render_pkg["depth_map"]
            Depth_Map_Hard = render_pkg["depth_map_hard"]
            Depth_Map_GT = scene.DepthMap[viewpoint_cam.image_name]

            soft_hard_loss = Soft_Hard_loss(Depth_Map.squeeze(0), Depth_Map_Hard.squeeze(0)) * lambda_SoftHard
            pearson_loss = pearson_depth_loss(Depth_Map.squeeze(0), Depth_Map_GT.squeeze(0)) * lambda_pearson
            # lp_loss = local_pearson_loss(Depth_Map.squeeze(0), Depth_Map_GT.squeeze(0), box_p, p_corr) * lambda_local_pearson

            # 深度损失
            # loss = loss + pearson_loss + lp_loss + soft_hard_loss
            loss = loss + pearson_loss + soft_hard_loss

        # 将原本的损失与负载均衡的损失加权平均
        if not pipe.NotGetLoad:
            loss = loss * (1 - opt.lambda_load) + opt.lambda_load * load_loss

        loss.backward()

        iter_end.record()  # 结束这一段的计时

        with torch.no_grad():
            if iteration % 100 == 0:
                ImageOutputDir = os.path.join(args.Model_Path_Dir, 'TargetCam')
                os.makedirs(ImageOutputDir, exist_ok=True)
                RenderTargetCam(TargetCam, gaussians, pipe, bg, iteration, ImageOutputDir)

            # 计算一个ema_loss_for_log特殊的损失
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            # 每十次训练更新一次进度条
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == FirstTrainingIterations:
                progress_bar.close()

            # 向Tensorboard writer中写入一些关于现在训练的一些信息
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))

            # 稠密化
            if iteration < opt.densify_until_iter:
                # visibility_filter 是一个布尔掩码，标识哪些高斯分布在当前视图中可见。
                # 通过 torch.max() 函数，代码将高斯的最大半径值（可能是图像空间中的半径）进行更新，保存当前可见高斯中半径的最大值。
                # 这样做可能是为了记录哪些高斯对象在图像中投影占据了最大的空间。
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])

                # 这一步将视图空间中的点 viewspace_point_tensor 和 visibility_filter 传递给 gaussians.add_densification_stats() 方法，可能是为了收集一些统计数据
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # 当训练次数处于[opt.densify_from_iter, opt.densify_until_iter]之间且为opt.densification_interval的整数倍
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # 只有当训练次数大于opt.opacity_reset_interval时才设置大小阈值为20
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    # 执行稠密化以及修剪的操作
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_threshold, scene.cameras_extent, size_threshold, radii)

                # 重新设置可见度
                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                if opt.optimizer_type == "default":
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                elif opt.optimizer_type == "sparse_adam":
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)

            # 每训练一定次数输出一次模型评估结果
            if ((iteration % 500 == 0 and iteration != 0) or iteration == FirstTrainingIterations):
                ThisDebugTimeStart = time.time()
                EvaluateModel(iteration, l1_loss, scene, render, (pipe, background), Diary, EvaluateDiary=EvaluateDiary)
                ThisDebugTimeEnd = time.time()
                DebugTime = DebugTime + ThisDebugTimeEnd - ThisDebugTimeStart

            # 完成这一次训练后，让当前训练影像的已训练次数+1
            ImagesAlreadyBeTrainedIterations[viewpoint_cam.image_name] = ImagesAlreadyBeTrainedIterations[viewpoint_cam.image_name] + 1

    # 返回训练好的场景
    ThisDebugTimeStart = time.time()

    if not NoDebug:
        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        print("\n[ITER {}] Saving Gaussians".format(iteration))
        scene.save(iteration)

    Diary.write("\n[ITER {}] Saving Checkpoint".format(iteration))
    Diary.write("\n[ITER {}] Saving Gaussians".format(iteration))
    ThisDebugTimeEnd = time.time()
    DebugTime = DebugTime + ThisDebugTimeEnd - ThisDebugTimeStart

    return scene, iteration, DebugTime

# 点云初始化，将新的场景对应的系数点云转变为高斯点云的形式
def SceneInitialQuickly(dataset, opt, Diary):
    # 设置初始训练次数
    first_iter = 1

    # 主要进行一系列的模型以及其它文件的输出准备（例如构建输出文件夹等），并返回一个Tensorboard writer对象
    tb_writer = prepare_output_and_logger_second(dataset)

    # 初始化3D Gaussian Splatting模型，主要是一些模型属性的初始化和神经网络中一些激活函数的初始化
    gaussians = GaussianModel(dataset.sh_degree)

    # 初始化一个场景
    scene = Scene_Second(dataset, gaussians, OnlyInitial=True)

    # 进行一些训练上的初始化设置
    gaussians.training_setup(opt)

    # 初始化Adam优化器，将state中的所有值均设置为0
    gaussians.InitialOptimizer()

    print("\nFinish Second Scene Initialization!!!")
    Diary.write(f"New scene points Num: {gaussians.get_xyz.shape[0]}\n")
    Diary.write("\nFinish Second Scene Initialization!!!\n")

    # 返回初始化的场景
    '''
    print("\n[ITER {}] Saving Checkpoint".format(first_iter))
    torch.save((gaussians.capture(), first_iter), scene.model_path + "/chkpnt" + str(first_iter) + ".pth")
    print("\n[ITER {}] Saving Gaussians".format(first_iter))
    scene.save(first_iter)
    '''

    return scene

# 用于确定新场景的3DGS点云中哪一些点是新加入的
def Get3DPointsMask(FirstScene : Scene, SecondScene : Scene_Second, args, distance_threshold=1, distance_buffeer=1.5):
    MatrixPath = args.source_path_second + r"/sparse/0/imageMatchMatrix.png"
    if not os.path.exists(MatrixPath):
        MatrixPath = args.source_path_second + r"/sparse/0/imageMatchMatrix.txt"
    ImagesNamePath = args.source_path_second + r"/sparse/0/imagesNames.txt"

    MatchMatrix = ReadInImageMatchMatrix(MatrixPath, ImagesNamePath, FirstScene)
    if max(MatchMatrix[-1]) < 0.95:

        points1 = FirstScene.gaussians.get_xyz.detach().cpu().numpy()
        points2 = SecondScene.gaussians.get_xyz.detach().cpu().numpy()

        # 使用 cKDTree 加速查找重叠点
        tree = cKDTree(points1)
        distances, indices = tree.query(points2, distance_upper_bound=distance_threshold)

        # 过滤掉与 vertices1 中距离小于阈值的点
        mask = distances >= distance_threshold  # True 表示保留该点
        points3 = points2[mask]

        # 使用 cKDTree 加速查找重叠点
        tree = cKDTree(points3)
        distances, indices = tree.query(points2, distance_upper_bound=distance_threshold * distance_buffeer)

        # 过滤掉与 filtered_vertices2 中距离大于阈值的点
        mask = distances <= distance_threshold * distance_buffeer  # True 表示保留该点
    else:
        pointsNum = SecondScene.gaussians.get_xyz.shape[0]
        mask = np.zeros(pointsNum, dtype=bool)

    return mask

# 用于确定新场景的3DGS点云中哪一些点是新加入的（这里直接对新加入的影像进行一次渲染，然后根据返回的ALL_Image_Visibility_Filter来判断哪一些点是新加入的点）
# 此方法相比于上面的那个方法应该会快一些，且不会因为模型中点的变多而越来越慢
def Get3DPointsMaskQuickly(ALL_viewpoint_stack, gaussians, args, pipe, NewImagesNum):
    # 初始化背景颜色
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = torch.rand((3), device="cuda") if args.random_background else background

    mask = None
    for i in range(NewImagesNum):
        New_viewpoint_cam = ALL_viewpoint_stack[-i]
        render_pkg_copy = render(New_viewpoint_cam, gaussians, pipe, bg)
        _, __, Sub_mask, ___ = render_pkg_copy["render"], render_pkg_copy["viewspace_points"], \
                                        render_pkg_copy["visibility_filter"], render_pkg_copy["radii"]

        if mask == None:
            mask = Sub_mask.clone()
        else:
            # mask = mask | Sub_mask
            mask = np.union1d(mask, Sub_mask)

    return mask.squeeze()

# 获取合并后的场景的一些高斯模型的参数
def MergeOptimizationDict(Model_Old_DICT, Model_New_DICT, mask):
    OLD_STATE = Model_Old_DICT['state']
    NEW_STATE = Model_New_DICT['state']
    OLD_ParamGroups = Model_Old_DICT['param_groups']
    New_ParamGroups = Model_New_DICT['param_groups']
    MERGE_STATE = {}
    MERGE_ParamGroups = []

    for key_X in OLD_STATE.keys():
        newstate = {}
        for key_Y in OLD_STATE[key_X].keys():
            if (key_Y == 'step'):
                newstate[key_Y] = OLD_STATE[key_X][key_Y]
            else:
                newstate[key_Y] = torch.cat((OLD_STATE[key_X][key_Y], NEW_STATE[key_X][key_Y][mask]), dim=0)
        MERGE_STATE[key_X] = newstate

    for i in range(len(New_ParamGroups)):
        newParamGroups = {}
        for key in New_ParamGroups[i].keys():
            newParamGroups[key] = New_ParamGroups[i][key]
        MERGE_ParamGroups.append(newParamGroups)

    NewDict = {'state': MERGE_STATE, 'param_groups': MERGE_ParamGroups}
    return NewDict

def GetMergeSceneGaussianParametersDict(FirstScene : Scene, SecondScene : Scene_Second, mask, Diary):
    FirstSceneModel = FirstScene.gaussians.capture()
    SecondSceneModel = SecondScene.gaussians.capture()

    MergeDict = {}

    # 球谐函数阶数：使用旧模型对应的数据
    MergeDict['active_sh_degree'] = FirstSceneModel[0]

    # 相机分布范围：使用新模型对应的数据
    MergeDict['spatial_lr_scale'] = SecondSceneModel[11]

    # 其余数据均与3D Gaussian椭球有关，需要将新旧模型的数据融合在一起
    MergeDict['optimizer_state_dict'] = MergeOptimizationDict(FirstSceneModel[10], SecondSceneModel[10], mask)
    MergeDict['xyz'] = torch.cat((FirstSceneModel[1], SecondSceneModel[1][mask]), dim=0)
    MergeDict['features_dc'] = torch.cat((FirstSceneModel[2], SecondSceneModel[2][mask]), dim=0)
    MergeDict['features_rest'] = torch.cat((FirstSceneModel[3], SecondSceneModel[3][mask]), dim=0)
    MergeDict['scaling'] = torch.cat((FirstSceneModel[4], SecondSceneModel[4][mask]), dim=0)
    MergeDict['rotation'] = torch.cat((FirstSceneModel[5], SecondSceneModel[5][mask]), dim=0)
    MergeDict['opacity'] = torch.cat((FirstSceneModel[6], SecondSceneModel[6][mask]), dim=0)
    MergeDict['max_radii2D'] = torch.cat((FirstSceneModel[7], SecondSceneModel[7][mask]), dim=0)
    MergeDict['xyz_gradient_accum'] = torch.cat((FirstSceneModel[8], SecondSceneModel[8][mask]), dim=0)
    MergeDict['denom'] = torch.cat((FirstSceneModel[9], SecondSceneModel[9][mask]), dim=0)

    Diary.write(f"Origin Scene Points Num: {FirstSceneModel[1].shape[0]}\n")
    Diary.write(f"Newly Added Points Num: {SecondSceneModel[1][mask].shape[0]}\n")
    Diary.write(f"Merge Scene Points Num: {MergeDict['xyz'].shape[0]}\n")

    # 返回结果
    return MergeDict

# 不使用前面的场景合并方法，而是直接使用RGB以及深度图
def ExpandGS_RGBD(Scene, cam):
    Scene.gaussians.extend_from_pcd_seq(cam, depthmap=Scene.DepthMap[cam.image_name])

# 进行合并后的场景的模型训练
def training_MergeScene(args, MergeScene, dataset, opt, pipe, debug_from, OnlyTrainNewScene, AlreadyIterations,
                        FirstTrainingIterations, MergeTrainingIterations, ImagesAlreadyBeTrainedIterations, Diary,
                        NoDebug=False, RenderAllImages=False, EvaluateDiary=None):

    # 设置初始训练次数
    first_iter = 0
    DebugTime = 0

    gaussians = MergeScene.gaussians
    scene = MergeScene

    # 初始化背景颜色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 设置背景颜色，一般是白色
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    # 获取影像权重、旧的影像列表、新加入的影像列表、所有影像对应的Camera类列
    Diary.write(
        f"Get Images_Match_Matrix From {dataset.source_path_second + r'/sparse/0/imageMatchMatrix.png or .txt'}\n")
    Diary.write(f"Get Images_Name_Path From {dataset.source_path_second + r'/sparse/0/imagesNames.txt'}\n")
    MatrixPath = dataset.source_path_second + r"/sparse/0/imageMatchMatrix.png"
    if not os.path.exists(MatrixPath):
        MatrixPath = dataset.source_path_second + r"/sparse/0/imageMatchMatrix.txt"
    ImagesNamePath = dataset.source_path_second + r"/sparse/0/imagesNames.txt"
    Images_Weights, OldImages, NewlyAddedImages, ALL_viewpoint_stack, ImageMatchMatrix = GetImagesWeightsFromMatrix(
        MatrixPath,
        ImagesNamePath, scene,
        dataset)

    if pipe.InitialTrainingTimesSetZero:
        # 检查viewpoint_cam对应的影像是否在ImagesAlreadyBeTrainedIterations中出现过，若没有出现过，则需要将这张影像加进去并将已训练的次数设置为0
        for i in range(len(NewlyAddedImages)):
            ImagesAlreadyBeTrainedIterations[NewlyAddedImages[i]] = 0
    else:
        # 对于新加入的影像，为其计算一个等效训练次数，而不是直接将这张影像的初始已训练次数设置为0
        for i in range(len(NewlyAddedImages)):
            viewpoint_cam_Index = -1
            for j in range(len(ALL_viewpoint_stack)):
                if NewlyAddedImages[i] == ALL_viewpoint_stack[j].image_name:
                    viewpoint_cam_Index = j
            Equivalent_training_times = 0
            RelatedImagesNum = 0
            for j in range(len(ImageMatchMatrix[viewpoint_cam_Index])):
                if ImageMatchMatrix[viewpoint_cam_Index][j] != 0 and ALL_viewpoint_stack[j].image_name in ImagesAlreadyBeTrainedIterations.keys():
                    Equivalent_training_times = Equivalent_training_times + ImageMatchMatrix[viewpoint_cam_Index][j] * ImagesAlreadyBeTrainedIterations[ALL_viewpoint_stack[j].image_name]
                    RelatedImagesNum = RelatedImagesNum + 1
            Equivalent_training_times = Equivalent_training_times / RelatedImagesNum
            ImagesAlreadyBeTrainedIterations[NewlyAddedImages[i]] = int(Equivalent_training_times)

    # 这是两个计时器，专门用于记录训练开始和结束时的时刻
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(0, AlreadyIterations + MergeTrainingIterations * len(NewlyAddedImages)), desc="Training progress", initial=AlreadyIterations)
    first_iter = FirstTrainingIterations + 1

    for iteration in range(first_iter, FirstTrainingIterations + MergeTrainingIterations * len(NewlyAddedImages) + 1):
        # 下面这一段应该是在连接服务器并将训练状态实时反馈在服务器上
        '''
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        '''
        # 记录开始训练的时刻
        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # 每训练1000次，重新将球谐函数的阶数增加到设定的最大阶数
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # viewpoint_stack用于存储后续用于训练的Camera类
        if not viewpoint_stack:
            viewpoint_stack = []

            # 如果只针对新加入的影像进行训练
            if OnlyTrainNewScene:
                for vs in ALL_viewpoint_stack:
                    if vs.image_name in NewlyAddedImages:
                        viewpoint_stack.append(vs)
            # 如果根据影像权重来决定训练影像
            else:
                # viewpoint_stack = GetImageChosenTimesByWeights(Images_Weights, NewlyAddedImages, ALL_viewpoint_stack, MergeTrainingIterations * len(NewlyAddedImages), Diary)

                viewpoint_stack = OnlyTrainMaxWeightsImages(Images_Weights, NewlyAddedImages, ALL_viewpoint_stack, MergeTrainingIterations * len(NewlyAddedImages), Diary)

            if args.OnlyUseGlobalOptimization:
                # 将Scene类中所有的self.train_cameras中指定缩放比例的训练影像及其相关信息存入到摄影机视点栈堆中
                viewpoint_stack = scene.getTrainCameras().copy()
                Diary.write("ALL image's weight are the same!!\n")

        # 随机从摄影机视点栈堆中任意取出一个影像以及其视点信息
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # 调整椭球3D位置坐标的学习率
        if not args.DifferentImagesLr:
            # 根据现在的训练次数来更新学习率（使用学习率的指数下降）
            gaussians.update_learning_rate(iteration - first_iter + AlreadyIterations)
        else:
            # 每一张影像将根据自身以及和周围影像的训练次数来决定学习率
            gaussians.UpdateDifferentImageLearningRate(viewpoint_cam, ImagesAlreadyBeTrainedIterations, args,
                                                       ALL_viewpoint_stack, ImageMatchMatrix)

        # 影像渲染
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # 进行影像渲染，渲染指定视点的影像
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
                                                                  render_pkg["visibility_filter"], render_pkg["radii"]

        # load_distribution中每一个值代表的是这一张影像上对每一个像素进行渲染时使用到的高斯求数量
        if not pipe.NotGetLoad:
            load_distribution = render_pkg["load_distribution"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # PGSR Scale loss
        if pipe.UseScaleLoss:
            scale_loss_weight = 100.0
            visibility_filter_bool = render_pkg["visibility_filter_bool"]
            if visibility_filter_bool.sum() > 0:
                scale = gaussians.get_scaling[visibility_filter_bool]
                sorted_scale, _ = torch.sort(scale, dim=-1)
                min_scale_loss = sorted_scale[..., 0]
                loss += scale_loss_weight * min_scale_loss.mean()

        # 以下是负载均衡的损失函数
        if not pipe.NotGetLoad:
            load_loss = torch.std(load_distribution)  # 计算标准差
            loss_adj = math.floor(math.log10(load_loss)) - math.floor(math.log10(loss))
            load_loss = load_loss / math.pow(10, loss_adj + 1.0)

        # 深度图
        if pipe.UseDepthLoss:
            box_p = 128
            p_corr = 0.5
            lambda_SoftHard = 0.00001
            lambda_pearson = 0.05
            lambda_local_pearson = 0.15

            Depth_Map = render_pkg["depth_map"]
            Depth_Map_Hard = render_pkg["depth_map_hard"]
            Depth_Map_GT = scene.DepthMap[viewpoint_cam.image_name]

            soft_hard_loss = Soft_Hard_loss(Depth_Map.squeeze(0), Depth_Map_Hard.squeeze(0)) * lambda_SoftHard
            pearson_loss = pearson_depth_loss(Depth_Map.squeeze(0), Depth_Map_GT.squeeze(0)) * lambda_pearson
            # lp_loss = local_pearson_loss(Depth_Map.squeeze(0), Depth_Map_GT.squeeze(0), box_p, p_corr) * lambda_local_pearson

            # 深度损失
            # loss = loss + pearson_loss + lp_loss + soft_hard_loss
            loss = loss + pearson_loss + soft_hard_loss

        # 将原本的损失与负载均衡的损失加权平均
        if not pipe.NotGetLoad:
            loss = loss * (1 - opt.lambda_load) + opt.lambda_load * load_loss

        loss.backward()

        iter_end.record()  # 结束这一段的计时

        with torch.no_grad():
            # 计算一个ema_loss_for_log特殊的损失
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            # 每十次训练更新一次进度条
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 稠密化
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration % args.MergeScene_Densification_Interval == 0:
                # 只有当训练次数大于opt.opacity_reset_interval时才设置大小阈值为20
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                # 执行稠密化以及修剪的操作
                if args.DifferentImagesDensify:
                    # 获取所有高斯球在每一张影像上的可见性
                    _, __, ___, ____, ALL_Image_Visibility_Filter = GetImageWeightsFromDynamicMonitoring(
                        ALL_viewpoint_stack, gaussians, pipe, bg, opt)

                    gaussians.DifferentImage_Densify_and_Prune(args.MergeScene_Densify_Grad_Threshold, opt.opacity_threshold,
                                                               scene.cameras_extent
                                                               , size_threshold, radii, args, ALL_viewpoint_stack,
                                                               ImagesAlreadyBeTrainedIterations,
                                                               ImageMatchMatrix, ALL_Image_Visibility_Filter)
                else:
                    gaussians.densify_and_prune(args.MergeScene_Densify_Grad_Threshold, opt.opacity_threshold, scene.cameras_extent,
                                                size_threshold, radii)

            '''
            if iteration < opt.densify_until_iter:
                # visibility_filter 是一个布尔掩码，标识哪些高斯椭球在当前视图中可见。
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

                # 这一步将视图空间中的点 viewspace_point_tensor 和 visibility_filter 传递给 gaussians.add_densification_stats() 方法，可能是为了收集一些统计数据
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # 当训练次数处于[opt.densify_from_iter, opt.densify_until_iter]之间且为opt.densification_interval的整数倍
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # 只有当训练次数大于opt.opacity_reset_interval时才设置大小阈值为20
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    # 执行稠密化以及修剪的操作
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                # 重新设置可见度
                if (iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter)):
                    gaussians.reset_opacity()


            # 参数优化器梯度更新
            if iteration < opt.iterations:
                # 根据反向传播计算的梯度更新模型参数
                gaussians.optimizer.step()

                # 这行代码用于在每次优化步骤之后清除所有计算得到的梯度，防止它们在下一次反向传播时被累积。
                gaussians.optimizer.zero_grad(set_to_none=True)'''

            # 完成这一次训练后，让当前训练影像的已训练次数+1
            ImagesAlreadyBeTrainedIterations[viewpoint_cam.image_name] = ImagesAlreadyBeTrainedIterations[viewpoint_cam.image_name] + 1

            if (iteration == FirstTrainingIterations + MergeTrainingIterations * len(NewlyAddedImages)) and not args.DifferentImagesResetOpacity:
                CurrentActualIterations1 = iteration - first_iter + AlreadyIterations + 1
                CurrentActualIterations2 = iteration - first_iter + AlreadyIterations + 1 - MergeTrainingIterations * len(NewlyAddedImages)
                Do = CurrentActualIterations1 - CurrentActualIterations1 % opt.opacity_reset_interval >= CurrentActualIterations2
                if CurrentActualIterations1 % opt.opacity_reset_interval == 0 or Do:
                    if CurrentActualIterations1 <= args.ResetOpacityUntilIter or CurrentActualIterations2 <= args.ResetOpacityUntilIter:
                        IM = [int(name) for name in os.listdir(args.Source_Path_Dir) if
                              os.path.isdir(os.path.join(args.Source_Path_Dir, name))]
                        IM = sorted(IM)
                        WholeIterations = args.IterationFirstScene + (len(IM) - 2) * args.IterationPerMergeScene + \
                                          int((len(IM) - 3) / args.GlobalOptimizationInterval) * args.GlobalOptimizationIteration + \
                                          args.FinalOptimizationIterations

                        if CurrentActualIterations1 <= WholeIterations - args.opacity_reset_interval or CurrentActualIterations2 <= WholeIterations - args.opacity_reset_interval:
                            print(f"[{CurrentActualIterations1} its] => Reset Opacity!!")
                            gaussians.reset_opacity()
            elif (iteration == FirstTrainingIterations + MergeTrainingIterations * len(NewlyAddedImages)) and args.DifferentImagesResetOpacity:
                CurrentActualIterations1 = iteration - first_iter + AlreadyIterations + 1
                CurrentActualIterations2 = iteration - first_iter + AlreadyIterations + 1 - MergeTrainingIterations * len(NewlyAddedImages)
                Do = CurrentActualIterations1 - CurrentActualIterations1 % opt.opacity_reset_interval >= CurrentActualIterations2
                if CurrentActualIterations1 % opt.opacity_reset_interval == 0 or Do:
                    if CurrentActualIterations1 <= args.ResetOpacityUntilIter or CurrentActualIterations2 <= args.ResetOpacityUntilIter:
                        IM = [int(name) for name in os.listdir(args.Source_Path_Dir) if os.path.isdir(os.path.join(args.Source_Path_Dir, name))]
                        IM = sorted(IM)
                        WholeIterations = args.IterationFirstScene + (len(IM) - 2) * args.IterationPerMergeScene + \
                                          int((len(IM) - 3) / args.GlobalOptimizationInterval) * args.GlobalOptimizationIteration + \
                                          args.FinalOptimizationIterations

                        if CurrentActualIterations1 <= WholeIterations - args.opacity_reset_interval or CurrentActualIterations2 <= WholeIterations - args.opacity_reset_interval:
                            print(f"[{CurrentActualIterations1} its] => Reset Opacity!!")
                            _, __, ___, ____, ALL_Image_Visibility_Filter, _____, ______ = GetImageWeightsFromDynamicMonitoring(ALL_viewpoint_stack, gaussians, pipe, bg, opt)

                            # gaussians.DifferentImage_ResetOpacity(ResetOpacityMask)

            # Optimizer step
            if opt.optimizer_type == "default":
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            elif opt.optimizer_type == "sparse_adam":
                visible = radii > 0
                gaussians.optimizer.step(visible, radii.shape[0])
                gaussians.optimizer.zero_grad(set_to_none=True)

            # 完成最后一次训练后，首先进行模型评估并输出一次全部的训练影像的模型渲染结果，然后可能会进行可见度重新设置
            if (iteration == FirstTrainingIterations + MergeTrainingIterations * len(NewlyAddedImages)):
                ThisDebugTimeStart = time.time()

                # 模型评估
                EvaluateModel(iteration - first_iter + AlreadyIterations + 1, l1_loss, scene, render,
                              (pipe, background), Diary, EvaluateDiary=EvaluateDiary)

                if (not NoDebug) and RenderAllImages:
                    print("Output All Images Rendering Results!")

                    # 对所有训练影像都进行一次预测并保存结果
                    RenderAllImagesAndEvaluate(args, ALL_viewpoint_stack, gaussians, pipe, bg, opt, AlreadyIterations,
                                               FirstTrainingIterations, first_iter, iteration, NewlyAddedImages)

                ThisDebugTimeEnd = time.time()
                DebugTime = DebugTime + ThisDebugTimeEnd - ThisDebugTimeStart

                '''
                # 重新设置可见度
                if (((iteration - first_iter + AlreadyIterations + 1) % opt.opacity_reset_interval == 0 or
                    (iteration - first_iter + AlreadyIterations + 1 - MergeTrainingIterations) % opt.opacity_reset_interval == 0)) and\
                        (iteration - first_iter + AlreadyIterations + 1 <= opt.densify_until_iter or
                         iteration - first_iter + AlreadyIterations + 1 - MergeTrainingIterations <= opt.densify_until_iter):
                    if (iteration - first_iter + AlreadyIterations + 1 <= args.ResetOpacityUntilIter or
                         iteration - first_iter + AlreadyIterations + 1 - MergeTrainingIterations <= args.ResetOpacityUntilIter):
                        print(f"[{iteration - first_iter + AlreadyIterations + 1} its] => Reset Opacity!!")
                        gaussians.reset_opacity()
                '''

                if not args.OnlyUseLocalOptimization:
                    return scene, AlreadyIterations + MergeTrainingIterations * len(NewlyAddedImages), DebugTime, ImageMatchMatrix
                else:
                    return scene, AlreadyIterations + MergeTrainingIterations * len(NewlyAddedImages), DebugTime, ImageMatchMatrix, Images_Weights, NewlyAddedImages

# 模型全局训练
def ModelGlobalOptimization(args, MergeScene, dataset, opt, pipe, debug_from, AlreadyIterations, FirstTrainingIterations,
                            MergeTrainingIterations, ImagesAlreadyBeTrainedIterations, ImageMatchMatrix, Diary,
                            NoDebug=False, RenderAllImages=False, NewImagesNum=1, Images_Weights=None, NewlyAddedImages=None, IsFinalOptimization=False, EvaluateDiary=None):

    if True:
        # 设置初始训练次数
        first_iter = 0
        DebugTime = 0

        gaussians = MergeScene.gaussians
        scene = MergeScene

        # 初始化背景颜色
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 设置背景颜色，一般是白色
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # 获取影像权重、旧的影像列表、新加入的影像列表、所有影像对应的Camera类列
        ALL_viewpoint_stack = scene.getTrainCameras().copy()
        print("This is Global optimization, do not read in the image weights!!")
        Diary.write("This is Global optimization, do not read in the image weights!!\n")

        # 这是两个计时器，专门用于记录训练开始和结束时的时刻
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        viewpoint_stack = None
        ema_loss_for_log = 0.0

        progress_bar = tqdm(range(0, AlreadyIterations + MergeTrainingIterations * NewImagesNum), desc="Training progress",
                            initial=AlreadyIterations)
        first_iter = FirstTrainingIterations + 1

        for iteration in range(first_iter, FirstTrainingIterations + MergeTrainingIterations * NewImagesNum + 1):
            # 下面这一段应该是在连接服务器并将训练状态实时反馈在服务器上
            '''
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                                   0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None
            '''
            # 记录开始训练的时刻
            iter_start.record()

            # Every 1000 its we increase the levels of SH up to a maximum degree
            # 每训练1000次，重新将球谐函数的阶数增加到设定的最大阶数
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # viewpoint_stack用于存储后续用于训练的Camera类
            if not viewpoint_stack:
                if args.OnlyUseLocalOptimization and not IsFinalOptimization:
                    viewpoint_stack = []
                    viewpoint_stack = GetImageChosenTimesByWeights(Images_Weights, NewlyAddedImages,
                                                                   ALL_viewpoint_stack,
                                                                   MergeTrainingIterations * len(NewlyAddedImages),
                                                                   Diary)
                else:
                    viewpoint_stack = scene.getTrainCameras().copy()
                    Diary.write("ALL image's weight are the same!!\n")
                    if IsFinalOptimization:
                        # 获取TargetCam，用于模型训练结果的展示
                        TargetCam = None
                        if TargetCam == None:
                            RenderCam = copy.deepcopy(viewpoint_stack[0])

                            RenderCam.R[0][0] = 0.99943693
                            RenderCam.R[0][1] = 0.02387958
                            RenderCam.R[0][2] = -0.02357095
                            RenderCam.R[1][0] = -0.02322576
                            RenderCam.R[1][1] = 0.99934829
                            RenderCam.R[1][2] = 0.02763265
                            RenderCam.R[2][0] = 0.02421545
                            RenderCam.R[2][1] = -0.02706963
                            RenderCam.R[2][2] = 0.99934021

                            RenderCam.T[0] = -18.03593356  # +向左，-向右
                            RenderCam.T[1] = 1.58552898  # +向上，-向下
                            RenderCam.T[2] = 26.30952862  # +向后，-向前

                            TargetCam = Camera(colmap_id=-2, R=RenderCam.R, T=RenderCam.T,
                                               FoVx=RenderCam.FoVx, FoVy=RenderCam.FoVy,
                                               image=RenderCam.original_image, gt_alpha_mask=None,
                                               image_name='Target', uid=-2, data_device=args.data_device)

            # 随机从摄影机视点栈堆中任意取出一个影像以及其视点信息
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

            # 调整椭球3D位置坐标的学习率
            if not args.DifferentImagesLr:
                # 根据现在的训练次数来更新学习率（使用学习率的指数下降）
                gaussians.update_learning_rate(iteration - first_iter + AlreadyIterations)
            else:
                # 每一张影像将根据自身以及和周围影像的训练次数来决定学习率
                if not IsFinalOptimization:
                    gaussians.UpdateDifferentImageLearningRate(viewpoint_cam, ImagesAlreadyBeTrainedIterations, args,
                                                           ALL_viewpoint_stack, ImageMatchMatrix)
                else:
                    gaussians.update_learning_rate(30000 - MergeTrainingIterations + iteration - first_iter)

            # 影像渲染
            if (iteration - 1) == debug_from:
                pipe.debug = True

            # 进行影像渲染，渲染指定视点的影像
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
                                                                      render_pkg["visibility_filter"], render_pkg["radii"]

            # load_distribution中每一个值代表的是这一张影像上对每一个像素进行渲染时使用到的高斯求数量
            if not pipe.NotGetLoad:
                load_distribution = render_pkg["load_distribution"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            # PGSR Scale loss
            if not IsFinalOptimization and pipe.UseScaleLoss:
                scale_loss_weight = 100.0
                visibility_filter_bool = render_pkg["visibility_filter_bool"]
                if visibility_filter_bool.sum() > 0:
                    scale = gaussians.get_scaling[visibility_filter_bool]
                    sorted_scale, _ = torch.sort(scale, dim=-1)
                    min_scale_loss = sorted_scale[..., 0]
                    loss += scale_loss_weight * min_scale_loss.mean()

            # 以下是负载均衡的损失函数
            if not pipe.NotGetLoad:
                load_loss = torch.std(load_distribution)  # 计算标准差
                loss_adj = math.floor(math.log10(load_loss)) - math.floor(math.log10(loss))
                load_loss = load_loss / math.pow(10, loss_adj + 1.0)

            # 深度图
            if (not IsFinalOptimization) and pipe.UseDepthLoss:
                box_p = 128
                p_corr = 0.5
                lambda_SoftHard = 0.00001
                lambda_pearson = 0.05
                lambda_local_pearson = 0.15

                Depth_Map = render_pkg["depth_map"]
                Depth_Map_Hard = render_pkg["depth_map_hard"]
                Depth_Map_GT = scene.DepthMap[viewpoint_cam.image_name]

                soft_hard_loss = Soft_Hard_loss(Depth_Map.squeeze(0), Depth_Map_Hard.squeeze(0)) * lambda_SoftHard
                pearson_loss = pearson_depth_loss(Depth_Map.squeeze(0), Depth_Map_GT.squeeze(0)) * lambda_pearson
                # lp_loss = local_pearson_loss(Depth_Map.squeeze(0), Depth_Map_GT.squeeze(0), box_p, p_corr) * lambda_local_pearson

                # 深度损失
                # loss = loss + pearson_loss + lp_loss + soft_hard_loss
                loss = loss + pearson_loss + soft_hard_loss

            # 将原本的损失与负载均衡的损失加权平均
            if not pipe.NotGetLoad:
                loss = loss * (1 - opt.lambda_load) + opt.lambda_load * load_loss

            loss.backward()

            iter_end.record()  # 结束这一段的计时

            with torch.no_grad():
                if IsFinalOptimization and iteration % 100 == 0:
                    ImageOutputDir = os.path.join(args.Model_Path_Dir, 'TargetCam')
                    os.makedirs(ImageOutputDir, exist_ok=True)
                    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
                    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                    bg = torch.rand((3), device="cuda") if args.random_background else background
                    RenderTargetCam(TargetCam, gaussians, pipe, bg, iteration, ImageOutputDir)
                # 计算一个ema_loss_for_log特殊的损失
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

                # 每十次训练更新一次进度条
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # 稠密化
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration % args.MergeScene_Densification_Interval == 0:
                    # 只有当训练次数大于opt.opacity_reset_interval时才设置大小阈值为20
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    # 执行稠密化以及修剪的操作
                    if args.DifferentImagesDensify:
                        # 获取所有高斯球在每一张影像上的可见性
                        _, __, ___, ____, ALL_Image_Visibility_Filter = GetImageWeightsFromDynamicMonitoring(
                            ALL_viewpoint_stack, gaussians, pipe, bg, opt)

                        gaussians.DifferentImage_Densify_and_Prune(args.MergeScene_Densify_Grad_Threshold, opt.opacity_threshold,
                                                                   scene.cameras_extent
                                                                   , size_threshold, radii, args, ALL_viewpoint_stack,
                                                                   ImagesAlreadyBeTrainedIterations,
                                                                   ImageMatchMatrix, ALL_Image_Visibility_Filter)
                    else:
                        if (not IsFinalOptimization) or (IsFinalOptimization and AlreadyIterations + iteration - first_iter + 1 <= args.densify_until_iter):
                            gaussians.densify_and_prune(args.MergeScene_Densify_Grad_Threshold, opt.opacity_threshold, scene.cameras_extent,
                                                    size_threshold, radii)

                '''
                if iteration < opt.densify_until_iter:
                    # visibility_filter 是一个布尔掩码，标识哪些高斯分布在当前视图中可见。
                    # 通过 torch.max() 函数，代码将高斯的最大半径值（可能是图像空间中的半径）进行更新，保存当前可见高斯中半径的最大值。
                    # 这样做可能是为了记录哪些高斯对象在图像中投影占据了最大的空间。
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                         radii[visibility_filter])
    
                    # 这一步将视图空间中的点 viewspace_point_tensor 和 visibility_filter 传递给 gaussians.add_densification_stats() 方法，可能是为了收集一些统计数据
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
    
                    # 当训练次数处于[opt.densify_from_iter, opt.densify_until_iter]之间且为opt.densification_interval的整数倍
                    if (iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0) and args.Use_Global_Optimization_densify:
                        # 只有当训练次数大于opt.opacity_reset_interval时才设置大小阈值为20
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
    
                        # 执行稠密化以及修剪的操作
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
    
                    # 重新设置可见度
                    if (iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter))\
                            and args.Use_Global_Optimization_OpacityReset:
                        gaussians.reset_opacity()
    
    
                # 参数优化器梯度更新
                if iteration < opt.iterations:
                    # 根据反向传播计算的梯度更新模型参数
                    gaussians.optimizer.step()
    
                    # 这行代码用于在每次优化步骤之后清除所有计算得到的梯度，防止它们在下一次反向传播时被累积。
                    gaussians.optimizer.zero_grad(set_to_none=True)
                '''

                # 完成这一次训练后，让当前训练影像的已训练次数+1
                ImagesAlreadyBeTrainedIterations[viewpoint_cam.image_name] = ImagesAlreadyBeTrainedIterations[
                                                                                 viewpoint_cam.image_name] + 1

                # Optimizer step
                if True:
                    if opt.optimizer_type == "default":
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none=True)
                    elif opt.optimizer_type == "sparse_adam":
                        visible = radii > 0
                        gaussians.optimizer.step(visible, radii.shape[0])
                        gaussians.optimizer.zero_grad(set_to_none=True)

                # 完成最后一次训练后，进行模型评估并输出一次全部的训练影像的模型渲染结果
                if (iteration == FirstTrainingIterations + MergeTrainingIterations * NewImagesNum):
                    ThisDebugTimeStart = time.time()

                    EvaluateModel(iteration - first_iter + AlreadyIterations + 1, l1_loss, scene, render,
                                  (pipe, background), Diary, IsFinalOptimization, EvaluateDiary=EvaluateDiary)

                    if (not NoDebug) and RenderAllImages:
                        print("Output All Images Rendering Results!")

                        # 对所有训练影像都进行一次预测并保存结果
                        RenderAllImagesAndEvaluate(args, ALL_viewpoint_stack, gaussians, pipe, bg, opt, AlreadyIterations,
                                                   FirstTrainingIterations, first_iter, iteration)

                    ThisDebugTimeEnd = time.time()

                    DebugTime = DebugTime + ThisDebugTimeEnd - ThisDebugTimeStart

                    return scene, AlreadyIterations + MergeTrainingIterations * NewImagesNum, DebugTime

if __name__ == "__main__":
    # 训练开始计时
    Peak_Memory = 0

    TimeCost = {'PreProcess': 0,
                'FirstSceneTrain': 0,
                'SecondSceneIntial': 0,
                'MergeScene': 0,
                'SecondSceneTrain': 0,
                'FinalTrain': 0}

    TrainStartTime = time.time()

    '''以下是在构建参数配置器'''
    # 构建一个空的原始参数配置器
    parser = ArgumentParser(description="Training script parameters")

    # 构建一个用于存储与模型有关参数的参数配置器
    lp = ModelParams(parser)

    # 构建一个用于存储与模型优化有关参数的参数配置器
    op = OptimizationParams(parser)

    # 构建一个用于存储与流程处理有关参数的参数配置器
    pp = PipelineParams(parser)

    # 为原始参数配置器加入一些参数
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)  # 是否开启某些异常检测，默认为False
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")  # quiet默认为False，当设置为True时，则在训练过程中不会输出任何内容到日志文件
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    '''以上是在构建参数配置器'''

    # 输出训练日志
    os.makedirs(args.Model_Path_Dir, exist_ok=True)
    Diary = open(os.path.join(args.Model_Path_Dir, "Diary.txt"), "w")
    EvaluateDiary = open(os.path.join(args.Model_Path_Dir, "EvaluateDiary.txt"), "w")
    AllDebugTime = 0

    # 这个字典用于存储每一张影像已经被训练了多少次
    ImagesAlreadyBeTrainedIterations = {}

    # 打印参数配置器中的内容
    # print("ModelParams", args)

    # 输出存储训练后模型的存储位置
    print("Optimizing " + args.Source_Path_Dir)

    # 这个函数主要用于改变后续的输出流走向（更改print函数的输出位置或者输出方法），同时设置一些随机种子和使用的GPU等
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # 初始化服务器连接
    # network_gui.init(args.ip, args.port)

    # 设置是否启用求导时的异常检测
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # 读取所有的source_path
    IM = [int(name) for name in os.listdir(args.Source_Path_Dir) if os.path.isdir(os.path.join(args.Source_Path_Dir, name))]
    IM = sorted(IM)
    print(IM)
    source_path_list = [os.path.join(args.Source_Path_Dir, str(im)) for im in IM]
    model_path_list = [os.path.join(args.Model_Path_Dir, str(im)) for im in IM]
    Diary.write(f"source_path_Dir: {args.Source_Path_Dir}\nModel_Path_Dir: {args.Model_Path_Dir}\n")
    Diary.write(f"Progress: {IM}\n")
    Diary.write("\n")

    Diary.write(
        f"OpacityThreshold: {args.opacity_threshold}, InitialTrainingTimesSetZero: {args.InitialTrainingTimesSetZero}, UseDifferentImageLr: {args.DifferentImagesLr}, UseDepthLoss: {args.UseDepthLoss}, UseScaleLoss: {args.UseScaleLoss}, UseNormalLoss: {args.GetNormal}\n")
    print(
        f"OpacityThreshold: {args.opacity_threshold}, InitialTrainingTimesSetZero: {args.InitialTrainingTimesSetZero}, UseDifferentImageLr: {args.DifferentImagesLr}, UseDepthLoss: {args.UseDepthLoss}, UseScaleLoss: {args.UseScaleLoss}, UseNormalLoss: {args.GetNormal}")

    # 如果使用原本的学习率更新方法，则重新设置position_lr_max_step
    if not args.DifferentImagesLr:
        WholeIterations = args.IterationFirstScene + (IM[-1] - IM[0] - 1) * args.IterationPerMergeScene + \
                      int((IM[-1] - IM[
                          0] - 2) / args.GlobalOptimizationInterval) * args.GlobalOptimizationIteration + \
                      args.FinalOptimizationIterations
        args.position_lr_max_steps = WholeIterations

    TimeCost['PreProcess'] = TimeCost['PreProcess'] + time.time() - TrainStartTime

    # 进行第一个场景的模型训练
    if args.skip_FirstSceneTraining:
        print("\n------SKIP : First Scene Gaussian Model Optimization------\n")
        Diary.write("------SKIP : First Scene Gaussian Model Optimization------\n\n")
    else:
        Start_Time = time.time()

        print("\n------First Scene Gaussian Model Optimization------\n")
        Diary.write("------First Scene Gaussian Model Optimization------\n\n")

        args.model_path = model_path_list[0]
        args.source_path = source_path_list[0]
        Diary.write(f"First scene source_path: {args.source_path}\nFirst scene model_path: {args.model_path}\n")

        FirstScene, iteration, DebugTime = training_firstGaussian(lp.extract(args), op.extract(args), pp.extract(args),
                                                       args.test_iterations, args.start_checkpoint, args.debug_from,
                                                       args.IterationFirstScene, ImagesAlreadyBeTrainedIterations, Diary, args.NoDebug, EvaluateDiary=EvaluateDiary)
        AllDebugTime = DebugTime + AllDebugTime

        End_Time = time.time()
        TimeCost['FirstSceneTrain'] = TimeCost['FirstSceneTrain'] + End_Time - Start_Time - DebugTime
        print("First Scene Training Time Cost: {}s".format(End_Time - Start_Time - DebugTime))
        Diary.write("\nFirst Scene Training Time Cost: {}s\n".format(End_Time - Start_Time - DebugTime))

        CurrentMemory = get_gpu_memory_usage(args.gpu)
        if Peak_Memory < CurrentMemory:
            Peak_Memory = CurrentMemory

    # 获取TargetCam，用于模型训练结果的展示
    TargetCam = None
    if TargetCam == None:
        RenderCam = copy.deepcopy(FirstScene.getTrainCameras().copy()[0])

        RenderCam.R[0][0] = 0.99943693
        RenderCam.R[0][1] = 0.02387958
        RenderCam.R[0][2] = -0.02357095
        RenderCam.R[1][0] = -0.02322576
        RenderCam.R[1][1] = 0.99934829
        RenderCam.R[1][2] = 0.02763265
        RenderCam.R[2][0] = 0.02421545
        RenderCam.R[2][1] = -0.02706963
        RenderCam.R[2][2] = 0.99934021

        RenderCam.T[0] = -18.03593356  # +向左，-向右
        RenderCam.T[1] = 1.58552898  # +向上，-向下
        RenderCam.T[2] = 26.30952862  # +向后，-向前

        TargetCam = Camera(colmap_id=-2, R=RenderCam.R, T=RenderCam.T,
                           FoVx=RenderCam.FoVx, FoVy=RenderCam.FoVy,
                           image=RenderCam.original_image, gt_alpha_mask=None,
                           image_name='Target', uid=-2, data_device=args.data_device)

    # 循环进行后续的渐进式模型训练
    for ProgressiveTrainingTime in range(len(source_path_list) - 2):
        NewImagesNum = IM[ProgressiveTrainingTime + 1] - IM[ProgressiveTrainingTime]
        if (ProgressiveTrainingTime + 1) % args.RenderAllImagesInterval == 0:
            RenderAllImages = True
        else:
            RenderAllImages = False

        args.source_path = source_path_list[ProgressiveTrainingTime]
        args.model_path = model_path_list[ProgressiveTrainingTime]
        args.source_path_second = source_path_list[ProgressiveTrainingTime + 1]
        args.model_path_second = model_path_list[ProgressiveTrainingTime + 1]
        print("\n**************Model Optimized From {} to {}**************\n".format(args.source_path.split('/')[-1], args.source_path_second.split('/')[-1]))
        Diary.write("\n")
        Diary.write("\n**************Model Optimized From {} to {}**************\n".format(args.source_path.split('/')[-1], args.source_path_second.split('/')[-1]))

        # 进行第二个场景的模型初始化
        if args.skip_SecondSceneInitialization:
            print("\n------SKIP : Second Scene Gaussian Model Initialization------\n")
            Diary.write("\n------SKIP : Second Scene Gaussian Model Initialization------\n")
        else:
            Start_Time = time.time()

            print("\n------Second Scene Gaussian Model Initialization------\n")
            Diary.write("\n------Second Scene Gaussian Model Initialization------\n")
            # SecondScene = SceneInitial(lp.extract(args), op.extract(args), pp.extract(args))
            SecondScene = SceneInitialQuickly(lp.extract(args), op.extract(args), Diary)

            End_Time = time.time()

            TimeCost['SecondSceneIntial'] = TimeCost['SecondSceneIntial'] + End_Time - Start_Time
            print("Second Scene Initialization Time Cost: {}s".format(End_Time - Start_Time))
            Diary.write("\nSecond Scene Initialization Time Cost: {}s\n".format(End_Time - Start_Time))

            # 输出初始化后的场景
            if args.OutputModel and (ProgressiveTrainingTime + 1) % args.OutputModelInterval == 0:
                Start_Time = time.time()
                SecondScene.save(0)
                End_Time = time.time()
                AllDebugTime = AllDebugTime + End_Time - Start_Time

        # 场景合并
        if args.skip_FirstSceneTraining or args.skip_SecondSceneInitialization:
            print("\n------SKIP : Merge Scene------\n")
            Diary.write("\n------SKIP : Merge Scene------\n")
        else:
            Start_Time = time.time()

            print("\n------Merge Scene------\n")
            Diary.write("\n------Merge Scene------\n")

            with torch.no_grad():
                if not args.UseTrueDepthMap:
                    Newcams = FirstScene.AddNewImages(args, Diary, NewImagesNum)

                    mask = None
                    if args.GetMergeMaskQuickly:
                        mask = Get3DPointsMaskQuickly(FirstScene.getTrainCameras().copy(), SecondScene.gaussians, args, pp.extract(args), NewImagesNum)
                    else:
                        mask = Get3DPointsMask(FirstScene, SecondScene, args)

                    print(f"Newly Added Gaussians Num {mask.sum().item()}")
                    Diary.write(f"Newly Added Gaussians Num {mask.sum().item()}\n")

                    NewSceneParametersDict = GetMergeSceneGaussianParametersDict(FirstScene, SecondScene, mask, Diary)

                    FirstScene.gaussians.UpdateGaussianParameters(NewSceneParametersDict, args)
                    FirstScene.cameras_extent = FirstScene.gaussians.spatial_lr_scale

                    FirstScene.model_path = args.model_path_second
                else:
                    Newcams = FirstScene.AddNewImages(args, Diary, NewImagesNum)

                    FirstScene.gaussians.spatial_lr_scale = SecondScene.gaussians.spatial_lr_scale
                    FirstScene.cameras_extent = FirstScene.gaussians.spatial_lr_scale

                    ExpandGS_RGBD(FirstScene, Newcams[0])

                    FirstScene.model_path = args.model_path_second

            End_Time = time.time()

            TimeCost['MergeScene'] = TimeCost['MergeScene'] + End_Time - Start_Time
            print("Merge Scene Time Cost: {}s".format(End_Time - Start_Time))
            Diary.write("\nMerge Scene Time Cost: {}s\n".format(End_Time - Start_Time))

            # 输出合并后的场景
            if args.OutputModel and (ProgressiveTrainingTime + 1) % args.OutputModelInterval == 0:
                Start_Time = time.time()
                FirstScene.ProgressiveSave(args, iteration)
                End_Time = time.time()
                AllDebugTime = AllDebugTime + End_Time - Start_Time

        with torch.no_grad():
            Start_Time = time.time()
            RenderNewlyAddedImages(FirstScene.getTrainCameras().copy(), FirstScene.gaussians, lp.extract(args),
                                   op.extract(args), pp.extract(args),
                                   os.path.join(args.Model_Path_Dir, "ProgressTrainImages"), ProgressiveTrainingTime, 1)

            ImageOutputDir = os.path.join(args.Model_Path_Dir, 'TargetCam')
            os.makedirs(ImageOutputDir, exist_ok=True)
            bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            bg = torch.rand((3), device="cuda") if args.random_background else background
            RenderTargetCam(TargetCam, FirstScene.gaussians, pp.extract(args), bg, iteration, ImageOutputDir)

            End_Time = time.time()
            AllDebugTime = AllDebugTime + End_Time - Start_Time

        # 进行合并后的场景的模型训练
        if args.skip_MergeSceneTraining:
            print("\n------SKIP : Merge Scene Gaussian Model Optimization------\n")
            Diary.write("\n------SKIP : Merge Scene Gaussian Model Optimization------\n")
        else:
            Start_Time = time.time()

            print("\n------Merge Scene Gaussian Model Optimization------\n")
            Diary.write("\n------Merge Scene Gaussian Model Optimization------\n")
            if not args.OnlyUseLocalOptimization:
                FirstScene, iteration, DebugTime, ImageMatchMatrix = training_MergeScene(args, FirstScene, lp.extract(args),
                                                                                     op.extract(args), pp.extract(args),
                                                                                     args.debug_from,
                                                                                     args.OnlyTrainNewScene,
                                                                                     iteration,
                                                                                     args.IterationFirstScene,
                                                                                     args.IterationPerMergeScene,
                                                                                     ImagesAlreadyBeTrainedIterations,
                                                                                     Diary, args.NoDebug,
                                                                                     RenderAllImages, EvaluateDiary=EvaluateDiary)
            else:
                FirstScene, iteration, DebugTime, ImageMatchMatrix, Images_Weights, NewlyAddedImages = training_MergeScene(args, FirstScene, lp.extract(args),
                                                                                     op.extract(args), pp.extract(args),
                                                                                     args.debug_from,
                                                                                     args.OnlyTrainNewScene,
                                                                                     iteration,
                                                                                     args.IterationFirstScene,
                                                                                     args.IterationPerMergeScene,
                                                                                     ImagesAlreadyBeTrainedIterations,
                                                                                     Diary, args.NoDebug,
                                                                                     RenderAllImages, EvaluateDiary=EvaluateDiary)

            AllDebugTime = AllDebugTime + DebugTime

            End_Time = time.time()

            TimeCost['SecondSceneTrain'] = TimeCost['SecondSceneTrain'] + End_Time - Start_Time - DebugTime
            print("Merge Scene Training Time Cost: {}s".format(End_Time - Start_Time - DebugTime))
            Diary.write("\nMerge Scene Training Time Cost: {}s\n".format(End_Time - Start_Time - DebugTime))

            CurrentMemory = get_gpu_memory_usage(args.gpu)
            if Peak_Memory < CurrentMemory:
                Peak_Memory = CurrentMemory

        with torch.no_grad():
            Start_Time = time.time()
            RenderNewlyAddedImages(FirstScene.getTrainCameras().copy(), FirstScene.gaussians, lp.extract(args),
                                   op.extract(args), pp.extract(args),
                                   os.path.join(args.Model_Path_Dir, "ProgressTrainImages"), ProgressiveTrainingTime, 2)
            ImageOutputDir = os.path.join(args.Model_Path_Dir, 'TargetCam')
            os.makedirs(ImageOutputDir, exist_ok=True)
            bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            bg = torch.rand((3), device="cuda") if args.random_background else background
            RenderTargetCam(TargetCam, FirstScene.gaussians, pp.extract(args), bg, iteration, ImageOutputDir)
            End_Time = time.time()
            AllDebugTime = AllDebugTime + End_Time - Start_Time

        # 每训练若干张影像进行一次全局优化
        if ProgressiveTrainingTime % args.GlobalOptimizationInterval == 0 and ProgressiveTrainingTime != 0:
            if args.skip_GlobalOptimization:
                print("\n------SKIP : Global Optimization------\n")
                Diary.write("\n------SKIP : Global Optimization------\n")
            else:
                Start_Time = time.time()

                print("\n------Global Optimization------\n")
                Diary.write("\n------Global Optimization------\n")
                if not args.OnlyUseLocalOptimization:
                    FirstScene, iteration, DebugTime = ModelGlobalOptimization(args, FirstScene, lp.extract(args),
                                                                           op.extract(args),
                                                                           pp.extract(args), args.debug_from, iteration,
                                                                           args.IterationFirstScene,
                                                                           args.GlobalOptimizationIteration,
                                                                           ImagesAlreadyBeTrainedIterations,
                                                                           ImageMatchMatrix, Diary, args.NoDebug,
                                                                           RenderAllImages, NewImagesNum, EvaluateDiary=EvaluateDiary)
                else:
                    FirstScene, iteration, DebugTime = ModelGlobalOptimization(args, FirstScene, lp.extract(args),
                                                                               op.extract(args),
                                                                               pp.extract(args), args.debug_from,
                                                                               iteration,
                                                                               args.IterationFirstScene,
                                                                               args.GlobalOptimizationIteration,
                                                                               ImagesAlreadyBeTrainedIterations,
                                                                               ImageMatchMatrix, Diary, args.NoDebug,
                                                                               RenderAllImages, NewImagesNum, Images_Weights,
                                                                               NewlyAddedImages, EvaluateDiary=EvaluateDiary)

                AllDebugTime = AllDebugTime + DebugTime

                End_Time = time.time()

                TimeCost['SecondSceneTrain'] = TimeCost['SecondSceneTrain'] + End_Time - Start_Time - DebugTime
                print("Global Optimization Time Cost: {}s".format(End_Time - Start_Time - DebugTime))
                Diary.write("\nGlobal Optimization Time Cost: {}s\n".format(End_Time - Start_Time - DebugTime))

                CurrentMemory = get_gpu_memory_usage(args.gpu)
                if Peak_Memory < CurrentMemory:
                    Peak_Memory = CurrentMemory

                # 输出合并后的训练场景
                if args.OutputModel and (ProgressiveTrainingTime + 1) % args.OutputModelInterval == 0:
                    Start_Time = time.time()
                    FirstScene.ProgressiveSave(args, iteration)
                    End_Time = time.time()
                    AllDebugTime = AllDebugTime + End_Time - Start_Time

        # 输出当前每一张影像的已训练次数
        with torch.no_grad():
            Start_Time = time.time()
            Diary.write("\nImages have been Trained for:\n")
            for cam in FirstScene.getTrainCameras().copy():
                Diary.write(f"{cam.image_name}: {ImagesAlreadyBeTrainedIterations[cam.image_name]}\n")

            RenderNewlyAddedImages(FirstScene.getTrainCameras().copy(), FirstScene.gaussians, lp.extract(args),
                                   op.extract(args), pp.extract(args), os.path.join(args.Model_Path_Dir, "ProgressTrainImages"), ProgressiveTrainingTime, 3)
            ImageOutputDir = os.path.join(args.Model_Path_Dir, 'TargetCam')
            os.makedirs(ImageOutputDir, exist_ok=True)
            bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            bg = torch.rand((3), device="cuda") if args.random_background else background
            RenderTargetCam(TargetCam, FirstScene.gaussians, pp.extract(args), bg, iteration, ImageOutputDir)
            End_Time = time.time()
            AllDebugTime = AllDebugTime + End_Time - Start_Time

    FirstScene.model_path = args.model_path_second = model_path_list[-2]
    print("\n[ITER {}] Saving Checkpoint".format(iteration))
    torch.save((FirstScene.gaussians.capture(), iteration), FirstScene.model_path + "/chkpnt" + str(iteration) + ".pth")
    print("\n[ITER {}] Saving Gaussians".format(iteration))
    FirstScene.save(iteration)

    # 最后进行一次整体模型的调整
    Start_Time = time.time()

    args.source_path = source_path_list[-2]
    args.model_path = model_path_list[-2]
    args.source_path_second = source_path_list[-1]
    args.model_path_second = model_path_list[-1]

    print("********************Final Optimization********************")
    Diary.write("\n")
    Diary.write("\n********************Final Optimization********************\n")

    # 初始化
    SecondScene = SceneInitialQuickly(lp.extract(args), op.extract(args), Diary)

    # 更新所有训练影像的位姿信息
    FirstScene.UpdateTrainingImagesPos(args)

    # 模型训练
    FirstScene, iteration, DebugTime = ModelGlobalOptimization(args, FirstScene, lp.extract(args), op.extract(args),
                                                               pp.extract(args), args.debug_from, iteration,
                                                               args.IterationFirstScene,
                                                               args.FinalOptimizationIterations,
                                                               ImagesAlreadyBeTrainedIterations,
                                                               ImageMatchMatrix, Diary, args.NoDebug, True, 1, IsFinalOptimization=True, EvaluateDiary=EvaluateDiary)

    AllDebugTime = AllDebugTime + DebugTime

    End_Time = time.time()

    TimeCost['FinalTrain'] = TimeCost['FinalTrain'] + End_Time - Start_Time - DebugTime
    print("Global Optimization Time Cost: {}s".format(End_Time - Start_Time - DebugTime))
    Diary.write("\nGlobal Optimization Time Cost: {}s\n".format(End_Time - Start_Time - DebugTime))

    TrainEndTime = time.time()

    CurrentMemory = get_gpu_memory_usage(args.gpu)
    if Peak_Memory < CurrentMemory:
        Peak_Memory = CurrentMemory

    # 输出当前每一张影像的已训练次数
    Diary.write("\nImages have been Trained for:\n")
    for cam in FirstScene.getTrainCameras().copy():
        Diary.write(f"{cam.image_name}: {ImagesAlreadyBeTrainedIterations[cam.image_name]}\n")

    # 训练结束，输出结果
    FirstScene.model_path = args.model_path_second = model_path_list[-1]
    print("\n[ITER {}] Saving Checkpoint".format(iteration))
    torch.save((FirstScene.gaussians.capture(), iteration), FirstScene.model_path + "/chkpnt" + str(iteration) + ".pth")
    print("\n[ITER {}] Saving Gaussians".format(iteration))
    FirstScene.save(iteration)

    # 训练完成的标志
    Diary.write("\nTraining complete. Training Time Cost: {}s".format(TrainEndTime - TrainStartTime - AllDebugTime))
    Diary.write("\nTraining complete. Training Time Cost: {}s".format(TimeCost['FirstSceneTrain'] + TimeCost['SecondSceneTrain'] + TimeCost['FinalTrain']))
    Diary.write(f"\nFirstSceneTrain: {TimeCost['FirstSceneTrain']}s, SecondSceneIntial: {TimeCost['SecondSceneIntial']}s, MergeScene: {TimeCost['MergeScene']}s, SecondSceneTrain: {TimeCost['SecondSceneTrain']}s, FinalTrain: {TimeCost['FinalTrain']}s")
    Diary.write(f"\nMemory: {Peak_Memory}MB")
    Diary.close()
    EvaluateDiary.close()
    print("\nTraining complete. Training Time Cost: {}s".format(TrainEndTime - TrainStartTime - AllDebugTime))
    print(TimeCost)

    # 进行一系列必要的数据统计或者整理
    # 绘制每一项张影像随着模型训练的PSNR的变化结果，折线图
    if not args.NoDebug:
        DrawImagesPSNR(os.path.join(args.Model_Path_Dir, "OutputImages"), args.IterationFirstScene, args.IterationPerMergeScene)

    # 简化输出日志
    GetSimpleDiary(args.Model_Path_Dir)

    # 绘制模型在训练过程中的PSNR变化
    DrawModelPSNR(args.Model_Path_Dir, args.IterationFirstScene)

    # 生成Demo
    with torch.no_grad():
        GetTrainingDemo(os.path.join(args.Model_Path_Dir, "ProgressTrainImages"), os.path.join(args.Model_Path_Dir, "ProgressTrainImages", "Demo.mp4"))
        GetResultDemo(os.path.join(args.Model_Path_Dir, "OutputImages", "Final"), FirstScene, args)