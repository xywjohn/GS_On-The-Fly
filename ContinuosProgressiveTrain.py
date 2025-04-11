import os
import random
import torch
from random import randint
from utils.loss_utils import l1_loss
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from fused_ssim import fused_ssim
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from PIL import Image
import torchvision.transforms as transforms
from scipy.spatial import cKDTree
import math
import time
from scene.cameras import Camera
from utils.camera_utils import camera_to_JSON
import json
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.general_utils import PILtoTorch
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def prepare_output_and_logger(args, Create_for_new=False):
    # 创建对应的输出文件夹
    if not Create_for_new:
        m_path = args.model_path
    else:
        m_path = args.model_path_second

    print("Output folder: {}".format(m_path))
    os.makedirs(m_path, exist_ok=True)

    # 在文件夹中创建并打开一个二进制文件cfg_args，并在里面输出参数配置其的所有内容
    with open(os.path.join(m_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 创建一个Tensorboard writer对象
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(m_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def GetArgs():
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

    # 输出存储训练后模型的存储位置
    print("Optimizing " + args.Source_Path_Dir)

    return args, lp, op, pp

def TrainingPreparation(args):
    TimeCost = {'PreProcess': 0.0,
                'FirstSceneTrain': 0.0,
                'SecondSceneIntial': 0.0,
                'MergeScene': 0.0,
                'SecondSceneTrain': 0.0,
                'FinalTrain': 0.0,
                'DebugTime': 0.0}

    # 输出训练日志
    os.makedirs(args.Model_Path_Dir, exist_ok=True)
    Diary = open(os.path.join(args.Model_Path_Dir, "Diary.txt"), "w")
    EvaluateDiary = open(os.path.join(args.Model_Path_Dir, "EvaluateDiary.txt"), "w")

    # 这个字典用于存储每一张影像已经被训练了多少次
    ImagesAlreadyBeTrainedIterations = {}

    # 一些输出上的设置
    safe_state(args.quiet)
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

    Diary.write(f"OpacityThreshold: {args.opacity_threshold}, InitialTrainingTimesSetZero: {args.InitialTrainingTimesSetZero}, UseDifferentImageLr: {args.DifferentImagesLr}, UseDepthLoss: {args.UseDepthLoss}, UseScaleLoss: {args.UseScaleLoss}, UseNormalLoss: {args.GetNormal}\n")
    print(f"OpacityThreshold: {args.opacity_threshold}, InitialTrainingTimesSetZero: {args.InitialTrainingTimesSetZero}, UseDifferentImageLr: {args.DifferentImagesLr}, UseDepthLoss: {args.UseDepthLoss}, UseScaleLoss: {args.UseScaleLoss}, UseNormalLoss: {args.GetNormal}")

    # 如果使用原本的学习率更新方法，则重新设置position_lr_max_step
    if not args.DifferentImagesLr:
        WholeIterations = args.IterationFirstScene + (IM[-1] - IM[0] - 1) * args.IterationPerMergeScene + \
                          int((IM[-1] - IM[0] - 2) / args.GlobalOptimizationInterval) * args.GlobalOptimizationIteration + \
                          args.FinalOptimizationIterations
        args.position_lr_max_steps = WholeIterations

    return TimeCost, Diary, EvaluateDiary, ImagesAlreadyBeTrainedIterations, source_path_list, model_path_list, IM

def RecordTime(TimeCost, TimeClass, TimeCosumption):
    if TimeClass in TimeCost.keys():
        TimeCost[TimeClass] = TimeCost[TimeClass] + TimeCosumption
    else:
        print(f"Only Record: {list(TimeCost.keys())}")

    return TimeCost

class Gaussian_On_The_Fly_Splatting:
    # 构造函数
    def __init__(self):
        # 用于统计峰值内存以及训练时间
        self.Peak_Memory = 0
        self.TrainStartTime = time.time()

        # 获取参数配置器
        self.args, self.lp, self.op, self.pp = GetArgs()

        # 训练开始前的准备工作
        self.TimeCost, self.Diary, self.EvaluateDiary, self.ImagesAlreadyBeTrainedIterations, self.source_path_list, self.model_path_list, self.IM = TrainingPreparation(self.args)

        # 记录完成准备工作所消耗的时间
        self.TimeCost = RecordTime(self.TimeCost, 'PreProcess', time.time() - self.TrainStartTime)

        # 记录模型的已训练次数
        self.AlreadyTrainingIterations = 0

        # 在没有开始渐进式训练之前，影像匹配矩阵和影像权重不存在
        self.ImageMatchingMatrix = None
        self.Images_Weights = None

        # 在没有开始渐进式训练之前，新增影像数量不存在
        self.NewImagesNum = -1

        # 该字典用于存储每一张影像对应的高斯球
        self.Image_Visibility = {}
        self.MaxWeightsImages = []

        # 初始化一个初始模型
        self.Initialize_Gaussians()

        # 进行初始模型的训练
        self.Train_Gaussians(self.args.IterationFirstScene, "Initialization")

        # 如果需要，输出一次高斯模型
        self.GS_Save_Times = 1
        self.GS_Save(UseSecondPath=False)

    # 初始化一个场景
    def Initialize_Gaussians(self):
        self.resolution_scales = [1.0]

        # 设置初始场景的文件路径
        self.args.model_path = self.model_path_list[0]
        self.args.source_path = self.source_path_list[0]
        self.Diary.write(f"First scene source_path: {self.args.source_path}\nFirst scene model_path: {self.args.model_path}\n")

        # 主要进行一系列的模型以及其它文件的输出准备（例如构建输出文件夹等），并返回一个Tensorboard writer对象
        self.tb_writer = prepare_output_and_logger(self.args)

        # 初始化3D Gaussian Splatting模型，主要是一些模型属性的初始化和神经网络中一些激活函数的初始化
        self.gaussians = GaussianModel(self.args.sh_degree)

        # 初始化一个场景
        self.scene = Scene(self.args, self.gaussians, load_iteration=None, shuffle=True, resolution_scales=[1.0])

        # 进行一些训练上的初始化设置
        self.gaussians.training_setup(self.args)

        # 初始化背景颜色
        self.bg_color = [1, 1, 1] if self.args.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")
        self.bg = torch.rand((3), device="cuda") if self.args.random_background else self.background

        # 初始化log损失累计
        self.ema_loss_for_log = 0.0

    # 在渐进式训练过程中，将根据特殊规则选取用于训练的影像
    def GetTraining_Viewpoints(self):
        # 如果读入的影像数量少于等于20，则直接返回所有影像
        if len(self.Images_Weights) <= 20:
            return self.scene.getTrainCameras().copy()
        # 如果读入的影像数量大于20
        else:
            ALL_viewpoint_stack = self.scene.getTrainCameras().copy()

            # 找到10张权值最大的影像
            MaxWeights = []
            MaxWeightImages = []
            MaxWeightIndexes = []
            for i in range(len(self.Images_Weights)):
                if len(MaxWeightImages) < 10:
                    MaxWeights.append(self.Images_Weights[i])
                    MaxWeightImages.append(ALL_viewpoint_stack[i])
                    MaxWeightIndexes.append(i)
                else:
                    if min(MaxWeights) < self.Images_Weights[i]:
                        cidx = MaxWeights.index(min(MaxWeights))
                        MaxWeights[cidx] = self.Images_Weights[i]
                        MaxWeightImages[cidx] = ALL_viewpoint_stack[i]
                        MaxWeightIndexes[cidx] = i

            self.MaxWeightsImages = MaxWeightImages.copy()

            # 从剩余的影像中随机挑出10张，且尽量使这10张影像均匀分布于整个场景，暂时使用随机选取的策略
            Allindex = [i for i in range(len(self.Images_Weights))]
            diff_index = [i for i in Allindex if i not in MaxWeightIndexes]
            SampleIndex = random.sample(diff_index, 10)
            for i in SampleIndex:
                MaxWeights.append(self.Images_Weights[i])
                MaxWeightImages.append(ALL_viewpoint_stack[i])
                MaxWeightIndexes.append(i)

            # 打印结果
            self.Diary.write("Chosen Images: [")
            for i in range(len(ALL_viewpoint_stack)):
                if i in MaxWeightIndexes:
                    self.Diary.write(f"{ALL_viewpoint_stack[i].image_name}, ")
            self.Diary.write("]\n")

            # 将本次所有被选中的影像加入到本次的训练影像栈中
            viewpoint_stack = []
            for i in range(len(MaxWeights)):
                viewpoint_stack.append(MaxWeightImages[i])

            return viewpoint_stack

    # 3DGS模型训练核心函数
    def Train_Gaussians(self, iteration, Training_Type):
        self.Diary.write('\n')

        Start_From_Its = self.AlreadyTrainingIterations
        if Training_Type == "Initialization":
            print("\n------First Scene Gaussian Model Optimization------\n")
            self.Diary.write("------First Scene Gaussian Model Optimization------\n\n")
        elif Training_Type == "On_The_Fly":
            print("\n------Merge Scene Gaussian Model Optimization------\n")
            self.Diary.write("\n------Merge Scene Gaussian Model Optimization------\n")

        # 本次训练开始计时
        Train_Gaussians_Start = time.time()

        # 初始化影像栈以及进度条
        viewpoint_stack = None
        EvaluateRender = False
        progress_bar = tqdm(range(0, Start_From_Its + iteration), desc=f"{Training_Type} Training progress", initial=Start_From_Its)

        for sub_iteration in range(iteration):
            # 更新模型已训练次数
            self.AlreadyTrainingIterations += 1

            # 当摄影机视点栈堆为空时：
            if not viewpoint_stack:
                if Training_Type != "On_The_Fly":
                    # 将Scene类中所有的self.train_cameras中指定缩放比例的训练影像及其相关信息存入到摄影机视点栈堆中
                    viewpoint_stack = self.scene.getTrainCameras().copy()

                    # 如果是模型初始化训练，那么需要先将初始模型就已存在的影像的已训练次数置为0
                    if Training_Type == "Initialization" and len(list(self.ImagesAlreadyBeTrainedIterations.keys())) == 0:
                        for _ in range(len(viewpoint_stack)):
                            self.ImagesAlreadyBeTrainedIterations[viewpoint_stack[_].image_name] = 0
                else:
                    # 在渐进式训练过程中，将根据特殊规则选取用于训练的影像
                    viewpoint_stack = self.GetTraining_Viewpoints()

            # 随机从摄影机视点栈堆中任意取出一个影像以及其视点信息
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

            # 学习率下降
            if Training_Type != "On_The_Fly":
                self.gaussians.update_learning_rate(self.AlreadyTrainingIterations)
            else:
                self.gaussians.On_The_Fly_Update_Lr(self.scene.getTrainCameras().copy(),
                                                    self.ImagesAlreadyBeTrainedIterations,
                                                    self.args, viewpoint_cam,
                                                    self.ImageMatchingMatrix)

            # 将球谐函数的阶数提高一阶，最多至3阶
            if self.AlreadyTrainingIterations % 1000 == 0:
                self.gaussians.oneupSHdegree()

            # 计算损失
            if Training_Type != "On_The_Fly":
                # 进行影像渲染，渲染指定视点的影像
                render_pkg = render(viewpoint_cam, self.gaussians, self.args, self.bg)
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    load_distribution
                 ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["load_distribution"]
                )

                self.Image_Visibility[viewpoint_cam.image_name] = visibility_filter

                # 计算损失
                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                loss = (1.0 - self.args.lambda_dssim) * Ll1 + self.args.lambda_dssim * (1.0 - ssim_value)

                load_loss = torch.std(load_distribution)  # 计算标准差
                loss_adj = math.floor(math.log10(load_loss)) - math.floor(math.log10(loss))
                load_loss = load_loss / math.pow(10, loss_adj + 1.0)

                loss = loss * (1 - self.args.lambda_load) + self.args.lambda_load * load_loss

                # 完成这一次训练后，让当前训练影像的已训练次数+1
                self.ImagesAlreadyBeTrainedIterations[viewpoint_cam.image_name] = self.ImagesAlreadyBeTrainedIterations[viewpoint_cam.image_name] + 1
            # 目前暂时采用与其他情况相同的训练策略
            else:
                # 进行影像渲染，渲染指定视点的影像
                render_pkg = render(viewpoint_cam, self.gaussians, self.args, self.bg)
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    load_distribution
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["load_distribution"]
                )

                self.Image_Visibility[viewpoint_cam.image_name] = visibility_filter

                # 计算损失
                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                loss = (1.0 - self.args.lambda_dssim) * Ll1 + self.args.lambda_dssim * (1.0 - ssim_value)

                load_loss = torch.std(load_distribution)  # 计算标准差
                loss_adj = math.floor(math.log10(load_loss)) - math.floor(math.log10(loss))
                load_loss = load_loss / math.pow(10, loss_adj + 1.0)

                loss = loss * (1 - self.args.lambda_load) + self.args.lambda_load * load_loss

                # 完成这一次训练后，让当前训练影像的已训练次数+1
                self.ImagesAlreadyBeTrainedIterations[viewpoint_cam.image_name] = self.ImagesAlreadyBeTrainedIterations[viewpoint_cam.image_name] + 1

            # 反向传播
            loss.backward()

            with torch.no_grad():
                # 计算log损失累计
                self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log

                # 每十次训练更新一次进度条
                progress_bar.update(1)
                progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                if sub_iteration + 1 == iteration:
                    EvaluateRender = True
                    progress_bar.close()

                # 如果出于模型初始化训练阶段
                GaussianPruned = False
                if Training_Type == "Initialization" and self.AlreadyTrainingIterations < self.args.densify_until_iter:
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    # 稠密化
                    if self.AlreadyTrainingIterations > self.args.densify_from_iter and self.AlreadyTrainingIterations % self.args.densification_interval == 0:
                        size_threshold = 20 if self.AlreadyTrainingIterations > self.args.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(self.args.densify_grad_threshold, self.args.opacity_threshold, self.scene.cameras_extent, size_threshold, radii)
                        GaussianPruned = True
                elif Training_Type == "On_The_Fly":
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    # 稠密化
                    if self.AlreadyTrainingIterations % self.args.MergeScene_Densification_Interval == 0:
                        size_threshold = 20 if self.AlreadyTrainingIterations > self.args.opacity_reset_interval else None
                        # self.gaussians.densify_and_prune(self.args.densify_grad_threshold, self.args.opacity_threshold, self.scene.cameras_extent, size_threshold, radii)
                        self.gaussians.On_The_Fly_Densify_and_Prune(self.args.densify_grad_threshold,
                                                                    self.args.opacity_threshold,
                                                                    self.scene.cameras_extent,
                                                                    size_threshold,
                                                                    radii,
                                                                    self.MaxWeightsImages,
                                                                    self.Image_Visibility,
                                                                    self.ImagesAlreadyBeTrainedIterations,
                                                                    self.scene.getTrainCameras().copy(),
                                                                    self.ImageMatchingMatrix)
                        GaussianPruned = True
                elif Training_Type == "Final_Refinement" and self.AlreadyTrainingIterations <= Start_From_Its + iteration / 2:
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    # 稠密化
                    if self.AlreadyTrainingIterations % self.args.densification_interval == 0:
                        size_threshold = 20 if self.AlreadyTrainingIterations > self.args.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(self.args.densify_grad_threshold, self.args.opacity_threshold, self.scene.cameras_extent, size_threshold, radii)
                        GaussianPruned = True

                if GaussianPruned:
                    self.Image_Visibility = {}

                # 重新设置可见度
                if self.AlreadyTrainingIterations % self.args.opacity_reset_interval == 0:
                    if (Training_Type != "Final_Refinement"
                    ) or (
                            Training_Type == "Final_Refinement" and
                            self.args.FinalOptimizationIterations - (self.AlreadyTrainingIterations - Start_From_Its) > self.args.opacity_reset_interval and
                            self.AlreadyTrainingIterations <= Start_From_Its + iteration / 2):
                        print(f"[{self.AlreadyTrainingIterations} its] => Reset Opacity!!")
                        self.gaussians.reset_opacity()

                # 优化器参数更新
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

                # 每训练一定次数输出一次模型评估结果
                self.Evaluate(Training_Type, EvaluateRender)
                EvaluateRender = False

        # 本次训练结束计时
        Train_Gaussians_End = time.time()

        # 记录用时
        if Training_Type == "Initialization":
            self.TimeCost = RecordTime(self.TimeCost, "FirstSceneTrain", Train_Gaussians_End - Train_Gaussians_Start - self.TimeCost["DebugTime"])
            print("First Scene Training Time Cost: {}s".format(self.TimeCost["FirstSceneTrain"]))
            self.Diary.write("\nFirst Scene Training Time Cost: {}s\n".format(self.TimeCost["FirstSceneTrain"]))

    # 对当前高斯模型进行一次评估
    def Evaluate(self, Training_Type, EvaluateRender):
        DoEvaluate = False
        if Training_Type == "Initialization" and (self.AlreadyTrainingIterations % self.args.InitialTrainingEvaluateInterval == 0 or EvaluateRender):
            DoEvaluate = True
        elif (Training_Type == "On_The_Fly" or Training_Type == "Final_Refinement") and EvaluateRender:
            DoEvaluate = True

        if DoEvaluate:
            ThisDebugTimeStart = time.time()
            self.EvaluateModel(l1_loss, render, (self.pp.extract(self.args), self.background))
            ThisDebugTimeEnd = time.time()
            self.TimeCost = RecordTime(self.TimeCost, "DebugTime", ThisDebugTimeEnd - ThisDebugTimeStart)

    # 打印模型的评价结果
    def EvaluateModel(self, l1_loss, renderFunc, renderArgs):
        # Report test and samples of training set
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': self.scene.getTestCameras()}, {'name': 'train', 'cameras': self.scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0

                PSNRs = []
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, self.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    l1_test += l1_loss(image, gt_image).mean().double()

                    PSNR = psnr(image, gt_image).mean().double()
                    psnr_test += PSNR
                    PSNRs.append(PSNR)

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} Gaussians {} PSNR {}".format(self.AlreadyTrainingIterations, config['name'], l1_test, self.gaussians.get_xyz.shape[0], psnr_test))
                self.Diary.write("[ITER {}] Evaluating {}: L1 {} Gaussians {} PSNR {}\n".format(self.AlreadyTrainingIterations, config['name'], l1_test, self.gaussians.get_xyz.shape[0], psnr_test))

                if self.EvaluateDiary is not None and len(PSNRs) != 0:
                    self.EvaluateDiary.write("-----------------------------------------------------------\n")
                    self.EvaluateDiary.write(f"Iterations: {self.AlreadyTrainingIterations}\n")
                    for i in range(len(PSNRs)):
                        self.EvaluateDiary.write(f"{config['cameras'][i].image_name}: {PSNRs[i]}\n")

    # 为所有影像进行一次渲染并输出一些评估结果
    def Render_Evaluate_All_Images(self):
        with torch.no_grad():
            ALL_viewpoint_stack = self.scene.getTrainCameras().copy()
            All_Predicted_Images = []
            ALL_Image_Visibility_Filter = []
            ALL_Image_PSNR = []
            progress_bar = tqdm(range(len(ALL_viewpoint_stack)), desc="Rendering progress")
            for i in range(len(ALL_viewpoint_stack)):
                ImageOutputDir = os.path.join(self.args.Model_Path_Dir, "OutputImages", ALL_viewpoint_stack[i].image_name)
                PSNR_File_Path = os.path.join(ImageOutputDir, "PSNR.txt")
                os.makedirs(ImageOutputDir, exist_ok=True)
                PSNR_File = open(PSNR_File_Path, 'w')

                progress_bar.update(1)

                viewpoint_cam = ALL_viewpoint_stack[i]
                render_pkg = render(viewpoint_cam, self.gaussians, self.args, self.bg)
                image, visibility_filter = render_pkg["render"], render_pkg["visibility_filter"]

                All_Predicted_Images.append(image)
                ALL_Image_Visibility_Filter.append(visibility_filter)

                ImageOutput = (image - image.min()) / (image.max() - image.min())
                ImageOutput = ImageOutput.permute(1, 2, 0).cpu().numpy()
                ImageOutput = (ImageOutput * 255).astype(np.uint8)
                ImageOutput = Image.fromarray(ImageOutput)
                ImageOutput.save(os.path.join(ImageOutputDir, f"PredictionImages{self.AlreadyTrainingIterations}.jpg"))

                gt_image = viewpoint_cam.original_image.cuda()
                PSNR = psnr(image, gt_image)
                ALL_Image_PSNR.append(PSNR.cpu().sum().item() / 3)

                PSNR_File.write(str(self.AlreadyTrainingIterations) + f": {PSNR}, Visible_Gaussians_Num: {visibility_filter.shape[0]}\n")
                PSNR_File.close()

    # 根据新的稀疏点云来扩张高斯点云
    def ExpandingGS_From_SparsePCD(self, distance_threshold=1.0, distance_buffer=1.5):
        if os.path.exists(os.path.join(self.args.source_path_second, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](self.args.source_path_second, self.args.images, self.args.eval)
        else:
            assert False, "Could not recognize scene type!"

        # 将ply_path对应的.ply文件复制到self.model_path文件夹中，并取名为input.ply
        prepare_output_and_logger(self.args, True)
        with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.args.model_path_second, "input.ply"), 'wb') as dest_file:
            dest_file.write(src_file.read())

        # 初始化两个用于存储相机信息的数组：
        json_cams = []
        camlist = []

        # 如果测试影像或者训练影像存在，则将他们都加入到camlist中去
        if scene_info.test_cameras:
            camlist.extend(scene_info.test_cameras)
        if scene_info.train_cameras:
            camlist.extend(scene_info.train_cameras)

        # 将camlist中的影像信息转换为JSON的模式，并存储在json_cams中
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))

        # 将json_cams中的数据存在cameras.json中
        with open(os.path.join(self.args.model_path_second, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

        # 将包裹所有相机的球的半径赋值给self.cameras_extent（相机分布范围）
        self.scene.cameras_extent = scene_info.nerf_normalization["radius"]

        # 将新加入的影像读入到系统之中
        self.AddNewImages(scene_info)

        # 确定稀疏点云中哪一些点是新增加的
        points1 = self.scene.basic_pcd.points
        points2 = scene_info.point_cloud.points

        # 使用 cKDTree 加速查找重叠点
        tree = cKDTree(points1)
        distances, indices = tree.query(points2, distance_upper_bound=distance_threshold)

        # 过滤掉与 vertices1 中距离小于阈值的点
        mask = distances >= distance_threshold  # True 表示保留该点
        points3 = points2[mask]

        # 使用 cKDTree 加速查找重叠点
        tree = cKDTree(points3)
        distances, indices = tree.query(points2, distance_upper_bound=distance_threshold * distance_buffer)

        # 过滤掉与 filtered_vertices2 中距离大于阈值的点
        mask = distances <= distance_threshold * distance_buffer  # True 表示保留该点
        print(f"Add new Gaussians {mask.sum().item()}")

        # 根据扩张后的稀疏点云来更新高斯点云
        self.gaussians.expand_from_pcd(scene_info.point_cloud, self.scene.cameras_extent, mask)

        # 记录这一次的稀疏点云
        self.scene.basic_pcd = scene_info.point_cloud

        # 更新模型输出位置
        self.scene.model_path = self.args.model_path_second

    # 读入新的影像数据并更新所有已有影像的位姿信息
    def AddNewImages(self, scene_info):
        # 为self.train_cameras进行更新，将新加入的影像的相关信息加入到其中
        # 先找到新加入的影像
        OLD_TrainCamInfos = self.scene.scene_info_traincam
        NEW_TrainCamInfos = scene_info.train_cameras

        self.Old_Cam_Name = [cam.image_name for cam in OLD_TrainCamInfos]
        self.New_Cam_Name = [cam.image_name for cam in NEW_TrainCamInfos]

        # 找出所有新的影像
        CurrentNewImagesNum = 0
        NewImageNames = "["
        NewCamInfos = []
        for i in range(len(self.New_Cam_Name)):
            if (self.New_Cam_Name[i] not in self.Old_Cam_Name):
                New_CamInfo = NEW_TrainCamInfos[i]
                NewImageNames = NewImageNames + New_CamInfo.image_name + ", "
                NewCamInfos.append(New_CamInfo)
                CurrentNewImagesNum = CurrentNewImagesNum + 1

            if CurrentNewImagesNum == self.NewImagesNum:
                break

        NewImageNames = NewImageNames + "]"
        print(f"Newly Added Images Name: {NewImageNames}")
        self.Diary.write(f"Newly Added Images Name: {NewImageNames}\n")

        # 在每一个resolution_scale下：
        NewCams = []
        for resolution_scale in self.resolution_scales:
            # 更新self.train_cameras中[resolution_scale]每一个影像的位姿信息
            for i in range(len(self.scene.train_cameras[resolution_scale])):
                for cam in NEW_TrainCamInfos:
                    if (cam.image_name == self.scene.train_cameras[resolution_scale][i].image_name):
                        self.scene.train_cameras[resolution_scale][i].R = cam.R
                        self.scene.train_cameras[resolution_scale][i].T = cam.T
                        break

            for cam_info in NewCamInfos:
                orig_w, orig_h = cam_info.image.size
                if self.args.resolution in [1, 2, 4, 8]:
                    resolution = round(orig_w / (resolution_scale * self.args.resolution)), round(
                        orig_h / (resolution_scale * self.args.resolution))
                else:  # should be a type that converts to float
                    if self.args.resolution == -1:
                        if orig_w > 1600:
                            global WARNED
                            if not WARNED:
                                print(
                                    "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                                    "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                                WARNED = True
                            global_down = orig_w / 1600
                        else:
                            global_down = 1
                    else:
                        global_down = orig_w / self.args.resolution

                    scale = float(global_down) * float(resolution_scale)
                    resolution = (int(orig_w / scale), int(orig_h / scale))

                resized_image_rgb = PILtoTorch(cam_info.image, resolution)

                gt_image = resized_image_rgb[:3, ...]
                loaded_mask = None

                if resized_image_rgb.shape[1] == 4:
                    loaded_mask = resized_image_rgb[3:4, ...]

                NewCam = Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                                FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                                image=gt_image, gt_alpha_mask=loaded_mask,
                                image_name=cam_info.image_name, uid=len(self.scene.train_cameras[resolution_scale]),
                                data_device=self.args.data_device)

                self.scene.train_cameras[resolution_scale].append(NewCam)
                NewCams.append(NewCam)

                # 将新加入影像的已训练次数设置为0
                self.ImagesAlreadyBeTrainedIterations[NewCam.image_name] = 0

            self.scene.scene_info_traincam = scene_info.train_cameras

            return NewCams

    # 读入影像匹配矩阵
    def GetImageMatchingMatrix(self):
        MatrixPath = self.args.source_path_second + r"/sparse/0/imageMatchMatrix.png"
        if not os.path.exists(MatrixPath):
            MatrixPath = self.args.source_path_second + r"/sparse/0/imageMatchMatrix.txt"
        ImagesNamePath = self.args.source_path_second + r"/sparse/0/imagesNames.txt"

        self.Diary.write(f"Get Images_Match_Matrix From {MatrixPath}\n")
        self.Diary.write(f"Get Images_Name_Path From {ImagesNamePath}\n")
        print(f"Get Images_Match_Matrix From {MatrixPath}")
        print(f"Get Images_Name_Path From {ImagesNamePath}")

        # 获取现在scene.train_cameras[resolution_scale]中影像的名称
        viewpoint_stack = self.scene.getTrainCameras().copy()

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
        Images = ImagesNameFile.readline().split(",")
        for i in range(len(Images)):
            ImagesNames.append(Images[i].split("\n")[0])

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
    def GetImagesWeightsFromMatrix(self):
        # 找出新加入的影像的名字和原本就有的影像名字，同时确定新加入的影像在ALL_viewpoint_stack中对应的是哪几个Camera类，并返回索引值
        NewlyAddedImages = []
        NewlyAddedImagesIndex = []
        ALL_viewpoint_stack = self.scene.getTrainCameras().copy()
        for new_img_name in self.New_Cam_Name:
            if new_img_name not in self.Old_Cam_Name:
                NewlyAddedImages.append(new_img_name.split('.')[0])
                for i in range(len(ALL_viewpoint_stack)):
                    if ALL_viewpoint_stack[i].image_name == NewlyAddedImages[-1]:
                        NewlyAddedImagesIndex.append(i)
                        break
        print(NewlyAddedImages)
        print(NewlyAddedImagesIndex)

        # 为每一张影像赋予一个权值，代表后续影像训练中这张影像的重视程度，权值取值为[0, 1]
        # 新加入的影像权值直接赋值为1，其余影像根据重叠度决定权值
        Images_Weights = []
        Calculated = []  # 这个列表用于存储哪一些影像的权值不是0，不是0的标记为True
        ZeroWeightExist = False
        for i in range(len(ALL_viewpoint_stack)):
            if ALL_viewpoint_stack[i].image_name in NewlyAddedImages:
                ThisImageWeight = 1
                # print("ThisImageWeight=", ThisImageWeight)
            else:
                MatchDegree = 0
                for j in range(len(NewlyAddedImagesIndex)):
                    MatchDegree = MatchDegree + self.ImageMatchingMatrix[i][NewlyAddedImagesIndex[j]]
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
                        if Calculated[j] and self.ImageMatchingMatrix[i][j] != 0:
                            MatchDegree = MatchDegree + self.ImageMatchingMatrix[i][j] * Images_Weights[j]
                            MatchTime = MatchTime + 1
                    ThisImageWeight = MatchDegree / MatchTime if MatchTime != 0 else 0
                    Images_Weights[i] = ThisImageWeight

                    if ThisImageWeight == 0:
                        ZeroWeightExist = True
                    else:
                        Calculated[i] = True

            CalculateTime = CalculateTime + 1

        return Images_Weights

    # 循环进行后续的渐进式模型训练
    def On_The_Fly_Train_Gaussians(self):
        for ProgressiveTrainingTime in range(len(self.source_path_list) - 2):
            # 数据更新同时计算这一次有多少新增影像
            self.NewImagesNum = self.IM[ProgressiveTrainingTime + 1] - self.IM[ProgressiveTrainingTime]
            self.args.source_path = self.source_path_list[ProgressiveTrainingTime]
            self.args.model_path = self.model_path_list[ProgressiveTrainingTime]
            self.args.source_path_second = self.source_path_list[ProgressiveTrainingTime + 1]
            self.args.model_path_second = self.model_path_list[ProgressiveTrainingTime + 1]
            print("\n**************Model Optimized From {} to {}**************\n".format(self.args.source_path.split('/')[-1], self.args.source_path_second.split('/')[-1]))
            self.Diary.write("\n")
            self.Diary.write("\n**************Model Optimized From {} to {}**************\n".format(self.args.source_path.split('/')[-1], self.args.source_path_second.split('/')[-1]))

            # 根据新的稀疏点云来扩张高斯点云
            self.ExpandingGS_From_SparsePCD()

            # 读入影像匹配矩阵
            self.ImageMatchingMatrix = self.GetImageMatchingMatrix()

            # 根据影像匹配矩阵来计算影像的权值
            self.Images_Weights = self.GetImagesWeightsFromMatrix()

            # 对扩张后的场景进行训练
            self.Train_Gaussians((self.args.IterationPerMergeScene + self.args.GlobalOptimizationIteration) * self.NewImagesNum, "On_The_Fly")

            # 如果需要，输出一次高斯模型
            self.GS_Save()

    # 模型最终优化
    def Final_Refinement(self):
        print("********************Final Optimization********************")
        self.Diary.write("\n")
        self.Diary.write("\n********************Final Optimization********************\n")

        self.args.source_path = self.source_path_list[-2]
        self.args.model_path = self.model_path_list[-2]
        self.args.source_path_second = self.source_path_list[-1]
        self.args.model_path_second = self.model_path_list[-1]
        prepare_output_and_logger(self.args, True)

        self.Train_Gaussians(self.args.FinalOptimizationIterations, "Final_Refinement")

        self.Render_Evaluate_All_Images()

    # 保存当前模型
    def GS_Save(self, UseSecondPath=True):
        print("\n[ITER {}] Saving Checkpoint".format(self.GS_Save_Times))
        if not UseSecondPath:
            torch.save((self.gaussians.capture(), self.GS_Save_Times), self.args.model_path + "/chkpnt" + str(self.GS_Save_Times) + ".pth")
        else:
            torch.save((self.gaussians.capture(), self.GS_Save_Times), self.args.model_path_second + "/chkpnt" + str(self.GS_Save_Times) + ".pth")
        print("\n[ITER {}] Saving Gaussians".format(self.GS_Save_Times))
        self.scene.save(self.GS_Save_Times)

        self.GS_Save_Times += 1

if __name__ == "__main__":
    # 模型初始化以及初始模型训练
    GS_Model = Gaussian_On_The_Fly_Splatting()

    # 渐进式训练模型
    GS_Model.On_The_Fly_Train_Gaussians()

    # 模型最终优化
    GS_Model.Final_Refinement()

    # 保存模型
    GS_Model.GS_Save()