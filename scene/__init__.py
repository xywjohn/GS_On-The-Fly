#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.general_utils import PILtoTorch
from scene.cameras import Camera
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F

WARNED = False

class Scene:

    gaussians : GaussianModel

    def __init__(self, args, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
            self.scene_info = scene_info
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    # 将新场景中新加入的影像加入到self.train_cameras[resolution_scale]
    def AddNewImage(self, args: ModelParams, Diary, resolution_scales=[1.0]):
        # 这里是在读入来自COLMAP的稀疏点云数据以及一些相关数据（source_path必须指向一个包含名为sparse的文件夹的文件夹）
        if os.path.exists(os.path.join(args.source_path_second, "sparse")):
            '''
            scene_info是一个不变类的实例化对象，属性包含点云数据、训练影像相机参数、测试影像相机参数、球半径与平移向量、点云的.ply文件的路径
            '''
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path_second, args.images, args.eval)

        # 为self.train_cameras进行更新，将新加入的影像的相关信息加入到其中
        # 先找到新加入的影像
        OLD_TrainCamInfos = self.scene_info.train_cameras
        NEW_TrainCamInfos = scene_info.train_cameras

        Old_Cam_Name = [cam.image_name for cam in OLD_TrainCamInfos]
        New_Cam_Name = [cam.image_name for cam in NEW_TrainCamInfos]

        for i in range(len(New_Cam_Name)):
            if (New_Cam_Name[i] not in Old_Cam_Name):
                New_CamInfo = NEW_TrainCamInfos[i]
                Diary.write(f"Newly Added Images Name: {New_CamInfo.image_name}\n")
                break

        cam_info = New_CamInfo
        orig_w, orig_h = cam_info.image.size
        for resolution_scale in resolution_scales:
            # 更新self.train_cameras中[resolution_scale]每一个影像的位姿信息
            for i in range(len(self.train_cameras[resolution_scale])):
                for cam in NEW_TrainCamInfos:
                    if (cam.image_name == self.train_cameras[resolution_scale][i].image_name):
                        self.train_cameras[resolution_scale][i].R = cam.R
                        self.train_cameras[resolution_scale][i].T = cam.T
                        break

            if args.resolution in [1, 2, 4, 8]:
                resolution = round(orig_w / (resolution_scale * args.resolution)), round(
                    orig_h / (resolution_scale * args.resolution))
            else:  # should be a type that converts to float
                if args.resolution == -1:
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
                    global_down = orig_w / args.resolution

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
                            image_name=cam_info.image_name, uid=len(self.train_cameras[resolution_scale]),
                            data_device=args.data_device)

            self.train_cameras[resolution_scale].append(NewCam)

        self.scene_info = scene_info

    def AddNewImages(self, args: ModelParams, Diary, NewImagesNum, resolution_scales=[1.0]):
        # 这里是在读入来自COLMAP的稀疏点云数据以及一些相关数据（source_path必须指向一个包含名为sparse的文件夹的文件夹）
        if os.path.exists(os.path.join(args.source_path_second, "sparse")):
            '''
            scene_info是一个不变类的实例化对象，属性包含点云数据、训练影像相机参数、测试影像相机参数、球半径与平移向量、点云的.ply文件的路径
            '''
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path_second, args.images, args.eval)

        # 为self.train_cameras进行更新，将新加入的影像的相关信息加入到其中
        # 先找到新加入的影像
        OLD_TrainCamInfos = self.scene_info.train_cameras
        NEW_TrainCamInfos = scene_info.train_cameras

        Old_Cam_Name = [cam.image_name for cam in OLD_TrainCamInfos]
        New_Cam_Name = [cam.image_name for cam in NEW_TrainCamInfos]

        # 找出所有新的影像
        CurrentNewImagesNum = 0
        NewImageNames = "["
        NewCamInfos = []
        for i in range(len(New_Cam_Name)):
            if (New_Cam_Name[i] not in Old_Cam_Name):
                New_CamInfo = NEW_TrainCamInfos[i]
                NewImageNames = NewImageNames + New_CamInfo.image_name + ", "
                NewCamInfos.append(New_CamInfo)
                CurrentNewImagesNum = CurrentNewImagesNum + 1

            if CurrentNewImagesNum == NewImagesNum:
                break

        NewImageNames = NewImageNames + "]"
        Diary.write(f"Newly Added Images Name: {NewImageNames}\n")

        # 在每一个resolution_scale下：
        NewCams = []
        for resolution_scale in resolution_scales:
            # 更新self.train_cameras中[resolution_scale]每一个影像的位姿信息
            for i in range(len(self.train_cameras[resolution_scale])):
                for cam in NEW_TrainCamInfos:
                    if (cam.image_name == self.train_cameras[resolution_scale][i].image_name):
                        self.train_cameras[resolution_scale][i].R = cam.R
                        self.train_cameras[resolution_scale][i].T = cam.T
                        break

            for cam_info in NewCamInfos:
                orig_w, orig_h = cam_info.image.size
                if args.resolution in [1, 2, 4, 8]:
                    resolution = round(orig_w / (resolution_scale * args.resolution)), round(
                        orig_h / (resolution_scale * args.resolution))
                else:  # should be a type that converts to float
                    if args.resolution == -1:
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
                        global_down = orig_w / args.resolution

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
                                image_name=cam_info.image_name, uid=len(self.train_cameras[resolution_scale]),
                                data_device=args.data_device)

                self.train_cameras[resolution_scale].append(NewCam)
                NewCams.append(NewCam)

            self.scene_info = scene_info

            return NewCams

    # 在进行最后的全局优化之前，更新一下每一个训练影像相关的位姿信息
    def UpdateTrainingImagesPos(self, args: ModelParams, resolution_scales=[1.0]):
        # 这里是在读入来自COLMAP的稀疏点云数据以及一些相关数据（source_path必须指向一个包含名为sparse的文件夹的文件夹）
        if os.path.exists(os.path.join(args.source_path_second, "sparse")):
            '''
            scene_info是一个不变类的实例化对象，属性包含点云数据、训练影像相机参数、测试影像相机参数、球半径与平移向量、点云的.ply文件的路径
            '''
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path_second, args.images, args.eval)

        NEW_TrainCamInfos = scene_info.train_cameras

        for resolution_scale in resolution_scales:
            # 更新self.train_cameras中[resolution_scale]每一个影像的位姿信息
            for i in range(len(self.train_cameras[resolution_scale])):
                for cam in NEW_TrainCamInfos:
                    if (cam.image_name == self.train_cameras[resolution_scale][i].image_name):
                        self.train_cameras[resolution_scale][i].R = cam.R
                        self.train_cameras[resolution_scale][i].T = cam.T
                        break

        self.scene_info = scene_info

    # 不断保存渐进式训练的3DGS点云模型
    def ProgressiveSave(self, args, iteration):
        point_cloud_path = os.path.join(args.model_path_second, "point_cloud", "iteration_{}".format(iteration),
                                        "point_cloud.ply")
        print(f"Save {point_cloud_path}")
        self.gaussians.save_ply(point_cloud_path)

    # 读入所有影像的深度图GT
    def ReadInDepthMapAll(self, path, device, depth_scale=1.0):
        print("Reading in depth maps")
        self.DepthMap = {}
        jpg_files = [file for file in os.listdir(path) if file.endswith('.jpg') or file.endswith('.JPG') or file.endswith('.png')]

        for image_file in jpg_files:
            image = Image.open(os.path.join(path, image_file)).convert("L")  # 确保图像是灰度模式
            image = np.array(image, dtype=np.float32) / depth_scale
            image = torch.tensor(image).to(device)

            if image.shape[1] > 1600:
                target_width = self.train_cameras[1.0][0].image_width
                target_height = self.train_cameras[1.0][0].image_height

                # 假设 image 是一个形状为 [C, H, W] 或 [B, C, H, W] 的张量
                # 如果是 [H, W]，需要先加一个通道维度变成 [1, H, W]
                if image.dim() == 2:
                    image = image.unsqueeze(0)  # 添加通道维度

                # 如果是单张图片 (形状为 [C, H, W])，需要添加 batch 维度变成 [1, C, H, W]
                if image.dim() == 3:
                    image = image.unsqueeze(0)  # 添加 batch 维度

                # 使用 interpolate 函数进行降采样
                downsampled_image = F.interpolate(image, size=(target_height, target_width), mode='bilinear', align_corners=False)

                # 如果需要还原成原始的形状 [C, H, W] 或 [H, W]，可以移除额外的维度
                downsampled_image = downsampled_image.squeeze(0)  # 移除 batch 维度

                self.DepthMap[image_file.split(".")[0]] = downsampled_image[0]
            else:
                self.DepthMap[image_file.split(".")[0]] = image

    # 读入所有的法线图GT
    def ReadInNormalAll(self, path, device):
        print("Reading in normal maps")
        self.NormalMap = {}
        jpg_files = [file for file in os.listdir(path) if file.endswith('.jpg')]

        for image_file in jpg_files:
            image = Image.open(os.path.join(path, image_file))
            image = np.array(image, dtype=np.float32)
            image = torch.tensor(image).to(device)
            image = image / 255.0

            if image.shape[1] > 1600:
                target_width = self.train_cameras[1.0][0].image_width
                target_height = self.train_cameras[1.0][0].image_height

                # 假设 image 是一个形状为 [C, H, W] 或 [B, C, H, W] 的张量
                # 如果是 [H, W]，需要先加一个通道维度变成 [1, H, W]
                if image.dim() == 2:
                    image = image.unsqueeze(0)  # 添加通道维度

                # 如果是单张图片 (形状为 [C, H, W])，需要添加 batch 维度变成 [1, C, H, W]
                if image.dim() == 3:
                    image = image.unsqueeze(0)  # 添加 batch 维度

                # 使用 interpolate 函数进行降采样
                downsampled_image = F.interpolate(image, size=(target_height, target_width), mode='bilinear', align_corners=False)

                # 如果需要还原成原始的形状 [C, H, W] 或 [H, W]，可以移除额外的维度
                downsampled_image = downsampled_image.squeeze(0)  # 移除 batch 维度

                image = downsampled_image

            normal = -((image * 2) - 1)

            self.NormalMap[image_file.split(".")[0]] = normal

class Scene_Second:

    gaussians : GaussianModel

    '''
    构造函数：初始化场景参数
    '''
    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], OnlyInitial=False):

        # model_path是训练好的模型存储的位置
        self.model_path = args.model_path_second

        # 用于存储可能存在的已经训练了一定次数的模型的训练次数
        self.loaded_iter = None

        # 高斯模型
        self.gaussians = gaussians

        # 如果存在已经训练了一定次数的模型，就会执行接下来的步骤，一般没有
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        # 这里是在读入来自COLMAP的稀疏点云数据以及一些相关数据（source_path必须指向一个包含名为sparse的文件夹的文件夹）
        if os.path.exists(os.path.join(args.source_path_second, "sparse")):
            '''
            scene_info是一个不变类的实例化对象，属性包含点云数据、训练影像相机参数、测试影像相机参数、球半径与平移向量、点云的.ply文件的路径
            scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
            '''
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path_second, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path_second, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path_second, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        # 当已经训练了一定次数的模型不存在时：
        if not self.loaded_iter:
            # 将ply_path对应的.ply文件复制到self.model_path文件夹中，并取名为input.ply
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
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
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)


        # 这里是随机打乱训练影像数据和测试影像数据的排序
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        # 将包裹所有相机的球的半径赋值给self.cameras_extent（相机分布范围）
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 这两个用于存储训练用和测试用的视角的相机参数
        self.train_cameras = {}
        self.test_cameras = {}

        # 在不同的分辨率下：
        if OnlyInitial:
            print("Only Initial! Do not read in the cameras!!")
        else:
            for resolution_scale in resolution_scales:
                # 将训练用和测试用影像对应的相机信息分别存在两个字典中，键均为分辨率，值为相机信息
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)


        # 存在已经训练了一定批次的模型：
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        # 不存在已经训练了一定批次的模型：
        else:
            # 为高斯模型初始化一个点云
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]