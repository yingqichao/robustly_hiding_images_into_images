import copy
import logging
import os

import cv2
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import canny
from torch.nn.parallel import DistributedDataParallel

import pytorch_ssim
from metrics import PSNR
from models.modules.Quantization import diff_round
from models.networks import RHI3Net
from models.patch_gan import NLayerDiscriminator
from noise_layers import *
from noise_layers.crop import Crop
from noise_layers.dropout import Dropout
from noise_layers.gaussian import Gaussian
from noise_layers.gaussian_blur import GaussianBlur
from noise_layers.middle_filter import MiddleBlur
from noise_layers.resize import Resize
from utils import stitch_images
from utils.JPEG import DiffJPEG
from .base_model import BaseModel

logger = logging.getLogger('base')

import data
# import lpips

class IRNpModel(BaseModel):
    def __init__(self, opt,args):
        """
        
        Args:
            opt: 
            args: val=0-> training, val=1-> testing, val=2->KD_JPEG training
        """
        super(IRNpModel, self).__init__(opt)
        self.IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
        ########### CONSTANTS ###############
        self.TASK_IMUGEV2 = "ImugeV2"
        self.TASK_TEST = "Test"
        self.TASK_CropLocalize = "CropLocalize"
        self.TASK_RHI3 = "RHI3"
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        self.gpu_id = self.opt['gpu_ids'][0]
        train_opt = opt['train']
        test_opt = opt['test']
        self.opt = opt
        self.args = args
        self.train_opt = train_opt
        self.test_opt = test_opt
        self.real_H, self.real_H_path, self.previous_images, self.previous_previous_images = None, None, None, None
        self.previous_canny = None
        self.task_name = self.opt['datasets']['train']['name'] #self.train_opt['task_name']
        print("Task Name: {}".format(self.task_name))
        self.global_step = 0
        self.new_task = self.train_opt['new_task']

        ############## Metrics and attacks #############
        self.tanh = nn.Tanh().cuda()
        self.psnr = PSNR(255.0).cuda()
        # self.lpips_vgg = lpips.LPIPS(net="vgg").cuda()
        # self.exclusion_loss = ExclusionLoss().type(torch.cuda.FloatTensor).cuda()
        self.ssim_loss = pytorch_ssim.SSIM().cuda()
        self.crop = Crop().cuda()
        self.dropout = Dropout().cuda()
        self.gaussian = Gaussian().cuda()
        self.salt_pepper = SaltPepper(prob=0.01).cuda()
        self.gaussian_blur = GaussianBlur().cuda()
        self.median_blur = MiddleBlur().cuda()
        self.resize = Resize().cuda()
        self.identity = Identity().cuda()
        self.width_height = opt['datasets']['train']['GT_size']
        self.jpeg_simulate = [
            [DiffJPEG(50, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(55, height=self.width_height, width=self.width_height).cuda(), ]
            ,[DiffJPEG(60, height=self.width_height, width=self.width_height).cuda(), ]
            ,[DiffJPEG(65, height=self.width_height, width=self.width_height).cuda(), ]
            ,[DiffJPEG(70, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(75, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(80, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(85, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(90, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(95, height=self.width_height, width=self.width_height).cuda(), ]
        ]

        self.bce_loss = nn.BCELoss().cuda()
        self.bce_with_logit_loss = nn.BCEWithLogitsLoss().cuda()
        self.l1_loss = nn.SmoothL1Loss(beta=0.5).cuda()  # reduction="sum"
        self.l2_loss = nn.MSELoss().cuda()  # reduction="sum"
    
        self.Quantization = diff_round
     
        self.CE_loss = nn.CrossEntropyLoss().cuda()
        self.width_height = opt['datasets']['train']['GT_size']
        self.init_gaussian = None
       
        ############## Nets ################################
        self.network_list = []
     
        self.network_list = ['hiding', 'decoup', 'reveal']

        self.decoup_net = RHI3Net(images_in=1,images_out=2)
        self.decoup_net = self.decoup_net.cuda()
        self.decoup_net = DistributedDataParallel(self.decoup_net, device_ids=[torch.cuda.current_device()],
                                                 )  # find_unused_parameters=True


        self.hiding_net = RHI3Net(images_in=self.args.images_in,images_out=1)
        self.hiding_net = self.hiding_net.cuda()
        self.hiding_net = DistributedDataParallel(self.hiding_net, device_ids=[torch.cuda.current_device()],
                                            )  # find_unused_parameters=True

        self.reveal_net = RHI3Net(images_in=1, images_out=self.args.images_in-1)
        self.reveal_net = self.reveal_net.cuda()
        self.reveal_net = DistributedDataParallel(self.reveal_net, device_ids=[torch.cuda.current_device()],
                                                  )  # find_unused_parameters=True
        
        self.discriminator_mask = NLayerDiscriminator(input_nc=3) #decoup_net().cuda()
        self.discriminator_mask = self.discriminator_mask.cuda()
        self.discriminator_mask = DistributedDataParallel(self.discriminator_mask,
                                                          device_ids=[torch.cuda.current_device()],
                                                          ) # find_unused_parameters=True

        # self.scaler_decoup = torch.cuda.amp.GradScaler()
        self.scaler_hiding = torch.cuda.amp.GradScaler()
        self.scaler_discriminator_mask = torch.cuda.amp.GradScaler()
        # self.scaler_reveal = torch.cuda.amp.GradScaler()

        ########## optimizers ##################
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0


        optim_params = []
        for k, v in self.hiding_net.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                if self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
        self.optimizer_hiding = torch.optim.AdamW(optim_params, lr=train_opt['lr_G'],
                                            weight_decay=wd_G,
                                            betas=(0.9,0.99)) # train_opt['beta1'], train_opt['beta2']
        self.optimizers.append(self.optimizer_hiding)

        # for mask discriminator
        optim_params = []
        for k, v in self.discriminator_mask.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                if self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
        self.optimizer_discriminator_mask = torch.optim.AdamW(optim_params, lr=train_opt['lr_G'],
                                                             weight_decay=wd_G,
                                                             betas=(0.9,0.99))
        self.optimizers.append(self.optimizer_discriminator_mask)

        # decoup_net
        optim_params = []
        for k, v in self.decoup_net.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                if self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
        self.optimizer_decoup_net = torch.optim.AdamW(optim_params, lr=train_opt['lr_D'],
                                                    weight_decay=wd_G,
                                                    betas=(0.9,0.99))
        self.optimizers.append(self.optimizer_decoup_net)

        # reveal
        optim_params = []
        for k, v in self.reveal_net.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                if self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
        self.optimizer_reveal_net = torch.optim.AdamW(optim_params, lr=train_opt['lr_D'],
                                                      weight_decay=wd_G,
                                                      betas=(0.9, 0.99))
        self.optimizers.append(self.optimizer_reveal_net)

        # ############## schedulers #########################
        self.schedulers = []
        for optimizer in self.optimizers:
            self.schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=118287))

        ######## init constants
        self.forward_image_buff = None
        self.reloaded_time = 0
        self.basic_weight_fw = 5

        ########## Load pre-trained ##################
     
        # good_models: '/model/Rerun_4/29999'
        self.out_space_storage = '/home/qcying/20220106_IMUGE'
        self.model_storage = '/model/Rerun_3/'
        self.model_path = str(2999) # 29999
     

      
        load_models = True
        load_state = False
        if load_models:
            self.pretrain = self.out_space_storage + self.model_storage + self.model_path
            self.reload(self.pretrain, self.network_list)
            # ## load states
            # state_path = self.out_space_storage + self.model_storage + '{}.state'.format(self.model_path)
            # if load_state:
            #     logger.info('Loading training state')
            #     if os.path.exists(state_path):
            #         self.resume_training(state_path, self.network_list)
            #     else:
            #         logger.info('Did not find state [{:s}] ...'.format(state_path))

    def clamp_with_grad(self,tensor):
        tensor_clamp = torch.clamp(tensor,0,1)
        return tensor+ (tensor_clamp-tensor).clone().detach()

    def feed_data(self, batch):
        img, label, canny_image = batch
        self.real_H = img.cuda()
        self.canny_image = canny_image.cuda()

    def gaussian_batch(self, dims):
        return self.clamp_with_grad(torch.randn(tuple(dims)).cuda())

    def optimize_parameters(self, step, latest_values=None, train=True, eval_dir=None):
        self.hiding_net.train()
        self.decoup_net.train()
        self.discriminator_mask.train()
        self.reveal_net.train()
        self.optimizer_discriminator_mask.zero_grad()
        self.optimizer_hiding.zero_grad()
        self.optimizer_decoup_net.zero_grad()
        self.optimizer_reveal_net.zero_grad()

        logs, debug_logs = [], []

        self.real_H = self.clamp_with_grad(self.real_H)
        ori_batch_size, ori_channels, height_width, _ = self.real_H.shape
        num_hidden_images = self.args.images_in
        batch = int(ori_batch_size//num_hidden_images)
        channels = ori_channels * num_hidden_images

        psnr_thresh = 33
        lr = self.get_current_learning_rate()
        logs.append(('lr', lr))

        modified_input = self.real_H.clone().detach()
        modified_input = self.clamp_with_grad(modified_input)
        modified_canny = self.canny_image.clone().detach()
        modified_canny = self.clamp_with_grad(modified_canny)
        check_status = self.l1_loss(modified_input, self.real_H) + self.l1_loss(modified_canny,self.canny_image)
        if check_status > 0:
            print(f"Strange input detected! {check_status} skip")
            return logs, debug_logs
        else:
            modified_input = modified_input.reshape(batch, channels, height_width, height_width)
            ## COVER SECRET SPLIT
            cover_image = modified_input[:, :3]
            secret_images = modified_input[:, 3:]
            with torch.cuda.amp.autocast():

                marked_image = self.hiding_net(modified_input)
                marked_image = marked_image[0]
                marked_image = self.clamp_with_grad(marked_image)
                psnr_forward = self.psnr(self.postprocess(modified_input),
                                         self.postprocess(marked_image)).item()

                cover_compressed, forward_compressed = self.benign_attacks(
                    forward_image=marked_image, logs=logs, modified_input=cover_image)

                extract_residual, extract_original = self.decoup_net(forward_compressed)
                extract_residual = self.clamp_with_grad(extract_residual)
                extract_original = self.clamp_with_grad(extract_original)

                extract_images = self.reveal_net(extract_residual)
                extract_images = self.clamp_with_grad(extract_images)
                extract_images = torch.cat(extract_images,dim=1)

                ###################
                ### LOSSES
                loss = 0
                #################
                # FORWARD LOSS

                l_forward = self.l1_loss(marked_image, cover_image)
                weight_fw = 1.5
                logs.append(('CK', self.global_step%5))
                loss += weight_fw * l_forward
                #################
                ### BACKWARD LOSS
                l_backward_cover =  self.l1_loss(extract_original, cover_compressed)
                l_backward = self.l1_loss(extract_images, secret_images)
                psnr_backward_cover = self.psnr(self.postprocess(extract_original),
                                                self.postprocess(cover_compressed)).item()
                psnr_backward = self.psnr(self.postprocess(extract_images[:,:3]),
                                                self.postprocess(secret_images[:,:3])).item()

                loss += 1.0 * l_backward
                loss += 1.0 * l_backward_cover

                ##################
                ## GAN
                # loss += weight_GAN * REV_GAN

                ## LOG FILE

                logs.append(('sum', loss.item()))
                logs.append(('lF', l_forward.item()))
                logs.append(('lB', l_backward.item()))

                logs.append(('PF', psnr_forward))
                logs.append(('PB', psnr_backward))
                logs.append(('PBC', psnr_backward_cover))

                logs.append(('FW', psnr_forward))
                logs.append(('BK', psnr_backward))

            self.optimizer_hiding.zero_grad()
            self.optimizer_decoup_net.zero_grad()
            self.optimizer_reveal_net.zero_grad()
            # loss.backward()
            self.scaler_hiding.scale(loss).backward()
            if self.train_opt['gradient_clipping']:
                nn.utils.clip_grad_norm_(self.hiding_net.parameters(), 2)
                nn.utils.clip_grad_norm_(self.reveal_net.parameters(), 2)
                nn.utils.clip_grad_norm_(self.decoup_net.parameters(), 2)
            # self.optimizer_hiding.step()
            self.scaler_hiding.step(self.optimizer_hiding)
            self.scaler_hiding.step(self.optimizer_reveal_net)
            self.scaler_hiding.step(self.optimizer_decoup_net)

            # logs.append(('STATE', STATE))
            # for scheduler in self.schedulers:
            #     scheduler.step()

            self.scaler_hiding.update()

            ################# observation zone
            # with torch.no_grad():
            #     REVERSE, _ = self.hiding_net(torch.cat((attacked_real_jpeg * (1 - masks),
            #                    torch.zeros_like(modified_canny).cuda()), dim=1), rev=True)
            #     REVERSE = self.clamp_with_grad(REVERSE)
            #     REVERSE = REVERSE[:, :3, :, :]
            #     l_REV = (self.l1_loss(REVERSE * masks_expand, modified_input * masks_expand))
            #     logs.append(('observe', l_REV.item()))
            anomalies = False #CE_recall.item()>0.5
            if anomalies or self.global_step % 200 == 3 or self.global_step<=10:
                images = stitch_images(
                    self.postprocess(cover_image),
                    self.postprocess(secret_images[:,:3]),
                    self.postprocess(secret_images[:, 3:]),
                    self.postprocess(marked_image),
                    self.postprocess(10 * torch.abs(marked_image - cover_image)),
                    self.postprocess(forward_compressed),
                    self.postprocess(10 * torch.abs(marked_image - forward_compressed)),

                    self.postprocess(extract_original),
                    self.postprocess(10 * torch.abs(extract_original - cover_image)),
                    self.postprocess(extract_images[:,:3]),
                    self.postprocess(10 * torch.abs(extract_images[:,:3] - secret_images[:,:3])),
                    self.postprocess(extract_images[:, 3:]),
                    self.postprocess(10 * torch.abs(extract_images[:, 3:] - secret_images[:, 3:])),
                    img_per_row=1
                )

                name = self.out_space_storage + '/images/'+self.task_name+'_'+str(self.gpu_id)+'/'\
                       +str(self.global_step).zfill(5) + "_ "+str(self.gpu_id) + "_ "+str(self.rank) \
                       +("" if not anomalies else "_anomaly")+ ".png"
                print('\nsaving sample ' + name)
                images.save(name)

        ######## Finally ####################
        if self.global_step % 1000== 999 or self.global_step==9:
            if self.rank==0:
                logger.info('Saving models and training states.')
                self.save(self.global_step, folder='model', network_list=self.network_list)
        if self.real_H is not None:
            self.previous_canny = self.canny_image
            if self.previous_images is not None:
                self.previous_previous_images = self.previous_images.clone().detach()
            self.previous_images = self.real_H
        self.global_step = self.global_step + 1
        return logs, debug_logs


    def benign_attacks(self,forward_image, modified_input,logs):
        # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        # cv2.imwrite(f'{save_name}_{str(idx_atkimg)}_{str(self.rank)}_compress.jpg', ndarr, encode_param)
        batch_size, height_width = self.real_H.shape[0], self.real_H.shape[2]

        GT_modified_input_real_jpeg = torch.rand_like(modified_input)
        forward_real_jpeg = torch.rand_like(forward_image)

        if self.global_step % 5 == 1:
            blurring_layer = self.gaussian_blur
        elif self.global_step%5==2:
            blurring_layer = self.median_blur
        elif self.global_step%5==0:
            blurring_layer = self.resize
        else:
            blurring_layer = self.identity
        quality_idx = np.random.randint(19, 21) if self.global_step % 5<=2 else np.random.randint(10, 17)
        quality = int(quality_idx * 5)
        ## simulation
        jpeg_layer_after_blurring = self.jpeg_simulate[quality_idx - 10][0] if quality < 100 else self.identity
        GT_modified_input_simulate = self.Quantization(self.clamp_with_grad(jpeg_layer_after_blurring(blurring_layer(modified_input))))
        forward_simulate = self.Quantization(self.clamp_with_grad(jpeg_layer_after_blurring(blurring_layer(forward_image))))
        if self.global_step%5==4:
            ## WE ONLY SIMULATE THE (JPEG) ATTACKS
            GT_modified_input = GT_modified_input_simulate
            forward_compressed = forward_simulate
        else:
            ## WE NOT ONLY SIMULATE ATTACKS BUT ALSO ADD RESIDUAL ONTO THE SIMULATED RESULT
            for idx_atkimg in range(batch_size):
                grid = modified_input[idx_atkimg].clone().detach()
                realworld_attack = self.real_world_attacking_on_ndarray(grid, quality)
                GT_modified_input_real_jpeg[idx_atkimg:idx_atkimg + 1] = realworld_attack

                grid = forward_image[idx_atkimg].clone().detach()
                realworld_attack = self.real_world_attacking_on_ndarray(grid, quality)
                forward_real_jpeg[idx_atkimg:idx_atkimg + 1] = realworld_attack

            GT_modified_input_real_jpeg = GT_modified_input_real_jpeg.clone().detach()
            GT_modified_input = GT_modified_input_simulate + (GT_modified_input_real_jpeg - GT_modified_input_simulate).clone().detach()

            forward_real_jpeg = forward_real_jpeg.clone().detach()
            forward_compressed = forward_simulate + (forward_real_jpeg - forward_simulate).clone().detach()

            ## ERROR
            error_detach = forward_real_jpeg - forward_simulate
            l_residual = self.l1_loss(error_detach, torch.zeros_like(error_detach).cuda())
            logs.append(('DETACH', l_residual.item()))

        error_scratch = forward_compressed - forward_image
        l_scratch = self.l1_loss(error_scratch, torch.zeros_like(error_scratch).cuda())
        logs.append(('SCRATCH', l_scratch.item()))
        return GT_modified_input, forward_compressed

    def real_world_attacking_on_ndarray(self,grid, qf_after_blur, index=None):
        batch_size, height_width = self.real_H.shape[0], self.real_H.shape[2]
        if index is None:
            index = self.global_step % 5
        if index == 0:
            grid = self.resize(grid.unsqueeze(0))[0]
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        if index == 1:
            realworld_attack = cv2.GaussianBlur(ndarr, (5, 5), 0)
        elif index == 2:
            realworld_attack = cv2.medianBlur(ndarr, 5)
        else:
            realworld_attack = ndarr
        _, realworld_attack = cv2.imencode('.jpeg', realworld_attack, (int(cv2.IMWRITE_JPEG_QUALITY), qf_after_blur))
        realworld_attack = cv2.imdecode(realworld_attack, cv2.IMREAD_UNCHANGED)
        realworld_attack = data.util.channel_convert(realworld_attack.shape[2], 'RGB', [realworld_attack])[0]
        realworld_attack = cv2.resize(copy.deepcopy(realworld_attack), (height_width, height_width),
                                      interpolation=cv2.INTER_LINEAR)
        realworld_attack = realworld_attack.astype(np.float32) / 255.
        realworld_attack = torch.from_numpy(
            np.ascontiguousarray(np.transpose(realworld_attack, (2, 0, 1)))).float()
        realworld_attack = realworld_attack.unsqueeze(0).cuda()
        return realworld_attack


    def GAN_loss(self,model, reversed_image,modified_input, logs):
        batch_size, height_width = self.real_H.shape[0], self.real_H.shape[2]
        gen_input_fake = reversed_image
        # dis_input_real = modified_input.clone().detach()
        # dis_real = self.discriminator_mask(dis_input_real)  # in: (grayscale(1) + edge(1))
        gen_fake = model(gen_input_fake)
        REV_GAN = self.bce_with_logit_loss(gen_fake, torch.ones_like(gen_fake))  # / torch.mean(masks)
        # gen_style_loss = 0
        # for i in range(len(dis_real_feat)):
        #     gen_style_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        # gen_style_loss = gen_style_loss / 5
        # logs.append(('REV_GAN', gen_style_loss.item()))
        # REV_GAN += gen_style_loss
        return REV_GAN

    def GAN_training(self,model, modified_input,reversed_image,masks_GT,logs):
        dis_input_real = modified_input
        dis_input_fake = reversed_image
        dis_real = model(dis_input_real)
        dis_fake = model(dis_input_fake)
        dis_real_loss = self.bce_with_logit_loss(dis_real, torch.ones_like(dis_real))
        # downsample_mask = Functional.interpolate(
        #                         1-masks_GT_expand,
        #                         size=[dis_real.shape[2], dis_real.shape[3]],
        #                         mode='bilinear')
        # downsample_mask = self.clamp_with_grad(downsample_mask)
        # dis_fake_loss = self.bce_with_logit_loss(dis_fake, torch.zeros_like(dis_real))
        dis_fake_loss = self.bce_with_logit_loss(dis_fake, torch.zeros_like(dis_fake))
        dis_loss = (dis_real_loss + dis_fake_loss) / 2
        return dis_loss

    def evaluate(self,data_origin=None,data_immunize=None,data_tampered=None,data_tampersource=None,data_mask=None):
        self.hiding_net.eval()
        self.decoup_net.eval()
        with torch.no_grad():
            psnr_forward_sum, psnr_backward_sum = [0,0,0,0,0],  [0,0,0,0,0]
            ssim_forward_sum, ssim_backward_sum =  [0,0,0,0,0],  [0,0,0,0,0]
            F1_sum =  [0,0,0,0,0]
            valid_images = [0,0,0,0,0]
            logs, debug_logs = [], []
            image_list_origin = None if data_origin is None else self.get_paths_from_images(data_origin)
            image_list_immunize = None if data_immunize is None else self.get_paths_from_images(data_immunize)
            image_list_tamper = None if data_tampered is None else self.get_paths_from_images(data_tampered)
            image_list_tampersource = None if data_tampersource is None else self.get_paths_from_images(data_tampersource)
            image_list_mask = None if data_mask is None else self.get_paths_from_images(data_mask)

            for idx in range(len(image_list_origin)):

                p, q, r = image_list_origin[idx]
                ori_path = os.path.join(p, q, r)
                img_GT = self.load_image(ori_path)
                print("Ori: {} {}".format(ori_path, img_GT.shape))
                self.real_H = self.img_random_crop(img_GT, 608, 608).cuda().unsqueeze(0)
                self.real_H = self.clamp_with_grad(self.real_H)
                img_gray = rgb2gray(img_GT)
                sigma = 2  # random.randint(1, 4)
                cannied = canny(img_gray, sigma=sigma, mask=None).astype(np.float)
                self.canny_image = self.image_to_tensor(cannied).cuda().unsqueeze(0)

                p, q, r = image_list_immunize[idx]
                immu_path = os.path.join(p, q, r)
                img_GT = self.load_image(immu_path)
                print("Imu: {} {}".format(immu_path, img_GT.shape))
                self.immunize = self.img_random_crop(img_GT, 608, 608).cuda().unsqueeze(0)
                self.immunize = self.clamp_with_grad(self.immunize)
                p, q, r = image_list_tamper[idx]
                attack_path = os.path.join(p, q, r)
                img_GT = self.load_image(attack_path)
                print("Atk: {} {}".format(attack_path, img_GT.shape))
                self.attacked_image = self.img_random_crop(img_GT, 608, 608).cuda().unsqueeze(0)
                self.attacked_image = self.clamp_with_grad(self.attacked_image)
                p, q, r = image_list_tampersource[idx]
                another_path = os.path.join(p, q, r)
                img_GT = self.load_image(another_path)
                print("Another: {} {}".format(another_path, img_GT.shape))
                self.another_image = self.img_random_crop(img_GT, 608, 608).cuda().unsqueeze(0)
                self.another_image = self.clamp_with_grad(self.another_image)
                p, q, r = image_list_mask[idx]
                mask_path = os.path.join(p, q, r)
                img_GT = self.load_image(mask_path, grayscale=True)
                print("Mask: {} {}".format(mask_path, img_GT.shape))
                self.mask = self.img_random_crop(img_GT, 608, 608, grayscale=True).cuda().unsqueeze(0)
                self.mask = self.clamp_with_grad(self.mask)
                self.mask = self.mask.repeat(1,3,1,1)

                ### skip images with too large tamper masks
                masked_rate = torch.mean(self.mask)
                redo_gen_mask = masked_rate>0.5
                    # print("Masked rate exceed maximum: {}".format(masked_rate))
                    # continue

                catogory = min(4,int(masked_rate*20))
                valid_images[catogory] += 1
                is_copy_move = False
                if True: #self.immunize is None:
                    ##### re-generates immunized images ########
                    modified_input = self.real_H
                    # print(self.canny_image.shape)
                    forward_stuff = self.hiding_net(x=torch.cat((modified_input, self.canny_image), dim=1))
                    self.immunize, forward_null = forward_stuff[:, :3, :, :], forward_stuff[:, 3:, :, :]
                    self.immunize = self.clamp_with_grad(self.immunize)
                    self.immunize = self.Quantization(self.immunize)
                    forward_null = self.clamp_with_grad(forward_null)

                ####### Tamper ###############
                if True: #self.attacked_image is None:

                    self.attacked_image = self.immunize * (1-self.mask) + self.another_image * self.mask
                    self.attacked_image = self.clamp_with_grad(self.attacked_image)

                index = np.random.randint(0,5)
                self.attacked_image = self.real_world_attacking_on_ndarray(self.attacked_image[0],
                                                                           qf_after_blur=100 if index<3 else 70,
                                                                           index=index)
                self.reverse_GT = self.real_world_attacking_on_ndarray(self.real_H[0],
                                                                           qf_after_blur=100 if index < 3 else 70,
                                                                           index=index)

                # self.attacked_image = self.clamp_with_grad(self.attacked_image)
                # self.attacked_image = self.Quantization(self.attacked_image)

                self.diffused_image = self.attacked_image.clone().detach()
                self.predicted_mask = torch.sigmoid(self.decoup_net(self.diffused_image))


                self.predicted_mask = torch.where(self.predicted_mask > 0.5, 1.0, 0.0)
                self.predicted_mask = self.Erode_Dilate(self.predicted_mask)

                F1, TP = self.F1score(self.predicted_mask, self.mask, thresh=0.5)
                F1_sum[catogory] += F1

                self.predicted_mask = self.predicted_mask.repeat(1, 3, 1, 1)

                self.rectified_image = self.attacked_image * (1 - self.predicted_mask)
                self.rectified_image = self.clamp_with_grad(self.rectified_image)


                canny_input = (torch.zeros_like(self.canny_image).cuda())

                reversed_stuff, reverse_feature = self.hiding_net(
                    torch.cat((self.rectified_image, canny_input), dim=1), rev=True)
                reversed_ch1, reversed_ch2 = reversed_stuff[:, :3, :, :], reversed_stuff[:, 3:, :, :]
                reversed_ch1 = self.clamp_with_grad(reversed_ch1)
                reversed_ch2 = self.clamp_with_grad(reversed_ch2)
                self.reversed_image = reversed_ch1
                self.reversed_canny = reversed_ch2

                psnr_forward = self.psnr(self.postprocess(self.real_H), self.postprocess(self.immunize)).item()
                psnr_backward = self.psnr(self.postprocess(self.reverse_GT),
                                          self.postprocess(self.reversed_image)).item()
                l_percept_fw_ssim = - self.ssim_loss(self.immunize, self.real_H)
                l_percept_bk_ssim = - self.ssim_loss(self.reversed_image, self.reverse_GT)
                ssim_forward = (-l_percept_fw_ssim).item()
                ssim_backward = (-l_percept_bk_ssim).item()
                psnr_forward_sum[catogory] += psnr_forward
                psnr_backward_sum[catogory] += psnr_backward
                ssim_forward_sum[catogory] += ssim_forward
                ssim_backward_sum[catogory] += ssim_backward
                print("PF {:2f} PB {:2f} ".format(psnr_forward,psnr_backward))
                print("SF {:3f} SB {:3f} ".format(ssim_forward, ssim_backward))
                print("PFSum {:2f} SFSum {:2f} ".format(np.sum(psnr_forward_sum)/np.sum(valid_images),
                                                        np.sum(ssim_forward_sum)/np.sum(valid_images)))
                print("PB {:3f} {:3f} {:3f} {:3f} {:3f}".format(
                    psnr_backward_sum[0]/(valid_images[0]+1e-3), psnr_backward_sum[1]/(valid_images[1]+1e-3),
                    psnr_backward_sum[2]/(valid_images[2]+1e-3),
                    psnr_backward_sum[3]/(valid_images[3]+1e-3), psnr_backward_sum[4]/(valid_images[4]+1e-3)))
                print("SB {:3f} {:3f} {:3f} {:3f} {:3f}".format(
                    ssim_backward_sum[0] / (valid_images[0]+1e-3), ssim_backward_sum[1] / (valid_images[1]+1e-3),
                    ssim_backward_sum[2] / (valid_images[2]+1e-3),
                    ssim_backward_sum[3] / (valid_images[3]+1e-3), ssim_backward_sum[4] / (valid_images[4]+1e-3)))
                print("F1 {:3f} {:3f} {:3f} {:3f} {:3f}".format(
                    F1_sum[0] / (valid_images[0]+1e-3), F1_sum[1] / (valid_images[1]+1e-3),
                    F1_sum[2] / (valid_images[2]+1e-3),
                    F1_sum[3] / (valid_images[3]+1e-3), F1_sum[4] / (valid_images[4]+1e-3)))
                print("Valid {:3f} {:3f} {:3f} {:3f} {:3f}".format(valid_images[0],valid_images[1],valid_images[2],
                                                                   valid_images[3],valid_images[4]))

                # ####### Save independent images #############
                save_images = True
                if save_images:
                    eval_kind = self.opt['eval_kind'] #'copy-move/results/RESIZE'
                    eval_attack = self.opt['eval_attack']
                    main_folder = os.path.join(self.out_space_storage,'results', self.opt['dataset_name'], eval_kind)
                    sub_folder = os.path.join(main_folder,eval_attack)
                    if not os.path.exists(main_folder): os.mkdir(main_folder)
                    if not os.path.exists(sub_folder): os .mkdir(sub_folder)
                    if not os.path.exists(sub_folder+ '/recovered_image'): os.mkdir(sub_folder+ '/recovered_image')
                    if not os.path.exists(sub_folder + '/predicted_masks'): os.mkdir(sub_folder + '/predicted_masks')

                    name = sub_folder + '/recovered_image/' + r
                    for image_no in range(self.reversed_image.shape[0]):
                        camera_ready = self.reversed_image[image_no].unsqueeze(0)
                        torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                     name, nrow=1, padding=0,
                                                     normalize=False)
                    print("Saved:{}".format(name))

                    name = sub_folder + '/predicted_masks/' + r

                    for image_no in range(self.predicted_mask.shape[0]):
                        camera_ready = self.predicted_mask[image_no].unsqueeze(0)
                        torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                                     name, nrow=1, padding=0,
                                                     normalize=False)
                    print("Saved:{}".format(name))

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self.IMG_EXTENSIONS)

    def get_paths_from_images(self, path):
        '''get image path list from image folder'''
        assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
        images = []
        for dirpath, _, fnames in sorted(os.walk(path)):
            for fname in sorted(fnames):
                if self.is_image_file(fname):
                    # img_path = os.path.join(dirpath, fname)
                    images.append((path, dirpath[len(path) + 1:], fname))
        assert images, '{:s} has no valid image file'.format(path)
        return images

    def print_individual_image(self, cropped_GT, name):
        for image_no in range(cropped_GT.shape[0]):
            camera_ready = cropped_GT[image_no].unsqueeze(0)
            torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                         name, nrow=1, padding=0, normalize=False)

    def load_image(self, path, readimg=False, Height=608, Width=608,grayscale=False):
        import data.util as util
        GT_path = path

        img_GT = util.read_img(GT_path)

        # change color space if necessary
        # img_GT = util.channel_convert(img_GT.shape[2], 'RGB', [img_GT])[0]
        if grayscale:
            img_GT = rgb2gray(img_GT)

        img_GT = cv2.resize(copy.deepcopy(img_GT), (Width, Height), interpolation=cv2.INTER_LINEAR)
        return img_GT

    def img_random_crop(self, img_GT, Height=608, Width=608, grayscale=False):
        # # randomly crop
        # H, W = img_GT.shape[0], img_GT.shape[1]
        # rnd_h = random.randint(0, max(0, H - Height))
        # rnd_w = random.randint(0, max(0, W - Width))
        #
        # img_GT = img_GT[rnd_h:rnd_h + Height, rnd_w:rnd_w + Width, :]
        #
        # orig_height, orig_width, _ = img_GT.shape
        # H, W = img_GT.shape[0], img_GT.shape[1]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if not grayscale:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        else:
            img_GT = self.image_to_tensor(img_GT)

        return img_GT.cuda()

    def tensor_to_image(self, tensor):

        tensor = tensor * 255.0
        image = tensor.permute(1, 2, 0).detach().cpu().numpy()
        # image = tensor.permute(0,2,3,1).detach().cpu().numpy()
        return np.clip(image, 0, 255).astype(np.uint8)

    def tensor_to_image_batch(self, tensor):

        tensor = tensor * 255.0
        image = tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
        # image = tensor.permute(0,2,3,1).detach().cpu().numpy()
        return np.clip(image, 0, 255).astype(np.uint8)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def image_to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(np.asarray(img)).float()
        return img_t

    def reload(self,pretrain, network_list=['netG','decoup_net']):
        if 'netG' in network_list:
            load_path_G = pretrain + "_netG.pth"
            if load_path_G is not None:
                logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.hiding_net, strict=True)
                else:
                    logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

        if 'KD_JPEG' in network_list:
            load_path_G = pretrain + "_KD_JPEG.pth"
            if load_path_G is not None:
                logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.KD_JPEG_net, strict=False)
                else:
                    logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

        if 'discriminator_mask' in network_list:
            load_path_G = pretrain + "_discriminator_mask.pth"
            if load_path_G is not None:
                logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.discriminator_mask, strict=False)
                else:
                    logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

        if 'qf_predict' in network_list:
            load_path_G = pretrain + "_qf_predict.pth"
            if load_path_G is not None:
                logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.qf_predict_network, self.opt['path']['strict_load'])
                else:
                    logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

        # if 'decoup_net' in network_list:
        #     load_path_G = pretrain + "_decoup_net.pth"
        #     if load_path_G is not None:
        #         logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
        #         if os.path.exists(load_path_G):
        #             self.load_network(load_path_G, self.decoup_net, strict=False)
        #         else:
        #             logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

        if 'generator' in network_list:
            load_path_G = pretrain + "_generator.pth"
            if load_path_G is not None:
                logger.info('Loading model for class [{:s}] ...'.format(load_path_G))
                if os.path.exists(load_path_G):
                    self.load_network(load_path_G, self.generator, strict=True)
                else:
                    logger.info('Did not find model for class [{:s}] ...'.format(load_path_G))

    def save(self, iter_label, folder='model', network_list=['netG','decoup_net']):
        if 'hiding_net' in network_list:
            self.save_network(self.hiding_net, 'netG', iter_label if self.rank==0 else 0, model_path=self.out_space_storage+f'/{folder}/'+self.task_name+'_'+str(self.gpu_id)+'/')
        if 'decoup_net' in network_list:
            self.save_network(self.decoup_net,  'decoup_net', iter_label if self.rank==0 else 0, model_path=self.out_space_storage+f'/{folder}/'+self.task_name+'_'+str(self.gpu_id)+'/')
        if 'discriminator_mask' in network_list:
            self.save_network(self.discriminator_mask, 'discriminator_mask', iter_label if self.rank==0 else 0, model_path=self.out_space_storage+f'/{folder}/'+self.task_name+'_'+str(self.gpu_id)+'/')
        # # if 'netG' in network_list:
        # self.save_training_state(epoch=0, iter_step=iter_label if self.rank==0 else 0, model_path=self.out_space_storage+f'/{folder}/'+self.task_name+'_'+str(self.gpu_id)+'/',
        #                          network_list=network_list)


    def random_float(self, min, max):
        return np.random.rand() * (max - min) + min

    def F1score(self, predict_image, gt_image, thresh=0.2):
        # gt_image = cv2.imread(src_image, 0)
        # predict_image = cv2.imread(dst_image, 0)
        # ret, gt_image = cv2.threshold(gt_image[0], int(255 * thresh), 255, cv2.THRESH_BINARY)
        # ret, predicted_binary = cv2.threshold(predict_image[0], int(255*thresh), 255, cv2.THRESH_BINARY)
        predicted_binary = self.tensor_to_image(predict_image[0])
        ret, predicted_binary = cv2.threshold(predicted_binary, int(255 * thresh), 255, cv2.THRESH_BINARY)
        gt_image = self.tensor_to_image(gt_image[0,:1,:,:])
        ret, gt_image = cv2.threshold(gt_image, int(255 * thresh), 255, cv2.THRESH_BINARY)

        # print(predicted_binary.shape)

        [TN, TP, FN, FP] = getLabels(predicted_binary, gt_image)
        # print("{} {} {} {}".format(TN,TP,FN,FP))
        F1 = getF1(TP, FP, FN)
        # cv2.imwrite(save_path, predicted_binary)
        return F1, TP

def getLabels(img, gt_img):
    height = img.shape[0]
    width = img.shape[1]
    #TN, TP, FN, FP
    result = [0, 0, 0 ,0]
    for row in range(height):
        for column in range(width):
            pixel = img[row, column]
            gt_pixel = gt_img[row, column]
            if pixel == gt_pixel:
                result[(pixel // 255)] += 1
            else:
                index = 2 if pixel == 0 else 3
                result[index] += 1
    return result

def getACC(TN, TP, FN, FP):
    return (TP+TN)/(TP+FP+FN+TN)
def getFPR(TN, FP):
    return FP / (FP + TN)
def getTPR(TP, FN):
    return TP/ (TP+ FN)
def getTNR(FP, TN):
    return TN/ (FP+ TN)
def getFNR(FN, TP):
    return FN / (TP + FN)
def getF1(TP, FP, FN):
    return (2*TP)/(2*TP+FP+FN)
def getBER(TN, TP, FN, FP):
    return 1/2*(getFPR(TN, FP)+FN/(FN+TP))

if __name__ == '__main__':
    ## TESTING THE CODE
    modified_input = torch.ones((3,3,64,64)).cuda()
    decoup_net = RHI3Net(images_in=1, images_out=2).cuda()
    hiding_net = RHI3Net(images_in=3, images_out=1).cuda()
    reveal_net = RHI3Net(images_in=1, images_out=2).cuda()
    ori_batch_size, ori_channels, height_width, _ = modified_input.shape
    num_hidden_images = 3
    batch = int(ori_batch_size // (num_hidden_images))
    channels = ori_channels * (num_hidden_images)

    modified_input = modified_input.reshape(batch, channels, height_width, height_width)
    ## COVER SECRET SPLIT
    cover_image = modified_input[:, :3]
    secret_images = modified_input[:, 3:]
    print(f"Shape of cover_image: {cover_image.shape}")
    print(f"Shape of secret_images: {secret_images.shape}")
    with torch.cuda.amp.autocast():
        marked_image = hiding_net(modified_input)
        print(len(marked_image))
        marked_image = marked_image[0]
        print(f"Shape of marked_image: {marked_image.shape}")
        extract_residual, extract_original = decoup_net(marked_image)
        print(len(extract_residual))
        print(len(extract_original))
        print(f"Shape of extract_residual: {extract_residual.shape}")
        print(f"Shape of extract_original: {extract_original.shape}")
        extract_images = reveal_net(extract_residual)
        print(len(extract_images))
        extract_images = torch.cat(extract_images, dim=1)
        print(f"Shape of extract_images: {extract_images.shape}")

