import os, sys, datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from training_modules import configure as cfg
from training_modules import data_load as dl
from training_modules import model_load as ml
from training_modules import codec_module as cm
from training_modules import loss_modules as lm
from auxiliary_modules import find_free_port as ffp
from auxiliary_modules import compute_metrics as metrics
from auxiliary_modules import video_tensor_processor as vtp

def distributed_train(rank, world_size:int, training_config:cfg.Configuration, distributed_port:str):
    # DISTRIBUTED TRAINING SETUP:
    os.environ['MASTER_ADDR']          = 'localhost'
    os.environ['MASTER_PORT']          = ffp.find_free_port(base_port=rank)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, training_config.GPU_index))
    device                             = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size, init_method='tcp://127.0.0.1:' + distributed_port)
    training_state_log = os.path.join(training_config.logdir, 'training_state_forDebug.txt')
    if os.path.exists(training_state_log) and training_config.keep_training_state and rank==0:
        os.remove(training_state_log)
    
    # DATA INDEX SET BUILDING:
    dataset_manager                   = dl.DatasetManager(training_config, rank, world_size)
    train_dataloader, eval_dataloader = dataset_manager.get_dataloader()
    
    # MODEL INITIALIZATION:
    rescaling_model    = ml.RUVC(training_config, rank).to(torch.float32).to(device).load_model(rank)
    rescaling_model    = nn.parallel.DistributedDataParallel(rescaling_model, device_ids=[rank])
    loss_model         = lm.LossModel(channel=3*training_config.RUVC_GOP, device=device)
    quantization_model = ml.QuantizationModel()
    codec              = cm.CodecPure(training_config, rank=rank, sample_num=len(train_dataloader))
    # codec              = cm.CodecWithSurrogate(training_config) if training_config.use_surrogate else cm.CodecPure(training_config, rank=rank)
    # constraint_computer= ml.ConstraintComputer(training_config.RUVC_GOP).to(device)
    # codec              = nn.parallel.DistributedDataParallel(codec.to(device), device_ids=[rank]) if training_config.use_surrogate else codec
    
    # TRAINING & Evaluation:
    for epoch in range(training_config.start_epoch, training_config.start_epoch + training_config.epoch):
        
        ## Training:
        model_train(
            rescaling_model    = rescaling_model, 
            codec              = codec, 
            quantization_model = quantization_model, 
            loss_model         = loss_model, 
            dataloader         = train_dataloader, 
            training_config    = training_config, 
            device             = device, 
            epoch              = epoch, 
            rank               = rank, 
            training_state_log = training_state_log
        )
        
        ## Evalution
        model_eval(
            rescaling_model    = rescaling_model, 
            codec              = codec, 
            quantization_model = quantization_model, 
            loss_model         = loss_model, 
            dataloader         = eval_dataloader, 
            training_config    = training_config, 
            device             = device, 
            epoch              = epoch, 
            rank               = rank, 
            training_state_log = training_state_log
        )

        ## New Epoch Preparation
        if epoch < training_config.start_epoch + training_config.epoch:
            dataset_manager.new_epoch(epoch=epoch+1)
            peak_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)
            print(f"GPU {rank} Peak Memory Usage: {peak_memory:.2f} GB")

            torch.cuda.reset_peak_memory_stats(device=device)
                        
                    
def model_train(
        rescaling_model    :ml.RUVC,                    ## real type: nn.parallel.DistributedDataParallel
        codec              :cm.CodecPure, 
        quantization_model :ml.QuantizationModel, 
        loss_model         :lm.LossModel, 
        dataloader         :dl.TrainDataset,            ## real type: torch.utils.data.Dataloader
        training_config    :cfg.Configuration, 
        device             :torch.device, 
        epoch              :int, 
        rank               :int, 
        training_state_log :str
    ) -> None:
    '''
    Separately encapsulate training and evaluation functions to avoid memory leaks.
        training_end: indicates whether the epoch is over
        rrv         : raw resolution video
        dcv         : downsampled constraint video
        LF and HF   : video (feature map) of low and high frequency
        LR and HR   : video (feature map) of low and high resolution
    '''
    rescaling_model.train()
    rescaling_model.module.optical_flow_estimation.eval()
    iteration_num   = 1
    for _, rrv in enumerate(dataloader):
        ### iteration setup: data preparation
        now_random_seed = (epoch + 1) * iteration_num + rank
        # iteration_num=3426
        # epoch=14
        # now_random_seed = (epoch+1)*iteration_num+2
        rescaling_model.module.optimizer.zero_grad()
        # training_end, rrv = dis.distributed_data_synchronization(is_training=True, device=device, rank=rank)
        # training_end, rrv = dis.pop_data(is_training=True, device=device)
        # training_end, rrv = dis.asynchronous_data_reading(is_training=True, device=device, random_seed=now_random_seed)
        # frame_number      = training_config.RUVC_GOP*training_config.batch_size
        ## for deal a bitstream writing bug of skvideo
        bitstream_writing_bug_time = 0
        while True:
            rrv    = rrv.to(device)
            HR_rrv = rrv[:,:3,:,:]
            LR_rrv = rrv[:,3:,:,:]
            
            ### forward step1: downsampling
            downsample_video, LF_constraint, HF_constraint = rescaling_model(LR_rrv)
            # HF_constraint *= 1e3
            # downsample_video, _, _ = rescaling_model(LR_rrv)
            LF_video    = downsample_video.narrow(1, 0, downsample_video.shape[1] // 4).detach()
            # HF_video = downsample_video.narrow(1, downsample_video.shape[1] // 4, downsample_video.shape[1] * 3 // 4)
            # LF_smooth_loss                  = rescaling_model.module.LF_constraint(LF_video, LR_dcv)
            
            ### forward setp2: encode and decode
            current_crf          = codec.random_CRF(random_seed=now_random_seed, current_iteration=iteration_num) \
                if training_config.training_CRF==None else training_config.training_CRF
            HR_video             = nn.PixelUnshuffle(2)(HR_rrv).reshape(-1,3,4,HR_rrv.shape[2]//2,HR_rrv.shape[3]//2)
            HR_video             = torch.transpose(HR_video, 1, 2).reshape(-1,3*4,HR_video.shape[-2], HR_video.shape[-1])
            encode_video         = torch.cat([HR_video, LF_video], dim=1)
            encode_video_flatten = encode_video.reshape(-1,3,encode_video.shape[2],encode_video.shape[3])
            
            # constraint_flatten  = LR_rrv.reshape(-1,3,LR_rrv.shape[2],LR_rrv.shape[3])
            encode_video_flatten = quantization_model(encode_video_flatten)
            # constraint_video    = quantization_model(constraint_flatten)
            # decoded_HR_video, _ = codec(HR_video, video_type="HR", crf=current_crf)
            decoded_video, _     = codec(encode_video_flatten, is_train=True, crf=current_crf)
            
            if decoded_video.shape[0] == encode_video_flatten.shape[0]:
                break
            else:
                bitstream_writing_bug_time += 1
                vtp.visualize_4d_tensor(rrv, os.path.join(training_config.intermediatedir,f'bug_sample_{rank}_{epoch}_{iteration_num}_{bitstream_writing_bug_time}.png'))
                rrv = skvideo_ndarray_bitstream_write_debug(rrv, bitstream_writing_bug_time)
                print(f"Warning: [rank={rank}, epoch={epoch}, iteration={iteration_num}] has a bitstream write bug. \nThe problematic samples have been saved to '{os.path.join(training_config.intermediatedir,f'bug_sample_{rank}_{epoch}_{iteration_num}_{bitstream_writing_bug_time}.png')}'. The original image has been flipped and this iteration have been retrained.")
            
        decoded_video       = decoded_video.reshape(-1,3*(4+training_config.RUVC_GOP),decoded_video.shape[-2],decoded_video.shape[-1])
        decoded_HR_video    = decoded_video[:, :3*4, :, :].reshape(-1,4,3,decoded_video.shape[-2],decoded_video.shape[-1])
        decoded_HR_video    = torch.transpose(decoded_HR_video, 1, 2).reshape(-1,12,decoded_HR_video.shape[-2], decoded_HR_video.shape[-1])
        decoded_HR_video    = nn.PixelShuffle(2)(decoded_HR_video).detach()
        decoded_LR_video    = decoded_video[:, 3*4:, :, :].reshape(-1, 3*training_config.RUVC_GOP, decoded_video.shape[2], decoded_video.shape[3]).detach()
        
        ### forward step3: upsampling
        HR_reconstruction, LF_predicted, HF_predicted \
                            = rescaling_model(HR_video=decoded_HR_video, LR_video=decoded_LR_video, reverse=True, quantization_parameter=current_crf / 51.0)
        
                
        ### backward:
        LF_loss = loss_model.LF_loss(LF_predicted, LF_constraint, decoded_LR_video)
        HF_loss = loss_model.HF_loss(HF_predicted, HF_constraint)
        re_loss = loss_model.re_loss(HR_reconstruction, LR_rrv)
        loss    = re_loss + LF_loss + HF_loss
        # loss = re_loss + LF_loss
        # loss = re_loss + LF_loss + HF_loss + haar_loss
        # print(re_loss, LF_loss, HF_loss)
        # loss = LF_loss + HF_loss
        # loss = re_loss
        # sys.exit()
        # loss = HF_loss + 20 * LF_lossgradients = {}

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            rescaling_model.module.gradient_hook_worker(epoch, iteration_num, torch.isnan(loss).any(), torch.isinf(loss).any())
        
        loss.backward()
        
        ### log writer
        if rank == 0 and iteration_num % max(len(dataloader)//5, 200) == 0 and training_config.keep_training_state:
            with open(f"{training_state_log}",'a') as f:
                f.write(f"\tEpoch: {epoch:3d}, Iteration: {iteration_num:5d}, loss: {loss:.6f},\treconstruction_loss: {re_loss:.6f},\tlow_frequency_loss: {LF_loss:.6f},\thigh_frequency_loss: {HF_loss:.6f}\n")
                # f.write(f"\tEpoch: {epoch:3d}, Iteration: {iteration_num:5d}, loss: {loss:.6f},\treconstruction_loss: {re_loss:.6f},\thigh_frequency_loss: {HF_loss:.6f}\n")
        
        # new iteration preparation
        if training_config.use_gradient_hook:
            frequency_fusion_network_original_params = [param.clone() for param in rescaling_model.module.frequency_fusion_network.parameters()]
            frequency_fusion_network_param_names     = [name for name, _ in rescaling_model.module.frequency_fusion_network.named_parameters()]
            quantization_adapter_original_params     = [param.clone() for param in rescaling_model.module.quantization_adapter.parameters()]
            quantization_adapter_param_names         = [name for name, _ in rescaling_model.module.quantization_adapter.named_parameters()]
            HF_fusion_original_params                = [param.clone() for param in rescaling_model.module.HF_fusion.parameters()]
            HF_fusion_param_names                    = [name for name, _ in rescaling_model.module.HF_fusion.named_parameters()]
            
            rescaling_model.module.optimizer.step()
            
            for i, (param, orig_param) in enumerate(zip(rescaling_model.module.frequency_fusion_network.parameters(), frequency_fusion_network_original_params)):
                unchanged = torch.allclose(param, orig_param, atol=1e-8)
                if unchanged:
                    print(f"frequency_fusion_network Parameter {frequency_fusion_network_param_names[i]} did not change after update")
                    
            for i, (param, orig_param) in enumerate(zip(rescaling_model.module.quantization_adapter.parameters(), quantization_adapter_original_params)):
                unchanged = torch.allclose(param, orig_param, atol=1e-8)
                if unchanged:
                    print(f"HF_fusion Parameter {quantization_adapter_param_names[i]} did not change after update")
                    
            for i, (param, orig_param) in enumerate(zip(rescaling_model.module.HF_fusion.parameters(), HF_fusion_original_params)):
                unchanged = torch.allclose(param, orig_param, atol=1e-8)
                if unchanged:
                    print(f"HF_fusion Parameter {HF_fusion_param_names[i]} did not change after update")
                    
        else:
            rescaling_model.module.optimizer.step()
            
        iteration_num += 1
    

def model_eval(
        rescaling_model    :ml.RUVC,                    ## real type: nn.parallel.DistributedDataParallel
        codec              :cm.CodecPure, 
        quantization_model :ml.QuantizationModel, 
        loss_model         :lm.LossModel, 
        dataloader         :dl.EvalDataset,            ## real type: torch.utils.data.Dataloader
        training_config    :cfg.Configuration, 
        device             :torch.device, 
        epoch              :int, 
        rank               :int, 
        training_state_log :str
    ) -> None:
    '''
    Separately encapsulate training and evaluation functions to avoid memory leaks.
        training_end: indicates whether the epoch is over
        rrv         : raw resolution video
        dcv         : downsampled constraint video
        LF and HF   : video (feature map) of low and high frequency
        LR and HR   : video (feature map) of low and high resolution
    '''
    rescaling_model.eval()
    evaluation_crf    = training_config.evaluation_CRF
    iteration_num     = 0
    evaluation_bpp    = 0
    evaluation_loss   = 0
    evaluation_psnr   = 0
    evaluation_lpips  = 0
    evaluation_ssim   = 0
    evaluation_msssim = 0
    eval_LF_psnr      = 0
    eval_HF_psnr      = 0
    with torch.no_grad():
        for rrv in dataloader:
            tmp_evaluation_loss   = 0
            tmp_evaluation_bpp    = 0
            tmp_evaluation_psnr   = 0
            tmp_evaluation_lpips  = 0
            tmp_evaluation_ssim   = 0
            tmp_evaluation_msssim = 0
            tmp_eval_LF_psnr      = 0
            tmp_eval_HF_psnr      = 0
            
            # evaluation_end, rrv      = dataloader.asynchronous_data_reading(is_training=False, device=device)
            frame_number             = (training_config.RUVC_GOP+1)*rrv.shape[0]
            rrv                      = rrv.to(device)
            HR_rrv                   = rrv[:,:3,:,:]
            LR_rrv                   = rrv[:,3:,:,:]
            downsample_video, LF_constraint, HF_constraint \
                                     = rescaling_model(LR_rrv)
            LF_video                 = downsample_video.narrow(1, 0, downsample_video.shape[1] // 4)
            HR_video                 = nn.PixelUnshuffle(2)(HR_rrv).reshape(-1,3, 4, HR_rrv.shape[2]//2,HR_rrv.shape[3]//2)
            HR_video                 = torch.transpose(HR_video, 1, 2).reshape(-1,12,HR_video.shape[-2], HR_video.shape[-1])
            encode_video             = torch.cat([HR_video, LF_video], dim=1)
            encode_video_flatten     = encode_video.reshape(-1,3,encode_video.shape[2],encode_video.shape[3])
            encode_video_flatten     = quantization_model(encode_video_flatten)
            for current_crf in evaluation_crf:
                decoded_video, file_size = codec(encode_video_flatten, is_train=False, crf=current_crf)
                decoded_video            = decoded_video.reshape(-1,3*(4+training_config.RUVC_GOP),decoded_video.shape[-2],decoded_video.shape[-1])
                decoded_HR_video         = decoded_video[:, :3*4, :, :].reshape(-1,4,3,decoded_video.shape[-2],decoded_video.shape[-1])
                decoded_HR_video         = torch.transpose(decoded_HR_video, 1, 2).reshape(-1,4*3,decoded_HR_video.shape[-2], decoded_HR_video.shape[-1])
                decoded_HR_video         = nn.PixelShuffle(2)(decoded_HR_video).detach()
                decoded_LR_video         = decoded_video[:, 3*4:, :, :].reshape(-1, 3*training_config.RUVC_GOP, decoded_video.shape[2], decoded_video.shape[3]).detach()

                HR_reconstruction, LF_predicted, HF_predicted \
                                         = rescaling_model(HR_video=decoded_HR_video, LR_video=decoded_LR_video, reverse=True, quantization_parameter=current_crf / 51.0)
                LF_loss                  = loss_model.LF_loss(LF_predicted, LF_constraint, decoded_LR_video)
                HF_loss                  = loss_model.HF_loss(HF_predicted, HF_constraint)
                reconstruction_loss      = loss_model.re_loss(HR_reconstruction, LR_rrv)
                reconstruction_video     = torch.clamp(torch.cat([decoded_HR_video, HR_reconstruction], dim=1), min=0.0, max=1.0)
                tmp_loss                 = reconstruction_loss + HF_loss + LF_loss
                assert not(torch.isnan(tmp_loss) or torch.isinf(tmp_loss)), f"Epoch {epoch:3d}: evaluation loss is nan or inf."
                tmp_evaluation_loss   += tmp_loss.item()
                tmp_evaluation_bpp    += file_size*8.0/(frame_number*rrv.shape[-2]*rrv.shape[-1])
                tmp_evaluation_psnr   += metrics.compute_metric(reconstruction_video, rrv, metric='PSNR', step_by_step=True)
                tmp_evaluation_lpips  += metrics.compute_metric(reconstruction_video, rrv, metric='LPIPS', step_by_step=True)
                tmp_evaluation_ssim   += metrics.compute_metric(reconstruction_video, rrv, metric='SSIM', step_by_step=True)
                tmp_evaluation_msssim += metrics.compute_metric(reconstruction_video, rrv, metric='MSSSIM', step_by_step=True)
                tmp_eval_LF_psnr      += metrics.compute_metric(LF_predicted, LF_constraint, metric='PSNR', step_by_step=True)
                tmp_eval_HF_psnr      += metrics.compute_metric(HF_predicted, HF_constraint, metric='PSNR', step_by_step=True)
            
            iteration_num     += 1
            evaluation_loss   += tmp_evaluation_loss / len(evaluation_crf)
            evaluation_bpp    += tmp_evaluation_bpp / len(evaluation_crf)
            evaluation_psnr   += tmp_evaluation_psnr / len(evaluation_crf)
            evaluation_lpips  += tmp_evaluation_lpips / len(evaluation_crf)
            evaluation_ssim   += tmp_evaluation_ssim / len(evaluation_crf)
            evaluation_msssim += tmp_evaluation_msssim / len(evaluation_crf)
            eval_LF_psnr      += tmp_eval_LF_psnr / len(evaluation_crf)
            eval_HF_psnr      += tmp_eval_HF_psnr / len(evaluation_crf)
    
            
    ## Logger and Scheduler Work
    evaluation_loss   /= iteration_num
    evaluation_bpp    /= iteration_num
    evaluation_psnr   /= iteration_num
    evaluation_lpips  /= iteration_num
    evaluation_ssim   /= iteration_num
    evaluation_msssim /= iteration_num
    eval_LF_psnr      /= iteration_num
    eval_HF_psnr      /= iteration_num
    learning_rate      = rescaling_model.module.optimizer.param_groups[0]["lr"]
    if rank == 0:
        model_save_path     = os.path.join(training_config.model_save_dir, f'RUVC_model_{epoch}.pt')
        optimizer_save_path = os.path.join(training_config.model_save_dir, f'RUVC_optimizer_{epoch}.pt')
        torch.save(rescaling_model.module.state_dict(), model_save_path)
        torch.save(rescaling_model.module.optimizer.state_dict(), optimizer_save_path)
        
        if training_config.keep_training_state:
            with open(f"{training_state_log}",'a') as f:
                f.write(f"{f'Epoch {epoch:>3d}':<10}[ {f'lr={learning_rate:.1e}':<11}, {f'Loss={evaluation_loss:.6f}':<14} ]:{f' PSNR={evaluation_psnr:.6f}':<16}({f'LF-PSNR={eval_LF_psnr:.6f}':<16}, {f'HF-PSNR={eval_HF_psnr:.6f}':<16}){f' SSIM={evaluation_ssim:.6f}':<16}{f' MS-SSIM={evaluation_msssim:.6f}':<16}{f' LPIPS={evaluation_lpips:.6f}':<16}{f' bpp={evaluation_bpp:.6f}':<16}\n")
            
        print(f"{f'Epoch {epoch:>3d}':<10}[{f' lr={learning_rate:.1e}':<11}, {f'Loss={evaluation_loss:.6f}':<15}]:{f' PSNR={evaluation_psnr:.6f}':<16}( {f'LF-PSNR={eval_LF_psnr:.6f}':<16}, {f'HF-PSNR={eval_HF_psnr:.6f}':<16} ){f' SSIM={evaluation_ssim:.6f}':<16}{f' MS-SSIM={evaluation_msssim:.6f}':<16}{f' LPIPS={evaluation_lpips:.6f}':<16}{f' bpp={evaluation_bpp:.6f}':<16}")

    rescaling_model.module.scheduler.step(evaluation_loss)
    
                    
def skvideo_ndarray_bitstream_write_debug(rrv, bug_time):
    '''
        During our experiments, we found that some images with certain content would cause stream writing exceptions, which resulted in the frame encoding failure and the entire bitstream only compressed the last frame of the video (if there is only a buggy video frame, the entire bitstream has no content). After investigation, we determined that the exception was caused by the underlying function of "ndarray.tostring()". FFmpeg can encode images normally using non-skvideo calls. Therefore, setting an image enhancement when encountering this bug during training can avoid errors, but the problem still exists in theory.
    '''
    if bug_time == 1:
        rrv = torch.flip(rrv, dims=[2])
    elif bug_time == 2:
        rrv = torch.flip(rrv, dims=[2,3])
    elif bug_time == 3:
        rrv = torch.flip(rrv, dims=[2])
    elif bug_time == 4:
        rrv = torch.flip(rrv, dims=[2,3])
        video = torch.rot90(video, k=2, dims=(2, 3))
    elif bug_time == 5:
        rrv = torch.flip(rrv, dims=[2])
    elif bug_time == 6:
        rrv = torch.flip(rrv, dims=[2,3])
    elif bug_time == 7:
        rrv = torch.flip(rrv, dims=[2])
    else:
        raise Exception("I'm sorry to tell you that no matter how this sample is flipped, it will trigger a bug that conflicts with the underlying command after being converted to a bitstream. It is recommended to modify the random number and try again or replace this sample.")
    
    return rrv


if __name__ == '__main__':
    # PARAMETER CONFIGURATION:
    training_config = cfg.Configuration()
    
    # DISTRIBUTED TRAINING:
    parallel_num     = len(training_config.GPU_index)
    distributed_port = ffp.find_free_port()
    if parallel_num >= 1:
        mp.spawn(distributed_train, args=(parallel_num, training_config, distributed_port), nprocs=parallel_num, join=True)

    # FINISH:
    print(f"\n====================>>>{'RUVC Training Completed'.center(25)}<<<====================")
    print(f"{'End':<4}{'Time':<4}{':':<2}{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{'(Y-M-D h:m:s)':>42}\n")