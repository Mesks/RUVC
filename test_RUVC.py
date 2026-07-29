import os
import csv
import sys, datetime
import time
import torch
import torch.nn.functional as F

from test_modules import configure as cfg
from test_modules import data_load as dl
from test_modules import model_load as ml
from test_modules import codec_module as cm

# from auxiliary_modules import compute_Tmap2HF as Tmap2HF
from auxiliary_modules import compute_metrics as compute_metrics
from auxiliary_modules import video_tensor_processor as vtp


if __name__ == '__main__':
    with torch.no_grad():
        ## Preperation stage
        root_dir           = os.path.dirname(os.path.abspath(__file__))
        test_config        = cfg.Configuration(root_dir)
        test_video         = dl.TestVideoLoading(test_config)
        video_adjuster     = dl.TestVideoProcessing(test_config, test_video=test_video)
        rescaling_model    = ml.RUVC(test_config).to(test_config.device)
        quantization_model = ml.QuantizationModel().to(test_config.device)
        codec              = cm.CodecWithSurrogate(test_config) if test_config.use_surrogate else cm.CodecPure(test_config)
        test_psnr          = 0
        test_ssim          = 0
        test_msssim        = 0
        test_lpips         = 0
        test_bpp           = 0
        test_bitrate       = 0
        # test_frame_number  = test_video.frame_number
        # raw_video          = test_video.get_video()
        # pad_number         = ((test_GOP+1)-test_video.frame_number%(test_GOP+1))%(test_GOP + 1)
        # if pad_number > 0:
        #     padding          = torch.zeros(pad_number, 3, test_video.video_height, test_video.video_width)
        #     padded_raw_video = torch.cat([raw_video,padding],dim=0).reshape(-1,3*(test_GOP+1),test_video.video_height,test_video.video_width)
        # else:
        #     padded_raw_video = raw_video.reshape(-1,3*(test_GOP+1),test_video.video_height,test_video.video_width)
        # raw_HR_video       = padded_raw_video[:,:3,:,:].reshape(-1,3,test_video.video_height,test_video.video_width)
        # raw_LR_video       = padded_raw_video[:,3:,:,:].reshape(-1,3*test_config.RUVC_GOP,test_video.video_height,test_video.video_width)
        
        to_reference_video, to_downsample_video = test_video.get_HR_LR_video()
        # print("to_reference_video.shape: ", to_reference_video.shape, "to_downsample_video.shape: ", to_downsample_video.shape)
        to_reference_video, to_downsample_video = video_adjuster.pad_before_downscale(to_reference_video, to_downsample_video)
        # print("to_reference_video.shape: ", to_reference_video.shape, "to_downsample_video.shape: ", to_downsample_video.shape)
        
        time_start = time.time()
        
        ## Dowscaling stage
        print(f"{'Downscaling...':<21}", end="")
        T1 = time.time()
        downsample_video   = rescaling_model(to_downsample_video)
        # print("downsample_video.shape: ", downsample_video.shape, "downsample_video.shape: ", downsample_video.shape)
        # print(f"Finished. Consumed {time.time() - T1:.6f} seconds")
        LF_video, HF_video = test_video.frequency_detach(downsample_video)
        # print("LF_video.shape: ", LF_video.shape, "HF_video.shape: ", HF_video.shape)
        quantization_video = quantization_model(LF_video.reshape(-1,3,LF_video.shape[-2],LF_video.shape[-1]))
        # print("quantization_video.shape: ", quantization_video.shape)
        encode_video       = video_adjuster.unpad_after_downscale(to_reference_video, quantization_video)
        # print("encode_video.shape: ", encode_video.shape)
        
        ## Encode&Decode stage
        decode_video, file_size, crf_value = codec(encode_video)
        # print("decode_video.shape: ", decode_video.shape)
        padded_reference, padded_LF_video  = video_adjuster.pad_after_decode(decode_video)
        # print("padded_reference.shape: ", padded_reference.shape, "padded_LF_video.shape: ", padded_LF_video.shape)
        
        
        ## Upscaling stage
        print(f"{'Upscaling...':<21}", end="")
        T1                   = time.time()
        HR_reconstruction, LF_RE, predicted_HF \
                             = rescaling_model(HR_video=padded_reference, LR_video=padded_LF_video, reverse=True, quantization_parameter=crf_value / 51.0)
        # print("HR_reconstruction.shape: ", HR_reconstruction.shape, "LF_RE.shape: ", LF_RE.shape, "predicted_HF.shape: ", predicted_HF.shape)
        print(f"Finished. Consumed {time.time() - T1:.6f} seconds")
        reconstruction_video = video_adjuster.unpad_after_upscale(padded_reference, HR_reconstruction)
        time_end = time.time()
        
        ## Metric computation stage
        '''
            There will be slight differences between the two metrics calculation methods, which is caused by the carry precision.
            When making a fair comparison, please play attention to check whether the metrics calculation is aligned .
        '''
        torch.cuda.empty_cache()
        print(f"{'Metrics computing...':<21}", end="")
        test_bpp     = file_size * 8.0 / (test_video.frame_number * test_video.video_height * test_video.video_width)
        test_bitrate = file_size * 8.0 / (test_video.frame_number / 25 * 1000)
        test_runtime = time_end - time_start
        raw_video    = test_video.get_video().reshape(-1,3,test_video.video_height,test_video.video_width)
        if test_config.measure_step_by_step:
            print("Each frames details:")
            for i in range(test_video.frame_number):
                temp_PSNR   = compute_metrics.compute_metric(reconstruction_video[i], raw_video[i], metric='PSNR', step_by_step=True, device=test_config.device)
                temp_SSIM   = compute_metrics.compute_metric(reconstruction_video[i], raw_video[i], metric='SSIM', step_by_step=True, device=test_config.device)
                temp_MSSSIM = compute_metrics.compute_metric(reconstruction_video[i], raw_video[i], metric='MSSSIM', step_by_step=True, device=test_config.device)
                temp_LPIPS  = compute_metrics.compute_metric(reconstruction_video[i], raw_video[i], metric='LPIPS', step_by_step=True, device=test_config.device)
                print(f"\tFrame index: {i:3d}, PSNR: {temp_PSNR:.6f}, SSIM: {temp_SSIM:.6f}, MSSSIM: {temp_MSSSIM:.6f}, LPIPS: {temp_LPIPS:.6f}.")
                test_psnr   += temp_PSNR
                test_ssim   += temp_SSIM
                test_msssim += temp_MSSSIM
                test_lpips  += temp_LPIPS
            test_psnr   /= test_video.frame_number
            test_ssim   /= test_video.frame_number
            test_msssim /= test_video.frame_number
            test_lpips  /= test_video.frame_number
        else:
            test_psnr   = compute_metrics.compute_metric(reconstruction_video, raw_video, metric='PSNR', step_by_step=True, device=test_config.device)
            test_ssim   = compute_metrics.compute_metric(reconstruction_video, raw_video, metric='SSIM', step_by_step=True, device=test_config.device)
            test_msssim = compute_metrics.compute_metric(reconstruction_video, raw_video, metric='MSSSIM', step_by_step=True, device=test_config.device)
            test_lpips  = compute_metrics.compute_metric(reconstruction_video, raw_video, metric='LPIPS', step_by_step=True, device=test_config.device)
            print("Finished. ")
        
        if test_config.show_each_frame:
            for i in range(test_video.frame_number):
                vtp.visualize_4d_tensor(reconstruction_video[i], test_config.intermediatedir + f'/reconstruction_video_{i}.png', need_normalize=False)
            # vtp.visualize_4d_tensor(reconstruction_video, test_config.intermediatedir + f'/reconstruction_video.png', need_normalize=False)
            print(f"\t{test_video.frame_number} reconstruction frames saved in '{test_config.intermediatedir}'.")
        
    ## Finish
    print(f"\n====================>>>{'RUVC Test Completed'.center(25)}<<<====================")
    print(f"{'End':<6}{'Time':<5}{':':<2}{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{'(Y-M-D h:m:s)':>39}")
    print(f"Total Video: (CRF: {test_config.x265_CRF})\n\t{'PSNR':<8}= {test_psnr:.6f}\n\t{'MSSSIM':<8}= {test_msssim:.6f}\n\t{'SSIM':<8}= {test_ssim:.6f}\n\t{'LPIPS':<8}= {test_lpips:.6f}\n\t{'bpp':<8}= {test_bpp:.6f}\n\t{'bitrate':<8}= {test_bitrate:.6f}\n\t{'runtime':<8}= {test_runtime:.6f}\n")
    with open(test_config.result_path, 'a') as csvfile:
        writer = csvfile.write(f"{test_config.testdata.split('/')[-1]},{test_config.x265_CRF},{test_psnr:.6f},{test_msssim:.6f},{test_ssim:.6f},{test_lpips:.6f},{test_bpp:.6f},{test_bitrate:.6f},{test_runtime:.6f},\n")
