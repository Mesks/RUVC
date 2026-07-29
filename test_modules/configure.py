import os, argparse
import sys, datetime
import torch
import random
import skvideo
import numpy as np

from auxiliary_modules import ffmpeg_help as fh

class Configuration():    
    def __init__(self, root_dir):
        # basic print        
        print(f"\n====================>>>{'Basic Configuration'.center(25)}<<<====================")
        print(f"{'Start':<8}{'Time':<7}{':':<2}{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{'(Y-M-D h:m:s)':>35}")
        print(f"{'python':<8}{'version:':<9}{sys.version}")
        print(f"{'pytorch':<8}{'version:':<9}{torch.__version__}")
        print(f"{'numpy':<8}{'version:':<9}{np.__version__}")
        print(f"{'skvideo':<8}{'version:':<9}{skvideo.__version__}")
    
        # configure capture
        args                           = self.parse_args()
        self.testdata                  = args.testdata
        self.logdir                    = os.path.join(root_dir, 'log', args.prefix)
        self.intermediatedir           = os.path.join(root_dir, 'intermediate', args.prefix)
        self.model                     = args.model
        self.optical_flow_model        = args.optical_flow_model
        self.device                    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.GPU_index                 = list(map(int, args.GPU_index.split(',')))
        # self.GPU_memory_limitation     = args.GPU_memory_limitation
        self.RUVC_GOP                  = args.RUVC_GOP
        self.reference_step            = args.reference_step
        self.rescaling_times           = args.rescaling_times
        self.upscaling_reference_frame = args.upscaling_reference_frame
        self.frame_number              = args.frame_number
        self.is_print_net              = args.is_print_net
        self.print_net_path            = args.print_net_path
        self.show_each_frame           = False if args.show_each_frame <= 0 else True
        self.measure_step_by_step      = False if args.measure_step_by_step <= 0 else True
        self.codec                     = args.codec
        self.x265_CRF                  = float(args.x265_CRF) if args.x265_CRF != "dynamic" else random.randint(18.0, 33.0)
        self.use_surrogate             = False if args.use_surrogate == 0 else True
        self.useLiteRUVC               = False if args.useLiteRUVC == 0 else True
        self.GOPbyGOP                  = False if args.GOPbyGOP <=0 else True
        self.FLOPs_statistics          = False if args.FLOPs_statistics <=0 else True
        self.random_seed               = None if args.random_seed == -1 else args.random_seed
        if self.random_seed is not None:
            random.seed(self.random_seed)
        
        # path and file check
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        if not os.path.exists(self.intermediatedir): os.makedirs(self.intermediatedir)
        assert os.path.exists(self.testdata), print("\033[31mCONFIGRE ERROR\033[0m: Dataset is not exist.")
        assert os.path.exists(self.model), print("CONFIGRE  ERROR: RUVC model is not exist.")
        assert os.path.exists(self.optical_flow_model), print("CONFIGRE  ERROR: optical flow estimation (RAFT) model is not exist.")
                                              
        # GPU setting
        if self.device.type=='cuda':
            print(f"{'CUDA':<8}{'version:':<9}{torch.version.cuda}")
            if torch.backends.cudnn.enabled == True:
                torch.backends.cudnn.benchmark = True
                print(f"{'CUDNN':<8}{'version:':<9}{torch.backends.cudnn.version()}")
            else:
                print(f"{'CUDNN':<8}{'version:':<9}unabled.")
                
            if len(self.GPU_index) == 1:
                if self.GPU_index[0] < 0:
                    self.device = torch.device('cpu')
                    print("\033[93mThe test is using CPU.\033[0m")
                else:
                    try:
                        torch.cuda.set_device(self.GPU_index[0])
                        # torch.cuda.set_per_process_memory_fraction(self.GPU_memory_limitation, device=self.GPU_index[0])
                    except Exception as e:
                        self.device = torch.device('cpu')
                        print("Index exceeds valid GPU number. \033[93mThe test is using CPU.\033[0m")
                    else:
                        self.device = torch.device(f'cuda:{self.GPU_index[0]}')
                        print(f"The list of GPU currently used is\n\tIndex {torch.cuda.current_device()}: {torch.cuda.get_device_name(0)}")
            else:
                for i in self.GPU_index:
                    if i < 0:
                        print(f"The GPU with index {i} is not available.")
                        self.GPU_index = list(filter(lambda x: x != i, self.GPU_index))
                    else:
                        try:
                            torch.cuda.set_device(i)
                        except Exception as e:
                            print(f"The GPU with index {i} is not available.")
                            self.GPU_index = list(filter(lambda x: x != i, self.GPU_index))
                
                if self.GPU_index!=[]:
                    print(f"The list of GPU currently used is")
                    for i in self.GPU_index:
                        print(f"\tIndex {i}: {torch.cuda.get_device_name(i)}")
                else:
                    self.device = torch.device('cpu')
                    print("\033[93mThe test is using CPU.\033[0m")
        else:
            print("\033[93mThe test is using CPU.\033[0m")
            
        # log setting
        result_name = 'result_vvenc.csv' if self.codec == 'vvenc' else 'result.csv'
        quality_label = 'QP' if self.codec == 'vvenc' else 'CRF'
        self.result_path = os.path.join(self.logdir, result_name)
        if not os.path.exists(self.result_path):
            with open(self.result_path, 'w') as csvfile:
                csvfile.write(f'Sequence,{quality_label},PSNR,MS-SSIM,SSIM,LPIPS,bpp,bitrate,runtime\n')
        else:
            print(f"\033[93mNote that:\033[0m Log file '{self.result_path}' already exists, \n{' ':<11}new results will be appended in the tile of the log file.")
            
    def parse_args(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        main_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

        parser.add_argument("--testdata", type=str, default=os.path.join(main_path,'dataset','testset'), help="Full path of the folder where the test data, the data should be jpg images.")
        parser.add_argument("--model", type=str, default="RUVC model absolute path for testing.")
        parser.add_argument("--prefix", type=str, default='RUVC_test', help="Prefix of checkpoints/logger, etc.")
        parser.add_argument("--GPU_index", type=str, default='0', help="Specifies the GPU index to use.")
        # parser.add_argument("--GPU_memory_limitation", type=str, default='1', help="Limit GPU efficiency. All video memory is used by default.")
        parser.add_argument("--random_seed",type=int, default=3, help="To replicate the effects of the author's experiment, ensure that the random number seed is the default value, if -1 is used, it is globally random.")
        
        parser.add_argument("--rescaling_times", type=int, default=2, help="We only research 2x.")
        parser.add_argument("--RUVC_GOP", type=int, default=6, help="Set the number of frames contained in the video sample during training. The default value is 6.")
        parser.add_argument("--reference_step", type=int, default=1, help="Used to specify the number of GOP that a reference frame will use for subsequent reference.")
        parser.add_argument("--upscaling_reference_frame", type=int, default=0, help="Reference frame number during up-sampling, I frame is recommended.")
        parser.add_argument("--frame_number", type=int, default=-1, help="Used frame number in test video.")        
        
        parser.add_argument("--is_print_net", type=bool, default=False, help="True to print model network, false do not.")
        parser.add_argument("--print_net_path", type=str, default=os.path.join(main_path, 'Model Network.png'), help="The save path of visual network model.")
        parser.add_argument("--measure_step_by_step", type=int, default=1, help="Control whether to use frame by frame evaluation indicators and finally calculate the average method, SOTA uses this method.")
        parser.add_argument("--show_each_frame", type=int, default=1, help="Controls whether to print detailed metrics for each frame, 0 indicates off others indicate on. When turned on, each reconstructed frame is saved in 'intermediate/[--prefix]'.")
        parser.add_argument("--FLOPs_statistics", type=int, default=0, help="Turning this on will calculate the FLOPs of the RUVC network model. Note that this process will be very slow. Less than or equal to 0 to disable, greater than 0 to enable.")
        
        parser.add_argument("--x265_CRF", type=str, default='dynamic', help="CRF for x265; QP for vvenc.")
        parser.add_argument("--codec", type=str, choices=['x265', 'vvenc'], default='x265', help="Codec backend used when the surrogate is disabled.")
        parser.add_argument("--use_surrogate", type=int, default=1, help="Gradient surrogate network switch. 0 indicates that the surrogate is disabled and others indicates that the proxy is enabled.")
        parser.add_argument("--useLiteRUVC", type=int, default=0, help="Use LiteRUVC version. 0 indicates that basic RUVC is used and others indicates that LiteRUVC is used.")
        
        parser.add_argument("--GOPbyGOP", type=int, default=1, help="Used to help test large-resolution videos. When rescaling, the data will be cut into batch_size GOPs for GPU calculation. Enabled when > 0, and disabled when <= 0, the default value is 1.")
        parser.add_argument("--optical_flow_model", type=str, default=os.path.join(main_path, 'dependencies', 'optical_flow_modules', 'fastflownet_ft_mix.pth'), help="The path of optical flow estimation model uses the fast flow network as default. Ensure that the models used for training and testing are consistent.")
        # parser.add_argument("--optical_flow_model", type=str, default=os.path.join(main_path, 'dependencies', 'optical_flow_modules', 'raft_kitti.pth'), help="The path of optical flow estimation model uses the fast flow network as default. Ensure that the models used for training and testing are consistent.")
        # parser.add_argument("--optical_flow_model", type=str, default=os.path.join(main_path, 'dependencies', 'optical_flow_modules', 'raft_small.pth'), help="The path of optical flow estimation model uses the fast flow network as default. Ensure that the models used for training and testing are consistent.")

        return parser.parse_args()