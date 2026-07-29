import os, argparse
import sys, datetime
import torch
import random
import skvideo
import numpy as np

class Configuration():    
    def __init__(self):
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        # basic print
        print(f"\n====================>>>{'Basic Configuration'.center(25)}<<<====================")
        print(f"{'Start':<8}{'Time':<7}{':':<2}{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{'(Y-M-D h:m:s)':>35}")
        print(f"{'python':<8}{'version:':<9}{sys.version}")
        print(f"{'pytorch':<8}{'version:':<9}{torch.__version__}")
        print(f"{'numpy':<8}{'version:':<9}{np.__version__}")
        print(f"{'skvideo':<8}{'version:':<9}{skvideo.__version__}")
        
        # configure capture
        args                               = self.parse_args()
        self.dataset                       = args.dataset
        self.sample_size                   = args.sample_size
        self.external_evalutionset         = args.external_evalutionset  if args.external_evalutionset != "fromDataset" else None
        self.logdir                        = os.path.join('log', args.prefix)
        self.modeldir                      = os.path.join('model', args.prefix)
        self.intermediatedir               = os.path.join('intermediate',args.prefix)
        self.init_model                    = args.init_model
        self.init_optimizer                = args.init_optimizer
        self.optical_flow_model            = args.optical_flow_model
        self.GPU_index                     = list(map(int, args.GPU_index.split(',')))
        self.device                        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate                 = args.learning_rate
        self.weight_decay                  = args.weight_decay
        self.beta1                         = args.beta1
        self.beta2                         = args.beta2
        self.batch_size                    = args.batch_size
        self.eval_batch_size               = args.eval_batch_size
        self.evaluation_CRF                = args.evaluation_CRF
        self.start_epoch                   = args.start_epoch
        self.epoch                         = args.start_epoch + args.epoch
        self.enhance_epoch                 = args.enhance_epoch + self.epoch
        self.num_workers                   = args.num_workers
        self.rescaling_times               = args.rescaling_times
        self.trainingset_split_coefficient = args.trainingset_split_coefficient if self.external_evalutionset == None else 1.0
        self.upscaling_reference_frame     = args.upscaling_reference_frame
        self.RUVC_GOP                      = args.RUVC_GOP
        self.is_print_net                  = args.is_print_net
        self.print_net_path                = args.print_net_path
        self.model_save_dir                = os.path.join(args.model_save_dir, args.prefix)
        self.training_CRF                  = float(args.training_CRF) if args.training_CRF != "dynamic" else None
        self.use_surrogate                 = False if args.use_surrogate == 0 else True
        self.useLiteRUVC                   = False if args.useLiteRUVC == 0 else True
        self.keep_training_state           = False if args.keep_training_state <= 0 else True
        self.use_gradient_hook             = False if args.use_gradient_hook <= 0 else True
        self.random_seed                   = None if args.random_seed == -1 else args.random_seed
        if self.random_seed is not None:
            random.seed(self.random_seed)
        
        # Correctness Check
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        if not os.path.exists(self.modeldir): os.makedirs(self.modeldir)
        if not os.path.exists(self.intermediatedir): os.makedirs(self.intermediatedir)
        if not os.path.exists(self.model_save_dir): os.makedirs(self.model_save_dir)
        assert os.path.exists(self.dataset), print("\033[31mCONFIGRE ERROR\033[0m: Dataset is not exist.")
        assert os.path.exists(self.optical_flow_model), print("CONFIGRE  ERROR: optical flow estimation model is not exist.")
        assert not (self.init_model !='' and (not os.path.exists(self.init_model))), print("CONFIGRE ERROR: Init model is not exist.")
        assert self.batch_size != 0 and self.eval_batch_size != 0, print("CONFIGRE ERROR: Batch size or evaluation batch size is 0.")
        assert self.trainingset_split_coefficient<=1 and self.trainingset_split_coefficient>=0, print("CONFIGRE ERROR: --trainingset_split_coefficient must be a float type between 0 and 1.")
        assert all(0.0 <= x <= 51.0 for x in self.evaluation_CRF), "CONFIGER ERROR: --evaluation_CRF must be a float type between 0 and 51."

                       
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
                    print("\033[93mThe training is using CPU.\033[0m")
                else:
                    try:
                        torch.cuda.set_device(self.GPU_index[0])
                    except Exception as e:
                        self.device = torch.device('cpu')
                        print("Index exceeds valid GPU number. \033[93mThe training is using CPU.\033[0m")
                    else:
                        self.device = torch.device(f'cuda:{self.GPU_index[0]}')
                        print(f"The GPU using now\n\tIndex {torch.cuda.current_device()}: {torch.cuda.get_device_name(0)}")
            else:
                for i in self.GPU_index:
                    if i < 0:
                        print(f"\tThe GPU with index {i} is not available.")
                        self.GPU_index = list(filter(lambda x: x != i, self.GPU_index))
                    else:
                        try:
                            torch.cuda.set_device(i)
                        except Exception as e:
                            print(f"\tThe GPU with index {i} is not available.")
                            self.GPU_index = list(filter(lambda x: x != i, self.GPU_index))
                
                if self.GPU_index!=[]:
                    print(f"The list of GPU currently used is")
                    for i in self.GPU_index:
                        print(f"\tIndex {i}: {torch.cuda.get_device_name(i)}")
                else:
                    self.device = torch.device('cpu')
                    print("\033[93mThe training is using CPU.\033[0m")
        else:
            print("\033[93mThe training is using CPU.\033[0m")
            
    def parse_args(self):
        parser    = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        main_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

        parser.add_argument("--dataset", type=str, default=os.path.join(main_path,'dataset','trainingset_rawResolution'), help="Full path of the folder where the training data, the data should be jpg images.")
        parser.add_argument("--sample_size", type=int, default=0, help="The square crop size of the sample, <=0 is not cropped.")
        parser.add_argument("--external_evalutionset", type=str, default="fromDataset", help="The data path of the verification phase is cut from the dataset parameter by default.")
        parser.add_argument("--rescaling_times", type=int, default=2, help="Determines the number of downsampling modules and the scaling times. Usually set 2/4.")
        parser.add_argument("--prefix", type=str, default='RUVC_training', help="Prefix of checkpoints/logger, etc.")
        
        parser.add_argument("--init_model", type=str, default='')
        parser.add_argument("--init_optimizer", type=str, default='')
        parser.add_argument("--optical_flow_model", type=str, default=os.path.join(main_path, 'dependencies', 'optical_flow_modules', 'fastflownet_ft_mix.pth'), help="The path of optical flow estimation model uses the fast flow network as default. Ensure that the models used for training and testing are consistent.")
        # parser.add_argument("--optical_flow_model", type=str, default=os.path.join(main_path, 'dependencies', 'optical_flow_modules', 'fastflownet_ft_mix.pth'), help="The path of optical flow estimation model uses the fast flow network as default. Ensure that the models used for training and testing are consistent.")
        # parser.add_argument("--optical_flow_model", type=str, default=os.path.join(main_path, 'dependencies', 'optical_flow_modules', 'fastflownet_ft_mix.pth'), help="The path of optical flow estimation model uses the fast flow network as default. Ensure that the models used for training and testing are consistent.")
        parser.add_argument("--GPU_index", type=str, default='0,1,2,3', help="Give training program a useful GPU index, or it will use CPU to train like when GPU_index is '\-1'. If multiple GPU are required in parallel, separate the serial numbers with commas ',' like '0,1,2,3'. ")
        parser.add_argument("--random_seed",type=int, default=3, help="To replicate the effects of the author's experiment, ensure that the random number seed is the default value, if -1 is used, it is globally random.")
        
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-12)
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.5)
        parser.add_argument("--batch_size", type=int, default=8, help="The default batch size is set to 8, which is to read videos of one resolution at a time. If setting a large batch is necessary, the resolution of each video needs to be the same.")
        parser.add_argument("--eval_batch_size", type=int, default=50, help="The default eval batch size is set to 50, which is to read videos of one resolution at a time. If setting a large batch is necessary, the resolution of each video needs to be the same. Note that please ensure that the evaluation batch size can be divided evenly by the number of evaluated samples. Otherwise, the evaluation results will be biased, which may lead to the failure of the learning rate decay strategy during training.")
        parser.add_argument("--num_workers", type=int, default=2)
        parser.add_argument("--epoch", type=int, default=50)
        parser.add_argument("--start_epoch", type=int, default=0)
        parser.add_argument("--enhance_epoch", type=int, default=0)
        parser.add_argument("--trainingset_split_coefficient", type=float, default=0.8, help="The ratio of the training set to the entire data set, that is, what percentage of the data set will be used as the training set, and the rest will be used as the evalution set.")
    
        parser.add_argument("--RUVC_GOP", type=int, default=6, help="Set the number of frames contained in the video sample during training. The default value is 5.")
        
        parser.add_argument("--is_print_net", type=bool, default=False, help="True to print model network, false do not.")
        parser.add_argument("--print_net_path", type=str, default=os.path.join(main_path, 'Model Network.png'), help="The save path of visual network model.")
        parser.add_argument("--model_save_dir", type=str, default=os.path.join(main_path, 'model'), help="The save directory of training model.")
        
        parser.add_argument("--training_CRF", type=str, default='dynamic', help="The codec crf size used for training, which defaults to dynamic crf training")
        parser.add_argument("--evaluation_CRF", nargs='+', type=float, default=[33.,28.,23.,18.] , help="Specify the encoder quantization parameters used for evaluation. The default is [33.,28.,23.,18.], and the parameter input paradigm is: --evaluation_CRF 33 28 23 18 --other_parameters.")
        parser.add_argument("--use_surrogate", type=int, default=0, help="Gradient surrogate network switch. 0 indicates that the surrogate is disabled and others indicates that the proxy is enabled.")
        parser.add_argument("--useLiteRUVC", type=int, default=0, help="Use LiteRUVC version. 0 indicates that basic RUVC is used and others indicates that LiteRUVC is used.")
        parser.add_argument("--keep_training_state", type=int, default=1, help="Record the status of RUVC training for timely debugging. Default 1 is open, 0 is close. This only affects the generation of file 'log/[prefix]/training_state.txt', not terminal.")
        parser.add_argument("--use_gradient_hook", type=int, default=0, help="It is used to track the gradient of each layer for debugging when the gradient disappears or explodes, but it incurs additional video memory overhead in each epoch and the training process will be slower. The value <= 0 means close, else use gradient hook, default 0.")
        
        parser.add_argument("--upscaling_reference_frame", type=int, default=0, help="Reference frame number during up-sampling, I frame is recommended.")
        
        
        return parser.parse_args()