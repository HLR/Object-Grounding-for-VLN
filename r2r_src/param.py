import argparse
import os
import torch


class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")

        # General
        self.parser.add_argument('--iters', type=int, default=80000)
        self.parser.add_argument('--name', type=str, default='default')
        self.parser.add_argument('--train', type=str, default='listener')

        # Data preparationff
        self.parser.add_argument('--maxInput', type=int, default=80, help="max input instruction")
        self.parser.add_argument('--maxDecode', type=int, default=120, help="max input instruction")
        self.parser.add_argument('--maxAction', type=int, default=20, help='Max Action sequence')
        self.parser.add_argument('--batchSize', type=int, default=4) 
        self.parser.add_argument('--ignoreid', type=int, default=-100)
        self.parser.add_argument('--feature_size', type=int, default=2048)
        self.parser.add_argument("--loadOptim",action="store_const", default=False, const=True)

        # Load the model from
        #self.parser.add_argument("--speaker", default="/home/joslin/R2R-EnvDrop/snap/speaker/best_val_unseen_bleu")
        self.parser.add_argument("--speaker", default=None)
        self.parser.add_argument("--listener", default=None)
        #1. 37800
        #20210726-124904
        #20210703-193418
        #20210729-214211.txt
        #self.parser.add_argument("--load", type=str, default='/home/hlr/shared/data/joslin/checkpoints/agent/state_dict/20210726-124904/best_val_unseen')
        #self.parser.add_argument("--load", type=str, default="/home/hlr/shared/data/joslin/checkpoints/agent/state_dict/20210702-220203/Iter_035000")
        self.parser.add_argument("--load", type=str, default=None)
        # More Paths from
        self.parser.add_argument("--aug", default="/home/joslin/R2R-EnvDrop/tasks/R2R/data/aug_paths.json")
        #self.parser.add_argument("--aug", default="/VL/space/zhan1624/selfmonitoring-agent/tasks/R2R-pano/data/R2R_literal_speaker_data_augmentation_paths.json")

        # Listener Model Config
        self.parser.add_argument("--zeroInit", dest='zero_init', action='store_const', default=False, const=True)
        self.parser.add_argument("--mlWeight", dest='ml_weight', type=float, default=0.05)
        self.parser.add_argument("--teacherWeight", dest='teacher_weight', type=float, default=1.)
        self.parser.add_argument("--accumulateGrad", dest='accumulate_grad', action='store_const', default=False, const=True)
        self.parser.add_argument("--features", type=str, default='imagenet')

        # Env Dropout Param
        self.parser.add_argument('--featdropout', type=float, default=0.3)

        # SSL configuration
        self.parser.add_argument("--selfTrain", dest='self_train', action='store_const', default=False, const=True)

        # Submision configuration
        self.parser.add_argument("--candidates", type=int, default=1)
        self.parser.add_argument("--paramSearch", dest='param_search', action='store_const', default=False, const=True)
        self.parser.add_argument("--submit", action='store_const', default=False, const=True)
        self.parser.add_argument("--beam", action="store_const", default=False, const=True)
        self.parser.add_argument("--alpha", type=float, default=0.5)

        # Training Configurations
        self.parser.add_argument('--optim', type=str, default='rms')    # rms, adam
        self.parser.add_argument('--lr', type=float, default=0.0001, help="The learning rate")
        self.parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
        self.parser.add_argument('--dropout', type=float, default=0)
        self.parser.add_argument('--feedback', type=str, default='teacher',
                            help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``')
        self.parser.add_argument('--teacher', type=str, default='final',
                            help="How to get supervision. one of ``next`` and ``final`` ")
        self.parser.add_argument('--epsilon', type=float, default=0.1)

        # Model hyper params:
        self.parser.add_argument('--rnnDim', dest="rnn_dim", type=int, default=512)
        self.parser.add_argument('--wemb', type=int, default=256)
        self.parser.add_argument('--aemb', type=int, default=64)
        self.parser.add_argument('--proj', type=int, default=512)
        self.parser.add_argument("--fast", dest="fast_train", action="store_const", default=False, const=True)
        self.parser.add_argument("--valid", action="store_const", default=False, const=True)
        self.parser.add_argument("--candidate", dest="candidate_mask",
                                 action="store_const", default=False, const=True)

        self.parser.add_argument("--bidir", type=bool, default=False)    # This is not full option
        self.parser.add_argument("--encode", type=str, default="word")  # sub, word, sub_ctx
        self.parser.add_argument("--subout", dest="sub_out", type=str, default="tanh")  # tanh, max
        self.parser.add_argument("--attn", type=str, default="soft")    # soft, mono, shift, dis_shift

        self.parser.add_argument("--angleFeatSize", dest="angle_feat_size", type=int, default=4)

        # A2C
        self.parser.add_argument("--gamma", default=0.9, type=float)
        self.parser.add_argument("--normalize", dest="normalize_loss", default="total", type=str, help='batch or total')


        ###########

        #Spatial Configuration
        self.parser.add_argument("--configuration", default=True, type=bool)
        self.parser.add_argument("--candidate_length", default=15, type=int, help="without considering end situation, if you want to consider end situation, you should add one during candidate viewpoint processing")
        self.parser.add_argument("--text_dimension", default=300, type=int, help="text dimenstion")
        self.parser.add_argument('--configpath', default="/home/hlr/shared/data/joslin/components2/configs/", type=str)

        #152-object_feature_new.npy
        #152-object_feature_relation_18_v2.npy
        #152-object_feature_relation_glove300.npy
        self.parser.add_argument("--using_obj", default=True, type=bool)
        self.parser.add_argument("--obj_img_feat_path1", default='/home/hlr/shared/data/joslin/new_mycsvfile_image_text.npy', type=str)
        #self.parser.add_argument("--obj_img_feat_path", default='/home/hlr/shared/data/joslin/152-object_feature_new_v4.npy', type=str)
        self.parser.add_argument("--obj_text_feat_path2", default='/home/hlr/shared/data/joslin/mycsvfile1.npy', type=str)
        
        self.parser.add_argument("--train_landmark_path", default='/home/hlr/shared/data/joslin/components4/landmarks/landmark_train.npy', type=str)
        self.parser.add_argument("--val_seen_landmark_path", default='/home/hlr/shared/data/joslin/components4/landmarks/landmark_val_seen.npy', type=str)
        self.parser.add_argument("--val_unseen_landmark_path", default='/home/hlr/shared/data/joslin/components4/landmarks/landmark_val_unseen.npy', type=str)
        self.parser.add_argument("--test_landmark_path", default='/home/hlr/shared/data/joslin/components4/landmarks/landmark_test.npy', type=str)

        
        self.parser.add_argument("--train_motion_indi_path", default='/home/hlr/shared/data/joslin/components2/motion_indicator/motion_indicator_train.npy', type=str)
        self.parser.add_argument("--val_seen_motion_indi_path", default='/home/hlr/shared/data/joslin/components2/motion_indicator/motion_indicator_val_seen.npy', type=str)
        self.parser.add_argument("--val_unseen_motion_indi_path", default='/home/hlr/shared/data/joslin/components2/motion_indicator/motion_indicator_val_unseen.npy', type=str)
        self.parser.add_argument("--test_motion_indi_path", default='/home/hlr/shared/data/joslin/components2/motion_indicator/motion_indicator_test.npy', type=str)

        # BERT Encoder
        self.parser.add_argument("--rnn_hidden_size", default=512, type=int)
        self.parser.add_argument("--rnn_dropout", default=0.5, type=float)
        self.parser.add_argument("--bidirectional", default=0, type=int)
        self.parser.add_argument("--rnn_num_layers", default=1, type=int)
        self.parser.add_argument('--word_embedding_size', default=768, type=int,
                    help='default embedding_size for language encoder /256')
    
        
        self.args = self.parser.parse_args()

        if self.args.optim == 'rms':
            print("Optimizer: Using RMSProp")
            self.args.optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            print("Optimizer: Using Adam")
            self.args.optimizer = torch.optim.Adam
        elif self.args.optim == 'sgd':
            print("Optimizer: sgd")
            self.args.optimizer = torch.optim.SGD
        else:
            assert False

param = Param()
args = param.args
args.TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
args.TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'

args.IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
args.CANDIDATE_FEATURES = 'img_features/ResNet-152-candidate.tsv'
args.features_fast = 'img_features/ResNet-152-imagenet-fast.tsv'
args.log_dir = 'snap/%s' % args.name

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
DEBUG_FILE = open(os.path.join('snap', args.name, "debug.log"), 'w')
