from zeroshot_re_kit.data_loader import get_loader, get_test_loader
from zeroshot_re_kit.framework import ZeroShotREFramework
from zeroshot_re_kit.sentence_encoder import BERTSentenceEncoder
from models.model import Model
import torch
import numpy as np
import json
import argparse
import torch
import random
import time
import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='../official_data',
                        help='file root')
                            
    # the number of unseen relations when testing
    parser.add_argument("-m", "--n_unseen", help="number of unseen classes", type=int, default=15, dest="m")

    # id of the random selection m=15 for splitting data into: training and test sets
    parser.add_argument("-id", "--id_round", help="id of rounds in running 5 times, ids in [1,2,3,4,5].", type=int, default=1, dest="id")

    parser.add_argument('--pid2name', default='pid2name_fewrel',
                        help='pid2name file: relation names and description on fewrel set')

    parser.add_argument('--K', default=5, type=int,
                       help='K pairs of a mini-batch, including K different relations and K corresponding instances.')
    
    parser.add_argument('--train_iter', default=40000, type=int,
                        help='num of iters in training')

    parser.add_argument('--val_step', default=500, type=int,
                        help='val after training how many iters')

    parser.add_argument('--encoder', default='bert',
                        help='encoder: bert')

    parser.add_argument('--max_length', default=128, type=int,
                        help='max length')

    parser.add_argument('--lr', default=2e-5, type=float,
                        help='learning rate')

    parser.add_argument('--weight_decay', default=0.01, type=float,
                        help='weight decay')

    parser.add_argument('--alpha', default=1, type=float,
                        help='rate of kl-loss')

    parser.add_argument('--grad_iter', default=1, type=int,
                        help='accumulate gradient every x iterations')

    parser.add_argument('--optim', default='adamw',
                        help='sgd / adam / adamw')
    parser.add_argument('--hidden_size', default=768, type=int,
                        help='hidden size')
    parser.add_argument('--load_ckpt', default=None,
                        help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
                        help='save ckpt')
    parser.add_argument('--only_test', default=False,
                        help='only test')

    parser.add_argument('--pretrain_ckpt', default='bert-base-uncased',
                        help='bert / roberta pre-trained checkpoint')

    parser.add_argument('--seed', default=20221225, type=int,
                        help='seed')
                        
    parser.add_argument('--path', default=None,
                        help='path to ckpt')

    opt = parser.parse_args()
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    N = opt.K
    encoder_name = opt.encoder
    max_length = opt.max_length
    print('opt: ', opt)

    print('START: ',datetime.datetime.now())
    print("Zero-Shot Relation Extraction")
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))

    # encoder
    sentence_encoder = BERTSentenceEncoder(opt.pretrain_ckpt, max_length, path=opt.path)

    # get TEST RELATIONS
    path_file_splitted = "../official_data/split_train_test_sets/official_data_divisions.json" # ids in [1,2,3,4,5]
    with open(path_file_splitted, 'r') as f:
        file_relations = json.load(f)
    test_relations = file_relations['fewrel'][str(opt.id)][str(opt.m)]
    
    # train data loader
    train_data_loader = get_loader(opt.pid2name, sentence_encoder,
                                   N=N, test_relations=test_relations, root=opt.root)
    
    # data for test
    U_test_data_loader = get_test_loader(opt.pid2name, sentence_encoder,test_relations=test_relations)
    
    framework = ZeroShotREFramework(train_data_loader, U_test_data_loader)

    model = Model(sentence_encoder, hidden_size=opt.hidden_size, max_len=max_length)
    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test:
        T1 = time.clock()
        framework.train(model, N, learning_rate=opt.lr, weight_decay=opt.weight_decay,
                        alpha=opt.alpha, train_iter=opt.train_iter, val_step=opt.val_step,
                        grad_iter=opt.grad_iter)
        T2 = time.clock()
        print('Total training time:%s s' % (T2 - T1))

    print('END: ', datetime.datetime.now())

if __name__ == "__main__":
    main()










