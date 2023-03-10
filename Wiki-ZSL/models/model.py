import sys

sys.path.append('..')
import zeroshot_re_kit
import torch
from torch import nn


class Model(zeroshot_re_kit.framework.ZeroShotREModel):

    def __init__(self, sentence_encoder, hidden_size, max_len):
        zeroshot_re_kit.framework.ZeroShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.rel_glo_linear = nn.Linear(hidden_size, hidden_size * 2)

    def __dist__(self, x, y, dim):
        return (x * y).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, support=None, N_rel_text=None, N=None, is_eval=False, dict_batch_test=None, dict_all_relations_test_U=None):
        if is_eval:
            # we input sentences and relation descriptions
            # prepare data with batch_size
            for key_ in dict_batch_test.keys():
                dict_batch_test[key_].cuda()
            # all unseen relations: U
            for key__ in dict_all_relations_test_U.keys():
                dict_all_relations_test_U[key__].cuda()
                
            test_sent_reps = self.sentence_encoder(dict_batch_test) # (bs, 2D)
            
            # EVALUATE ON UNSEEN RELATIONS
            rel_desp_reps_U = self.sentence_encoder(dict_all_relations_test_U,cat=False) # (num_rel, D)
            test_rel_text_glob_2D_U = self.rel_glo_linear(rel_desp_reps_U) # (num_rel,2D)
            test_logits_U = torch.matmul(test_sent_reps, test_rel_text_glob_2D_U.t())
            
            return test_logits_U
            
        support_glo = self.sentence_encoder(support)
        rel_text_glo = self.sentence_encoder(N_rel_text, cat=False)
        support_glo = support_glo.view(-1, N, self.hidden_size * 2)
        rel_text_glo = rel_text_glo.view(-1, N, self.hidden_size)

        # loss_1: logits based on relation descp
        rel_text_glob_2D = self.rel_glo_linear(rel_text_glo) # (B,N,2D)
        rel_text_based_logits = self.__batch_dist__(rel_text_glob_2D, support_glo)
        minn, _ = rel_text_based_logits.min(-1)
        rel_text_based_logits_bonus = torch.cat([rel_text_based_logits, minn.unsqueeze(2) - 1], 2)

        # loss_2: KL-Divergence between 
        A = rel_text_glob_2D.squeeze(0)
        B = support_glo.squeeze(0)
        
        prob_dis_AB = torch.matmul(A,B.t())
        prob_dis_BA = torch.matmul(B,A.t())

        f_softmax = torch.nn.Softmax(dim=1)

        # softmax
        prob_dis_AB = f_softmax(prob_dis_AB) # rel -> sent
        prob_dis_BA = f_softmax(prob_dis_BA) # sent -> rel

        # KL-Divergence loss between these two prob distributions:
        # KL(prob_dis_AB || prob_dis_BA)
        prob_dis_BA = torch.log(prob_dis_BA)
        g =  torch.nn.KLDivLoss(reduction='batchmean')
        kl_loss = g(prob_dis_BA,prob_dis_AB)
        
        return rel_text_based_logits_bonus, kl_loss

        
        


















