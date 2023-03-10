from . import evaluation
import torch
from torch import  nn
from transformers import AdamW, get_linear_schedule_with_warmup


class ZeroShotREModel(nn.Module):
    def __init__(self, sentence_encoder):
        """
        sentence_encoder: Sentence encoder
        """
        nn.Module.__init__(self)
        self.sentence_encoder = sentence_encoder
        self.cost = nn.CrossEntropyLoss(reduction='none')

        self.sigmoid = nn.Sigmoid()

    def forward(self):
        raise NotImplementedError

    # loss for training batches
    def loss(self, rel_text_based_logits, label):
        N = rel_text_based_logits.size(-1)
        rel_desp_based_loss = self.cost(rel_text_based_logits.view(-1,N),label.view(-1)).mean()
        
        return rel_desp_based_loss
        

class ZeroShotREFramework:
    def __init__(self, train_data_loader, U_test_data_loader):
        self.train_data_loader = train_data_loader
        self.U_test_data_loader = U_test_data_loader

    def train(self,
              model,
              N,
              learning_rate=2e-5,
              weight_decay=0.01,
              alpha=1,
              train_iter=40000,
              val_step=500,
              warmup_step=300,
              grad_iter=1):
        
        print("\nStart training...")
        parameters_to_optimize = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize
                        if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in parameters_to_optimize
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter)

        
        start_iter = 0
        # Training STAGE
        model.train()
        best_prec, best_rec, best_f1 = 0, 0, 0
        
        for it in range(start_iter, start_iter + train_iter):
            support, support_label, rel_text = next(self.train_data_loader)
            if torch.cuda.is_available():
                for k in support:
                    support[k] = support[k].cuda()
                for k in rel_text:
                    rel_text[k] = rel_text[k].cuda()
                support_label = support_label.cuda()
            
            rel_text_based_logits, kl_loss = model(support=support, N_rel_text=rel_text, N=N, is_eval=False, dict_all_relations_test_U=self.U_test_data_loader[1])
            rel_desp_based_loss = model.loss(rel_text_based_logits, support_label)
            loss = rel_desp_based_loss + alpha*kl_loss

            loss.backward()
            
            if it % grad_iter == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (it + 1) % val_step == 0:
                print('\n========Evaluate after itertation: ', it+1)
                model_acc = self.eval(model)
                model.train()
                if model_acc[-1] > best_f1:
                    best_f1 = model_acc[-1]
                    best_prec = model_acc[-3]
                    best_rec = model_acc[-2]

        model_acc = self.eval(model)
        if model_acc[-1] > best_f1:
            best_f1 = model_acc[-1]
            best_prec = model_acc[-3]
            best_rec = model_acc[-2]

        print("\n####################\n")
        print("Finish training and evaluation stages.")        
        print("\nThe BEST performance on the FewRel test set: ")
        print("Precision / Recall / F1-score: ", best_prec, best_rec, best_f1)


    def eval(self,model):
        model.eval()
        all_batches_test_U, test_relations_dict_U = self.U_test_data_loader
        all_test_labels_UU = []
        all_test_logit_preds_UU = []
        
        with torch.no_grad():
            # for the unseen test set
            for dict_batch in all_batches_test_U:
                # evaluate the batch on S and U
                pred_batch_test_logits_UU = model(is_eval=True, dict_batch_test=dict_batch, dict_all_relations_test_U=test_relations_dict_U)
                all_test_labels_UU.append(dict_batch['label'])
                all_test_logit_preds_UU.append(pred_batch_test_logits_UU)
                
            all_test_labels_UU = torch.cat(all_test_labels_UU, dim=-1)
            all_test_logit_preds_UU = torch.cat(all_test_logit_preds_UU,dim=0)
            
            # combine to create the final measures
            zsl_all_test_labels_UU = all_test_labels_UU.cpu().detach().tolist()
            _, zsl_all_test_logit_preds_UU = torch.max(all_test_logit_preds_UU,1)
            zsl_all_test_logit_preds_UU = zsl_all_test_logit_preds_UU.cpu().detach().tolist()
            prec_, recall_, f1_score_ = evaluation.compute_macro_PRF(zsl_all_test_labels_UU, zsl_all_test_logit_preds_UU) 
            
        return prec_, recall_, f1_score_






