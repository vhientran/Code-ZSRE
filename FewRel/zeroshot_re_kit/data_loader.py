import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json


class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """

    def __init__(self, pid2name, encoder, N, root, test_relations=None):
        self.root = root
        # all data for train and test
        path = os.path.join(root, "fewrel_all.json")
        # relation to its name and descriptions
        pid2name_path = os.path.join(root, pid2name + ".json")

        if not os.path.exists(path) or not os.path.exists(pid2name_path):
            print("[ERROR] Data file does not exist!")
            assert 0
        # load dictionary: relation to descriptions
        self.pid2name = json.load(open(pid2name_path))

        # load json data
        json_full_data = json.load(open(path))
        json_train_data = {}
        # remove relations in test set, to prepare the train set
        for rel_ in json_full_data.keys():
            if rel_ not in test_relations:
                # save to train set
                json_train_data[rel_] = json_full_data[rel_]

        self.json_data = json_train_data
        self.classes = list(self.json_data.keys())
        print("\nTrain Relations: ", self.classes, len(self.classes))

        self.N = N
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
                                                       item['h'][2][0],
                                                       item['t'][2][0])
        return word, pos1, pos2, mask

    def __getrel__(self, item):
        word, mask = self.encoder.tokenize_rel(item)
        return word, mask

    def __getname__(self, name):
        word, mask = self.encoder.tokenize_name(name)
        return word, mask

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        relation_set = {'word': [], 'mask': []}
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        support_label = []
        for i, class_name in enumerate(target_classes):
            rel_text, rel_text_mask = self.__getrel__(self.pid2name[class_name])
            rel_text, rel_text_mask = torch.tensor(rel_text).long(), torch.tensor(rel_text_mask).long()
            relation_set['word'].append(rel_text)
            relation_set['mask'].append(rel_text_mask)

            indices = np.random.choice(list(range(len(self.json_data[class_name]))), 1, False)
            for j in indices:
                word, pos1, pos2, mask = self.__getraw__(self.json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                self.__additem__(support_set, word, pos1, pos2, mask)
                
            support_label += [i]

        return support_set, support_label, relation_set

    def __len__(self):
        return 1000000000


def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_label = []
    batch_relation = {'word': [], 'mask': []}
    
    support_sets, support_labels, relation_sets = zip(*data)
    # task i in the batch
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        
        for k in relation_sets[i]:
            batch_relation[k] += relation_sets[i][k]

        batch_label += support_labels[i]

    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)

    for k in batch_relation:
        batch_relation[k] = torch.stack(batch_relation[k], 0)

    batch_label = torch.tensor(batch_label)

    return batch_support, batch_label, batch_relation

# prepare data for training stage
def get_loader(pid2name, encoder, N,
               num_workers=8, collate_fn=collate_fn, root='../official_data',test_relations=None):
    dataset = FewRelDataset(pid2name, encoder, N, root,test_relations)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    
    return iter(data_loader)


# prepare data for test stage
def get_test_loader(pid2name, encoder, batch_size=32, test_relations=None, root='../official_data'):
    pid2name_path = os.path.join(root, pid2name + ".json")
    pid2name = json.load(open(pid2name_path))
    
    # load data
    path = os.path.join(root, "fewrel_all.json")
    json_full_data = json.load(open(path))
    # get only test set
    json_test_data = {}
    for rel_ in json_full_data.keys():
        if rel_ in test_relations:
            json_test_data[rel_] = json_full_data[rel_]

    # evaluate on the test set
    ids_classes = list(json_test_data.keys()) # the number of classes in test set
    print("\nTest Relations: ", ids_classes)
    # process each item
    test_sent_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    test_set_labels = []
    relation_set = {'word': [], 'mask': []} # only the number of classes 

    for i,id_label in enumerate(ids_classes):
        # get the name and relation description of the relation
        rel_text, rel_text_mask = encoder.tokenize_rel(pid2name[id_label])
        relation_set['word'].append(rel_text)
        relation_set['mask'].append(rel_text_mask)
        
        # get the sentences of a relation
        data_id_label = json_test_data[id_label]
        for item in data_id_label:
            word, pos1, pos2, mask = encoder.tokenize(item['tokens'],
                                                       item['h'][2][0],
                                                       item['t'][2][0])
            test_sent_set['word'].append(word)
            test_sent_set['pos1'].append(pos1)
            test_sent_set['pos2'].append(pos2)
            test_sent_set['mask'].append(mask)
            # get relation label 
            test_set_labels.append(i)
    
    # create mini-batches
    all_batches = []
    num_batches = int(len(test_set_labels)/batch_size)
    if len(test_set_labels)%batch_size!=0:
        num_batches = num_batches + 1
    for j in range(num_batches):
        dict_batch = {}
        batch_sents_word = test_sent_set['word'][j*batch_size:(j+1)*batch_size]
        batch_sents_pos1 = test_sent_set['pos1'][j*batch_size:(j+1)*batch_size]
        batch_sents_pos2 = test_sent_set['pos2'][j*batch_size:(j+1)*batch_size]
        batch_sents_mask = test_sent_set['mask'][j*batch_size:(j+1)*batch_size]
        batch_sents_label = test_set_labels[j*batch_size:(j+1)*batch_size]

        assert len(batch_sents_label) > 0

        dict_batch['word'] = torch.tensor(np.array(batch_sents_word))
        dict_batch['pos1'] = torch.tensor(np.array(batch_sents_pos1))
        dict_batch['pos2'] = torch.tensor(np.array(batch_sents_pos2))
        dict_batch['mask'] = torch.tensor(np.array(batch_sents_mask))
        dict_batch['label'] = torch.tensor(np.array(batch_sents_label))
        all_batches.append(dict_batch)

    # batch of relations
    relation_set['word'] = torch.tensor(np.array(relation_set['word']))
    relation_set['mask'] = torch.tensor(np.array(relation_set['mask']))
    
    return all_batches, relation_set
    
            
            

    














