"""
"""

import os
import pandas as pd
import torch
import modules.utils as utils
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from itertools import chain
from transformers import ElectraModel, ElectraTokenizer
from transformers import FunnelTokenizerFast, FunnelModel
from transformers import BertTokenizerFast, BertModel


class ElectraCustomDataset(Dataset):
    def __init__(self, data_dir, mode):
        self.data_dir = data_dir
        self.mode = mode
        self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
        self.inputs, self.labels = self.data_loader()

    def data_loader(self):
        print('Loading ' + self.mode + ' dataset..')
        if os.path.isfile(os.path.join(self.data_dir, self.mode, self.mode + '_X.pt')):
            inputs = torch.load(os.path.join(self.data_dir, self.mode, self.mode + '_X.pt'))
            labels = torch.load(os.path.join(self.data_dir, self.mode, self.mode + '_Y.pt'))

        else:
            file_path = os.path.join(self.data_dir, self.mode, self.mode + '.json')
            df = utils.load_json(file_path)
            inputs = pd.DataFrame(columns=['src'])
            labels = pd.DataFrame(columns=['trg'])
            inputs['src'] =  df['article_original']
            labels['trg'] =  df['extractive']
            # Preprocessing
            inputs, labels = self.preprocessing(inputs, labels)
            # Save data
            torch.save(inputs ,os.path.join(self.data_dir, self.mode, self.mode + '_X.pt'))
            torch.save(labels, os.path.join(self.data_dir, self.mode, self.mode + '_Y.pt'))

        inputs = inputs.values
        labels = labels.values

        return inputs, labels

    def pad(self, data, pad_id, max_len):
        padded_data = data.map(lambda x : torch.cat([x, torch.tensor([pad_id] * (max_len - len(x)))]))
        return padded_data

    def preprocessing(self, inputs, labels):
        print('Preprocessing ' + self.mode + ' dataset..')

        #Encoding original text
        inputs['src'] = inputs['src'].map(lambda x: torch.tensor(list(chain.from_iterable([self.tokenizer.encode(x[i], max_length = int(512 / len(x)),  add_special_tokens=True) for i in range(len(x))]))))
        inputs['clss'] = inputs.src.map(lambda x : torch.cat([torch.where(x == 2)[0], torch.tensor([len(x)])]))
        inputs['segs'] = inputs.clss.map(lambda x : torch.tensor(list(chain.from_iterable([[0] * (x[i+1] - x[i]) if i % 2 == 0 else [1] * (x[i+1] - x[i]) for i, val in enumerate(x[:-1])]))))
        inputs['clss'] = inputs.clss.map(lambda x : x[:-1])
        # ##Padding
        max_encoding_len = max(inputs.src.map(lambda x: len(x)))
        max_label_len = max(inputs.clss.map(lambda x: len(x)))
        inputs['src'] = self.pad(inputs.src, 0, max_encoding_len)
        inputs['segs'] = self.pad(inputs.segs, 0, max_encoding_len)
        inputs['clss'] = self.pad(inputs.clss, -1, max_label_len)
        inputs['mask'] = inputs.src.map(lambda x: ~ (x == 0))
        inputs['mask_clss'] = inputs.clss.map(lambda x: ~ (x == -1))

        # #Binarize label {Extracted sentence : 1, Not Extracted sentence : 0}
        labels = labels['trg'].map(lambda  x: torch.tensor([1 if i in x else 0 for i in range(max_label_len)]))
        return inputs, labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return [self.inputs[index][i] for i in range(5)], self.labels[index]

    

class ElectraCCustomDataset(Dataset):
    def __init__(self, data_dir, mode):
        self.data_dir = data_dir
        self.mode = mode
        self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
        self.inputs = self.data_loader()

    def data_loader(self):
        print('Loading ' + self.mode + ' dataset..')
        if os.path.isfile(os.path.join(self.data_dir, self.mode, self.mode + '_X.pt')):
            inputs = torch.load(os.path.join(self.data_dir, self.mode, self.mode + '_X.pt'))
            #print(inputs.head())
            #labels = torch.load(os.path.join(self.data_dir, self.mode, self.mode + '_Y.pt'))

        else:
            file_path = os.path.join(self.data_dir, self.mode, self.mode + '.json')
            df = utils.load_json(file_path)
            inputs = pd.DataFrame(columns=['src'])
            #labels = pd.DataFrame(columns=['trg'])
            inputs['src'] =  df['article_original']
            #labels['trg'] =  df['extractive']
            # Preprocessing
            #print(inputs.head())
            inputs = self.preprocessing(inputs)
            # Save data
            torch.save(inputs ,os.path.join(self.data_dir, self.mode, self.mode + '_X.pt'))

        inputs = inputs.values
        #labels = labels.values

        return inputs

    def pad(self, data, pad_id, max_len):
        padded_data = data.map(lambda x : torch.cat([x, torch.tensor([pad_id] * (max_len - len(x)))]))
        return padded_data

    def preprocessing(self, inputs):
        print('Preprocessing ' + self.mode + ' dataset..')

        #Encoding original text
        inputs['src'] = inputs['src'].map(lambda x: torch.tensor(list(chain.from_iterable([self.tokenizer.encode(x[i], max_length = int(512 / len(x)),  add_special_tokens=True) for i in range(len(x))]))))
        inputs['clss'] = inputs.src.map(lambda x : torch.cat([torch.where(x == 2)[0], torch.tensor([len(x)])]))
        inputs['segs'] = inputs.clss.map(lambda x : torch.tensor(list(chain.from_iterable([[0] * (x[i+1] - x[i]) if i % 2 == 0 else [1] * (x[i+1] - x[i]) for i, val in enumerate(x[:-1])]))))
        inputs['clss'] = inputs.clss.map(lambda x : x[:-1])
        # ##Padding
        max_encoding_len = max(inputs.src.map(lambda x: len(x)))
        max_label_len = max(inputs.clss.map(lambda x: len(x)))
        inputs['src'] = self.pad(inputs.src, 0, max_encoding_len)
        inputs['segs'] = self.pad(inputs.segs, 0, max_encoding_len)
        inputs['clss'] = self.pad(inputs.clss, -1, max_label_len)
        inputs['mask'] = inputs.src.map(lambda x: ~ (x == 0))
        inputs['mask_clss'] = inputs.clss.map(lambda x: ~ (x == -1))

        # #Binarize label {Extracted sentence : 1, Not Extracted sentence : 0}
        #labels = labels['trg'].map(lambda  x: torch.tensor([1 if i in x else 0 for i in range(max_label_len)]))
        return inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return [self.inputs[index][i] for i in range(5)]

class FunnelCustomDataset(Dataset):
    def __init__(self, data_dir, mode):
        self.data_dir = data_dir
        self.mode = mode
        self.tokenizer = FunnelTokenizerFast.from_pretrained("kykim/funnel-kor-base")
        self.inputs, self.labels = self.data_loader()

    def data_loader(self):
        print('Loading ' + self.mode + ' dataset..')
        if os.path.isfile(os.path.join(self.data_dir, self.mode, self.mode + '_X.pt')):
            inputs = torch.load(os.path.join(self.data_dir, self.mode, self.mode + '_X.pt'))
            labels = torch.load(os.path.join(self.data_dir, self.mode, self.mode + '_Y.pt'))

        else:
            file_path = os.path.join(self.data_dir, self.mode, self.mode + '.json')
            df = utils.load_json(file_path)
            inputs = pd.DataFrame(columns=['src'])
            labels = pd.DataFrame(columns=['trg'])
            inputs['src'] =  df['article_original']
            labels['trg'] =  df['extractive']
            # Preprocessing
            inputs, labels = self.preprocessing(inputs, labels)
            # Save data
            torch.save(inputs ,os.path.join(self.data_dir, self.mode, self.mode + '_X.pt'))
            torch.save(labels, os.path.join(self.data_dir, self.mode, self.mode + '_Y.pt'))

        inputs = inputs.values
        labels = labels.values

        return inputs, labels

    def pad(self, data, pad_id, max_len):
        padded_data = data.map(lambda x : torch.cat([x, torch.tensor([pad_id] * (max_len - len(x)))]))
        return padded_data

    def preprocessing(self, inputs, labels):
        print('Preprocessing ' + self.mode + ' dataset..')

        #Encoding original text
        inputs['src'] = inputs['src'].map(lambda x: torch.tensor(list(chain.from_iterable([self.tokenizer.encode(x[i], max_length = int(512 / len(x)),  add_special_tokens=True) for i in range(len(x))]))))
        inputs['clss'] = inputs.src.map(lambda x : torch.cat([torch.where(x == 2)[0], torch.tensor([len(x)])]))
        inputs['segs'] = inputs.clss.map(lambda x : torch.tensor(list(chain.from_iterable([[0] * (x[i+1] - x[i]) if i % 2 == 0 else [1] * (x[i+1] - x[i]) for i, val in enumerate(x[:-1])]))))
        inputs['clss'] = inputs.clss.map(lambda x : x[:-1])
        # ##Padding
        max_encoding_len = max(inputs.src.map(lambda x: len(x)))
        max_label_len = max(inputs.clss.map(lambda x: len(x)))
        inputs['src'] = self.pad(inputs.src, 0, max_encoding_len)
        inputs['segs'] = self.pad(inputs.segs, 0, max_encoding_len)
        inputs['clss'] = self.pad(inputs.clss, -1, max_label_len)
        inputs['mask'] = inputs.src.map(lambda x: ~ (x == 0))
        inputs['mask_clss'] = inputs.clss.map(lambda x: ~ (x == -1))

        # #Binarize label {Extracted sentence : 1, Not Extracted sentence : 0}
        labels = labels['trg'].map(lambda  x: torch.tensor([1 if i in x else 0 for i in range(max_label_len)]))
        return inputs, labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return [self.inputs[index][i] for i in range(5)], self.labels[index]
    


class FunnelCCustomDataset(Dataset):
    def __init__(self, data_dir, mode):
        self.data_dir = data_dir
        self.mode = mode
        self.tokenizer = FunnelTokenizerFast.from_pretrained("kykim/funnel-kor-base")
        self.inputs = self.data_loader()

    def data_loader(self):
        print('Loading ' + self.mode + ' dataset..')
        if os.path.isfile(os.path.join(self.data_dir, self.mode, self.mode + '_X.pt')):
            inputs = torch.load(os.path.join(self.data_dir, self.mode, self.mode + '_X.pt'))
            #print(inputs.head())
            #labels = torch.load(os.path.join(self.data_dir, self.mode, self.mode + '_Y.pt'))

        else:
            file_path = os.path.join(self.data_dir, self.mode, self.mode + '.json')
            df = utils.load_json(file_path)
            inputs = pd.DataFrame(columns=['src'])
            #labels = pd.DataFrame(columns=['trg'])
            inputs['src'] =  df['article_original']
            #labels['trg'] =  df['extractive']
            # Preprocessing
            #print(inputs.head())
            inputs = self.preprocessing(inputs)
            # Save data
            torch.save(inputs ,os.path.join(self.data_dir, self.mode, self.mode + '_X.pt'))

        inputs = inputs.values
        #labels = labels.values

        return inputs

    def pad(self, data, pad_id, max_len):
        padded_data = data.map(lambda x : torch.cat([x, torch.tensor([pad_id] * (max_len - len(x)))]))
        return padded_data

    def preprocessing(self, inputs):
        print('Preprocessing ' + self.mode + ' dataset..')

        #Encoding original text
        inputs['src'] = inputs['src'].map(lambda x: torch.tensor(list(chain.from_iterable([self.tokenizer.encode(x[i], max_length = int(512 / len(x)),  add_special_tokens=True) for i in range(len(x))]))))
        inputs['clss'] = inputs.src.map(lambda x : torch.cat([torch.where(x == 2)[0], torch.tensor([len(x)])]))
        inputs['segs'] = inputs.clss.map(lambda x : torch.tensor(list(chain.from_iterable([[0] * (x[i+1] - x[i]) if i % 2 == 0 else [1] * (x[i+1] - x[i]) for i, val in enumerate(x[:-1])]))))
        inputs['clss'] = inputs.clss.map(lambda x : x[:-1])
        # ##Padding
        max_encoding_len = max(inputs.src.map(lambda x: len(x)))
        max_label_len = max(inputs.clss.map(lambda x: len(x)))
        inputs['src'] = self.pad(inputs.src, 0, max_encoding_len)
        inputs['segs'] = self.pad(inputs.segs, 0, max_encoding_len)
        inputs['clss'] = self.pad(inputs.clss, -1, max_label_len)
        inputs['mask'] = inputs.src.map(lambda x: ~ (x == 0))
        inputs['mask_clss'] = inputs.clss.map(lambda x: ~ (x == -1))

        # #Binarize label {Extracted sentence : 1, Not Extracted sentence : 0}
        #labels = labels['trg'].map(lambda  x: torch.tensor([1 if i in x else 0 for i in range(max_label_len)]))
        return inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return [self.inputs[index][i] for i in range(5)]


class BertCustomDataset(Dataset):
    def __init__(self, data_dir, mode):
        self.data_dir = data_dir
        self.mode = mode
        self.tokenizer = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
        self.inputs, self.labels = self.data_loader()

    def data_loader(self):
        print('Loading ' + self.mode + ' dataset..')
        if os.path.isfile(os.path.join(self.data_dir, self.mode, self.mode + '_X.pt')):
            inputs = torch.load(os.path.join(self.data_dir, self.mode, self.mode + '_X.pt'))
            labels = torch.load(os.path.join(self.data_dir, self.mode, self.mode + '_Y.pt'))

        else:
            file_path = os.path.join(self.data_dir, self.mode, self.mode + '.json')
            df = utils.load_json(file_path)
            inputs = pd.DataFrame(columns=['src'])
            labels = pd.DataFrame(columns=['trg'])
            inputs['src'] =  df['article_original']
            labels['trg'] =  df['extractive']
            # Preprocessing
            inputs, labels = self.preprocessing(inputs, labels)
            # Save data
            torch.save(inputs ,os.path.join(self.data_dir, self.mode, self.mode + '_X.pt'))
            torch.save(labels, os.path.join(self.data_dir, self.mode, self.mode + '_Y.pt'))

        inputs = inputs.values
        labels = labels.values

        return inputs, labels

    def pad(self, data, pad_id, max_len):
        padded_data = data.map(lambda x : torch.cat([x, torch.tensor([pad_id] * (max_len - len(x)))]))
        return padded_data

    def preprocessing(self, inputs, labels):
        print('Preprocessing ' + self.mode + ' dataset..')

        #Encoding original text
        inputs['src'] = inputs['src'].map(lambda x: torch.tensor(list(chain.from_iterable([self.tokenizer.encode(x[i], max_length = int(512 / len(x)),  add_special_tokens=True) for i in range(len(x))]))))
        inputs['clss'] = inputs.src.map(lambda x : torch.cat([torch.where(x == 2)[0], torch.tensor([len(x)])]))
        inputs['segs'] = inputs.clss.map(lambda x : torch.tensor(list(chain.from_iterable([[0] * (x[i+1] - x[i]) if i % 2 == 0 else [1] * (x[i+1] - x[i]) for i, val in enumerate(x[:-1])]))))
        inputs['clss'] = inputs.clss.map(lambda x : x[:-1])
        # ##Padding
        max_encoding_len = max(inputs.src.map(lambda x: len(x)))
        max_label_len = max(inputs.clss.map(lambda x: len(x)))
        inputs['src'] = self.pad(inputs.src, 0, max_encoding_len)
        inputs['segs'] = self.pad(inputs.segs, 0, max_encoding_len)
        inputs['clss'] = self.pad(inputs.clss, -1, max_label_len)
        inputs['mask'] = inputs.src.map(lambda x: ~ (x == 0))
        inputs['mask_clss'] = inputs.clss.map(lambda x: ~ (x == -1))

        # #Binarize label {Extracted sentence : 1, Not Extracted sentence : 0}
        labels = labels['trg'].map(lambda  x: torch.tensor([1 if i in x else 0 for i in range(max_label_len)]))
        return inputs, labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return [self.inputs[index][i] for i in range(5)], self.labels[index]
    


class BertCCustomDataset(Dataset):
    def __init__(self, data_dir, mode):
        self.data_dir = data_dir
        self.mode = mode
        self.tokenizer = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
        self.inputs = self.data_loader()

    def data_loader(self):
        print('Loading ' + self.mode + ' dataset..')
        if os.path.isfile(os.path.join(self.data_dir, self.mode, self.mode + '_X.pt')):
            inputs = torch.load(os.path.join(self.data_dir, self.mode, self.mode + '_X.pt'))
            #print(inputs.head())
            #labels = torch.load(os.path.join(self.data_dir, self.mode, self.mode + '_Y.pt'))

        else:
            file_path = os.path.join(self.data_dir, self.mode, self.mode + '.json')
            df = utils.load_json(file_path)
            inputs = pd.DataFrame(columns=['src'])
            #labels = pd.DataFrame(columns=['trg'])
            inputs['src'] =  df['article_original']
            #labels['trg'] =  df['extractive']
            # Preprocessing
            #print(inputs.head())
            inputs = self.preprocessing(inputs)
            # Save data
            torch.save(inputs ,os.path.join(self.data_dir, self.mode, self.mode + '_X.pt'))

        inputs = inputs.values
        #labels = labels.values

        return inputs

    def pad(self, data, pad_id, max_len):
        padded_data = data.map(lambda x : torch.cat([x, torch.tensor([pad_id] * (max_len - len(x)))]))
        return padded_data

    def preprocessing(self, inputs):
        print('Preprocessing ' + self.mode + ' dataset..')

        #Encoding original text
        inputs['src'] = inputs['src'].map(lambda x: torch.tensor(list(chain.from_iterable([self.tokenizer.encode(x[i], max_length = int(512 / len(x)),  add_special_tokens=True) for i in range(len(x))]))))
        inputs['clss'] = inputs.src.map(lambda x : torch.cat([torch.where(x == 2)[0], torch.tensor([len(x)])]))
        inputs['segs'] = inputs.clss.map(lambda x : torch.tensor(list(chain.from_iterable([[0] * (x[i+1] - x[i]) if i % 2 == 0 else [1] * (x[i+1] - x[i]) for i, val in enumerate(x[:-1])]))))
        inputs['clss'] = inputs.clss.map(lambda x : x[:-1])
        # ##Padding
        max_encoding_len = max(inputs.src.map(lambda x: len(x)))
        max_label_len = max(inputs.clss.map(lambda x: len(x)))
        inputs['src'] = self.pad(inputs.src, 0, max_encoding_len)
        inputs['segs'] = self.pad(inputs.segs, 0, max_encoding_len)
        inputs['clss'] = self.pad(inputs.clss, -1, max_label_len)
        inputs['mask'] = inputs.src.map(lambda x: ~ (x == 0))
        inputs['mask_clss'] = inputs.clss.map(lambda x: ~ (x == -1))

        # #Binarize label {Extracted sentence : 1, Not Extracted sentence : 0}
        #labels = labels['trg'].map(lambda  x: torch.tensor([1 if i in x else 0 for i in range(max_label_len)]))
        return inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return [self.inputs[index][i] for i in range(5)]




