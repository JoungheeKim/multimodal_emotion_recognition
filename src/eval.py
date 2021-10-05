# coding=utf-8
# Copyright 2020 The JoungheeKim All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hydra
import os
from src.utils import reset_logging, init, load_data, save_data, ResultWriter
import logging
from src.dataset.sentiment_dataset import SentimentDataset, load_dataloader
from src.model import get_model_class
from src.configs import DefaultConfig
import torch
import numpy as np
from torch.utils.data import (
    DataLoader
)
from transformers import (
    AdamW,
    BertForSequenceClassification,
    BertConfig,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
import random
from tqdm import tqdm, trange
from collections import OrderedDict
LOSS_MAPPING = OrderedDict(
    [
        # Add configs here
        ("bce", torch.nn.BCELoss()),
        ('cross', torch.nn.NLLLoss()),
    ]
)
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import pickle

@hydra.main(config_path=os.path.join("..", "configs"), config_name="eval")
def main(cfg):
    ## Resent Logging
    reset_logging()

    args = cfg.base
    ## GPU setting
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    set_seed(args.seed)

    ## load_dataset
    valid_dataset = SentimentDataset.load_with_config(cfg, 'valid')
    eval_dataset = SentimentDataset.load_with_config(cfg, 'test')

    writer = ResultWriter(args.experiments_path)
    results = {}

    args.pretrained_model = args.save_path

    loss_pretrained_model_path = os.path.join(args.pretrained_model, 'loss')
    model = get_model_class(cfg.model).from_pretrained(loss_pretrained_model_path)
    model.to(args.device)
    test_results, total_preds, total_labels = evaluate(args, cfg.label, valid_dataset, model)
    test_results.update(
        {
            'valid_check': 'loss',
            'split': 'valid',
        }
    )
    results.update(test_results)
    writer.update(cfg.base, cfg.model, cfg.label, cfg.audio_feature, cfg.language_feature, **results)

    save_dump_results(args, "loss_valid.pkl", total_preds, total_labels)

    
    ## test
    results = {}
    test_results, total_preds, total_labels = evaluate(args, cfg.label, eval_dataset, model)
    test_results.update(
        {
            'valid_check': 'loss',
            'split': 'test',
        }
    )
    results.update(test_results)
    writer.update(cfg.base, cfg.model, cfg.label, cfg.audio_feature, cfg.language_feature, **results)

    save_dump_results(args, "loss_test.pkl", total_preds, total_labels)

    loss_pretrained_model_path = os.path.join(args.pretrained_model, 'accuracy')
    model = get_model_class(cfg.model).from_pretrained(loss_pretrained_model_path)
    model.to(args.device)
    results = {}
    test_results, total_preds, total_labels = evaluate(args, cfg.label, eval_dataset, model)
    test_results.update(
        {
            'valid_check': 'accuracy',
            'split': 'test',
        }
    )
    results.update(test_results)
    writer.update(cfg.base, cfg.model, cfg.label, cfg.audio_feature, cfg.language_feature, **results)

    save_dump_results(args, "acc_test.pkl", total_preds, total_labels)


def evaluate(args, label_cfg, test_dataset, model, global_step=0):

    ## load dataloader
    test_dataloader = load_dataloader(
        test_dataset, shuffle=False, batch_size=args.eval_batch_size
    )

    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    logging.info("***** Running evaluation {}*****".format(global_step))
    eval_loss = 0.0
    nb_eval_steps = 0
    total_preds = []
    total_labels = []
    criterion = LOSS_MAPPING[label_cfg.loss_type]

    model.eval()
    for batch in tqdm(test_dataloader, desc="Evaluating"):

        batch = {key: item.to(args.device) for key, item in batch.items() if type(item) == torch.Tensor}

        with torch.no_grad():
            net_output = model(**batch)

            lprobs = model.get_normalized_probs(
                net_output, log_probs=True, loss_type=label_cfg.loss_type,
            ).contiguous()

            probs = model.get_normalized_probs(
                net_output, log_probs=False, loss_type=label_cfg.loss_type,
            ).contiguous()

            entropy_loss = criterion(lprobs, batch['label'])
            eval_loss += entropy_loss.mean().item()

        nb_eval_steps += 1

        if label_cfg.loss_type == 'bce':
            preds = probs.detach().cpu()
        else:
            preds = probs.detach().cpu().argmax(axis=1)
        labels = batch['label'].detach().cpu()

        total_preds.append(preds)
        total_labels.append(labels)

    total_preds = torch.cat(total_preds)
    total_labels = torch.cat(total_labels)

    eval_loss = eval_loss/nb_eval_steps

    if label_cfg.loss_type == 'bce':
        f1, acc, eval_dict = eval_iemocap(label_cfg, total_preds, total_labels)
        results = {
            'loss': eval_loss,
            'accuracy': acc,
            'f1_score': f1,
        }
        results.update(eval_dict)
        for key, item in eval_dict.items():
            logging.info("  %s = %s", str(key), str(item))
    else:
        acc = accuracy_score(total_labels,total_preds)
        f1 = f1_score(total_labels, total_preds, average="weighted")
        results = {
            'loss' : eval_loss,
            'accuracy' : acc,
            'f1_score' : f1,
        }
        f1, acc, eval_dict = eval_iemocap(label_cfg, total_preds, total_labels)
        results.update(eval_dict)
        for key, item in eval_dict.items():
            logging.info("  %s = %s", str(key), str(item))

    # f1, acc, eval_dict =eval_iemocap(config, total_preds, total_labels)
    #
    # results = {
    #     "loss": eval_loss,
    #     "accuracy": acc,
    #     'f1_score' : f1,
    # }
    # results.update(eval_dict)

    # for key in sorted(results.keys()):
    #     logger.info("  %s = %s", key, str(results[key]))
    logging.info("  %s = %s", 'loss', str(results['loss']))
    logging.info("  %s = %s", 'f1_score', str(results['f1_score']))
    logging.info("  %s = %s", 'acc', str(results['accuracy']))
    model.train()

    return results, total_preds, total_labels

def eval_iemocap(label_cfg, results, truths, single=-1):
    logging.info("eval with iemocap formulation")
    f1s = []
    accs = []
    save_dict = {}
    if label_cfg.loss_type == 'bce':
        emos = eval(label_cfg.selected_class)
        # if single < 0:
        test_preds = results.cpu().detach().numpy()
        test_truth = truths.cpu().detach().numpy()

        for emo_ind in range(len(emos)):
            #logger.info(f"{emos[emo_ind]}: ")
            test_preds_i = np.round(test_preds[:, emo_ind])
            test_truth_i = test_truth[:, emo_ind]
            f1 = f1_score(test_truth_i, test_preds_i, average="weighted")
            f1s.append(f1)
            acc = accuracy_score(test_truth_i, test_preds_i)
            accs.append(acc)
            #logger.info("  - F1 Score: %.3f", f1)
            #logger.info("  - Accuracy: %.3f", acc)
            save_dict['{}_f1'.format(emos[emo_ind])]=f1
            save_dict['{}_acc'.format(emos[emo_ind])]=acc
            save_dict['{}_count'.format(emos[emo_ind])]=sum(test_preds_i)/sum(test_truth_i)
    if label_cfg.loss_type == 'cross':
        emos = eval(label_cfg.selected_class)
        # if single < 0:
        test_preds = results.cpu().detach().numpy()
        test_truth = truths.cpu().detach().numpy()

        test_preds = np.eye(len(emos))[test_preds]
        test_truth = np.eye(len(emos))[test_truth]
        for emo_ind in range(len(emos)):
            #logger.info(f"{emos[emo_ind]}: ")
            test_preds_i = test_preds[:, emo_ind]
            test_truth_i = test_truth[:, emo_ind]
            f1 = f1_score(test_truth_i, test_preds_i, average="weighted")
            f1s.append(f1)
            acc = accuracy_score(test_truth_i, test_preds_i)
            accs.append(acc)
            #logger.info("  - F1 Score: %.3f", f1)
            #logger.info("  - Accuracy: %.3f", acc)
            save_dict['{}_f1'.format(emos[emo_ind])]=f1
            save_dict['{}_acc'.format(emos[emo_ind])]=acc
            save_dict['{}_count'.format(emos[emo_ind])] = sum(test_preds_i) / sum(test_truth_i)

    save_dict.update({
        "f1_2class" :np.mean(f1s),
        "acc_2class": np.mean(accs),
    })

    return np.mean(f1s), np.mean(accs), save_dict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

def save_dump_results(args, name:str, total_preds, total_labels):
    import pickle
    ## dump
    if args.result_path:
        os.makedirs(args.result_path, exist_ok=True)
        assert '.pkl' in name, "이름 다시 지어야 해."
        save_path = os.path.join(args.result_path, name)
        save_file = {
            'total_preds' : total_preds,
            'total_labels' : total_labels
        }
        
        with open(save_path, 'wb') as f:
             pickle.dump(save_file, f)



if __name__ == "__main__":
    init()
    main()

