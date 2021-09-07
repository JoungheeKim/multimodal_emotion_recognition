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

@hydra.main(config_path=os.path.join("..", "configs"), config_name="train")
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
    train_dataset = SentimentDataset.load_with_config(cfg, 'train')
    valid_dataset = SentimentDataset.load_with_config(cfg, 'valid')
    eval_dataset = SentimentDataset.load_with_config(cfg, 'test')

    ## load model
    model = get_model_class(cfg.model).load_with_config(cfg)
    model.to(args.device)

    writer = ResultWriter(args.experiments_path)
    results = {}

    ## training
    train_results = train(args, cfg.label, train_dataset, valid_dataset, model)
    results.update(**train_results)

    args.pretrained_model = args.save_path

    loss_pretrained_model_path = os.path.join(args.pretrained_model, 'loss')
    model = get_model_class(cfg.model).from_pretrained(loss_pretrained_model_path)
    model.to(args.device)
    test_results = evaluate(args, cfg.label, valid_dataset, model)
    test_results.update(
        {
            'valid_check': 'loss',
            'split': 'valid',
        }
    )
    results.update(test_results)
    writer.update(cfg.base, cfg.model, cfg.label, cfg.audio_feature, cfg.language_feature, **results)

    ## test
    results = {}
    test_results = evaluate(args, cfg.label, eval_dataset, model)
    test_results.update(
        {
            'valid_check': 'loss',
            'split': 'test',
        }
    )
    results.update(test_results)
    writer.update(cfg.base, cfg.model, cfg.label, cfg.audio_feature, cfg.language_feature, **results)

    loss_pretrained_model_path = os.path.join(args.pretrained_model, 'accuracy')
    model = get_model_class(cfg.model).from_pretrained(loss_pretrained_model_path)
    model.to(args.device)
    results = {}
    test_results = evaluate(args, cfg.label, eval_dataset, model)
    test_results.update(
        {
            'valid_check': 'accuracy',
            'split': 'test',
        }
    )
    results.update(test_results)
    writer.update(cfg.base, cfg.model, cfg.label, cfg.audio_feature, cfg.language_feature, **results)



def train(args:DefaultConfig, label_cfg, train_dataset, valid_dataset, model):
    logging.info("start training")

    ## load dataloader
    train_dataloader = load_dataloader(
        train_dataset, shuffle=True, batch_size=args.train_batch_size
    )

    if args.max_steps > 0:
        t_total = args.max_steps

        args.num_train_epochs = (
            args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        )
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    no_decay = ["bias", "LayerNorm.weight"]
    wav2vec_parameter_name = 'w2v_encoder'
    bert_parameter_name = 'bert_encoder'
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and wav2vec_parameter_name in n
            ],
            "weight_decay": args.weight_decay,
            'lr' : args.wav2vec_learning_rate if args.wav2vec_learning_rate > 0 else args.learning_rate,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and bert_parameter_name in n
            ],
            "weight_decay": args.weight_decay,
            'lr' : args.bert_learning_rate if args.bert_learning_rate > 0 else args.learning_rate,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and wav2vec_parameter_name not in n and bert_parameter_name not in n
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    args.warmup_steps = int(args.warmup_percent * t_total)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)
    logging.info("  Train Batch size = %d", args.train_batch_size)
    logging.info("  Train Data size = %d", len(train_dataset))

    step = 0
    global_step = 0
    best_loss = 1e10
    best_loss_step = 0
    best_f1 = 0
    best_f1_step = 0
    stop_iter = False

    train_iterator = trange(0, int(args.num_train_epochs), desc="Epoch")
    model.zero_grad()
    model.set_step(global_step)
    for _ in train_iterator:
        for batch in train_dataloader:
            step += 1
            model.train()

            batch = {key:item.to(args.device) for key, item in batch.items() if type(item)==torch.Tensor}
            net_output = model(**batch)

            final_loss = model.get_loss(net_output, batch['label'])

            if args.gradient_accumulation_steps > 1:
                final_loss = final_loss / args.gradient_accumulation_steps
            if args.n_gpu > 1:
                final_loss = final_loss.mean()
            if args.fp16:
                with amp.scale_loss(final_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                final_loss.backward()

            train_iterator.set_postfix_str(s="loss = {:.8f}".format(float(final_loss)), refresh=True)
            if step % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                ## set global step
                model.set_step(global_step)

                if (args.logging_steps > 0 and global_step % args.logging_steps == 0 and global_step > 0):
                    # if (args.logging_steps > 0 and global_step % args.logging_steps == 0):
                    results = evaluate(args, label_cfg, valid_dataset, model, global_step)
                    eval_loss = results['loss']
                    eval_f1 = results['f1_score']
                    eval_acc = results['accuracy']

                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        #best_f1 = eval_f1
                        best_loss_step = global_step

                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(os.path.join(args.save_path, 'loss'))
                        #torch.save(args, os.path.join(args.save_path, 'loss', "training_args.bin"))

                    if eval_f1 > best_f1:
                        best_f1 = eval_f1
                        best_f1_step = global_step

                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(os.path.join(args.save_path, 'accuracy'))
                        #torch.save(args, os.path.join(args.save_path, 'accuracy', "training_args.bin"))

                    logging.info("***** best_acc : %.4f *****", eval_acc)
                    logging.info("***** best_f1 : %.4f *****", best_f1)
                    logging.info("***** best_loss : %.4f *****", best_loss)



            if args.max_steps > 0 and global_step > args.max_steps:
                stop_iter = True
                break

        if stop_iter:
            break

    return {'best_valid_loss': best_loss,
            'best_valid_loss_step': best_loss_step,
            'best_valid_f1': eval_f1,
            'best_valid_f1_step': best_f1_step,
            }



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

    return results

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

if __name__ == "__main__":
    init()
    main()

