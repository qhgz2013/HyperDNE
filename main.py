import argparse
import re
import numpy as np
import data_preprocess
import logging
import utils
import model
import torch
import torch.optim as opt
import sklearn.metrics as metrics
from typing import *
from time import time
import torch.utils.tensorboard as tb  # tensorboard visualization
import os
from datetime import datetime
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    # global args
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=404)

    # dataset processing args
    parser.add_argument('--dataset', type=str, default='uci', choices=['enron', 'm1', 'uci', 'yelp'])
    parser.add_argument('--batch_size', type=int, default=512)
    # negative sample(s) per positive sample(s) during training (neg_sample_size) and testing (test_neg_sample_size)
    parser.add_argument('--neg_sample_size', type=int, default=10, help='# neg samples per pos sample (training)')
    parser.add_argument('--test_neg_sample_size', type=int, default=1)
    parser.add_argument('--directed_graph', default=False, action='store_true')  # changed to directed_graph!
    # time period to predict, starts from 0, -1 means using the last time period
    parser.add_argument('--predict_time', type=int, default=-1)
    # how many time periods should be used to predict since predict_time, default 1 means single-step prediction
    parser.add_argument('--n_predict_steps', type=int, default=1)

    # model args
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--n_hgcn_layers', type=int, default=2)
    parser.add_argument('--n_line_conv_layers', type=int, default=0)
    parser.add_argument('--n_transformer_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--dropout_graph', type=float, default=0.3)
    parser.add_argument('--no_positional_embedding', default=False, action='store_true')
    parser.add_argument('--include_line_conv_weight', default=False, action='store_true')

    # training args
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2_reg', type=float, default=0.0)
    parser.add_argument('--evaluate_per_epochs', type=int, default=10)
    # specify a directory or file to load previously saved model weights
    # e.g. "--load_weight weights/run1/epoch5.pth" can load the model params from the specified file
    # "--load_weight weights/run1" will load the latest model params in the specified directory
    parser.add_argument('--load_weight', type=str, default=None)

    # model / log args
    parser.add_argument('--weight_dir', type=str, default='weights')  # dir for saving trained models
    # numbers of weights to keep (the previous saved weight files will be removed)
    parser.add_argument('--max_keep_weights', type=int, default=10)
    parser.add_argument('--log_dir', type=str, default='runs')
    # logging for distributions of static / dynamic embedding are disabled by default, set this flag to enable it
    parser.add_argument('--log_embedding_histogram', default=False, action='store_true')
    # remove existed tensorboard run logs / weights: --reset_log_dir and --reset_weight_dir
    parser.add_argument('--reset_log_dir', default=False, action='store_true')
    parser.add_argument('--reset_weight_dir', default=False, action='store_true')
    parser.add_argument('--disable_log', default=False, action='store_true')  # disable tensorboard logging
    # tensorboard logging naming stuff
    parser.add_argument('--extra_log_prefix', default=None, type=str)
    parser.add_argument('--extra_log_postfix', default=None, type=str)

    args = parser.parse_args()
    return args


def _feed_forward(nn_model, summary_writer: Optional[tb.SummaryWriter] = None, epoch: Optional[int] = None, *args):
    # np array -> tensor
    args = [model.to_cuda(torch.LongTensor(x)) for x in args]
    # if error raised here, make sure it is the exact output produced from DatasetLoader.__iter__()
    assert len(args) in (4, 6), 'Invalid args'
    if len(args) == 4:
        # shared pos u and neg u
        guid_u, guid_pos_v, guid_neg_v, time_period = args
        guid_pos_u = guid_u
        time_period_pos = time_period
        guid_neg_u = torch.unsqueeze(guid_u, 1)
        time_period_neg = torch.unsqueeze(time_period, 1)
    else:
        # separated pos u and neg u
        guid_pos_u, guid_pos_v, guid_neg_u, guid_neg_v, time_period_pos, time_period_neg = args
    return nn_model(guid_pos_u, guid_pos_v, guid_neg_u, guid_neg_v, time_period_pos, time_period_neg,
                    summary_writer=summary_writer, global_step=epoch)


def _cross_entropy(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
    eps = torch.tensor(1e-10)
    return -torch.sum(torch.log(torch.sigmoid(pos_logits) + eps)) - \
        torch.sum(torch.log(1 - torch.sigmoid(neg_logits) + eps))


def _evaluate(dataset: data_preprocess.Dataset, nn_model: model.HGSocialRec, subset: str, epoch: int, predict_step: int,
              summary_writer: Optional[tb.SummaryWriter] = None) -> Tuple[float, float]:
    logger = logging.getLogger('evaluate')
    logger.debug('Evaluating subset: %s at step %d', subset, predict_step)
    nn_model.eval()
    loader = data_preprocess.PredictDataLoader(dataset, subset, predict_step)
    with torch.no_grad():
        pos_prob, neg_prob = [], []
        loss = 0
        for args in loader:
            embeddings, logits = _feed_forward(nn_model, None, None, *args)
            pos_logits, neg_logits = logits
            batch_pos_prob = model.to_cpu(torch.sigmoid(pos_logits).view(-1).detach()).numpy()
            batch_neg_prob = model.to_cpu(torch.sigmoid(neg_logits).view(-1).detach()).numpy()
            pos_prob.append(batch_pos_prob)
            neg_prob.append(batch_neg_prob)
            cross_entropy = _cross_entropy(pos_logits, neg_logits)
            batch_loss = cross_entropy / (1 + dataset.test_neg_sample_size) / args[0].shape[0]
            loss += model.to_cpu(batch_loss.detach()).numpy()
        pos_prob, neg_prob = np.concatenate(pos_prob), np.concatenate(neg_prob)
        y_pred = np.concatenate([pos_prob, neg_prob])
        y_label = np.concatenate([np.ones_like(pos_prob), np.zeros_like(neg_prob)])
        auc = metrics.roc_auc_score(y_label, y_pred)
    if summary_writer:
        summary_writer.add_scalar(f'loss/{subset}/step{predict_step+1}', loss, epoch)
        summary_writer.add_scalar(f'auc/{subset}/step{predict_step+1}', auc, epoch)
        summary_writer.add_histogram(f'pred_dist/{subset}/step{predict_step+1}', y_pred, epoch)
    return auc, loss


_best_result = [[0.0, 0.0, 0]]  # AUC test, loss test, epoch


def evaluate(dataset: data_preprocess.Dataset, nn_model: model.HGSocialRec, epoch: int, n_predict_steps: int,
             summary_writer: Optional[tb.SummaryWriter] = None, target_step: int = 2) \
        -> Tuple[bool, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    # returns: (is_best_epoch, (loss_val, auc_val, loss_test, auc_test)) from multi-step prediction result
    global _best_result
    logger = logging.getLogger('evaluate')
    auc_val, loss_val, auc_test, loss_test = [], [], [], []
    is_best = False
    # for generality test, we use step 2 (target_step) results (since we follow the setting in DySAT implementation
    # that links from step 1 are also included into training), but if n_predict_steps does not satisfy target_step,
    # use the minimal value
    target_step = min(target_step, n_predict_steps) - 1
    for t in range(n_predict_steps):
        cur_auc_val, cur_loss_val = _evaluate(dataset, nn_model, 'val', epoch, t, summary_writer)
        cur_auc_test, cur_loss_test = _evaluate(dataset, nn_model, 'test', epoch, t, summary_writer)
        auc_val.append(cur_auc_val)
        auc_test.append(cur_auc_test)
        loss_val.append(cur_loss_val)
        loss_test.append(cur_loss_test)
        if len(_best_result) <= t:
            _best_result.append([0.0, 0.0, 0])
        if cur_auc_test > _best_result[t][0]:
            _best_result[t] = [cur_auc_test, cur_loss_test, epoch]
            if t == target_step:
                is_best = True  # only use single-step prediction result the majority result
        logger.info('Epoch: %d, predict step: %d, val set: [loss: %f, AUC: %f], test set: [loss: %f, AUC: %f], '
                    'best (test): [epoch: %d, loss: %f, AUC: %f]', epoch, t, cur_loss_val, cur_auc_val, cur_loss_test,
                    cur_auc_test, _best_result[t][2], _best_result[t][1], _best_result[t][0])
    result = np.array(loss_val), np.array(auc_val), np.array(loss_test), np.array(auc_test)
    return is_best, result


def _evaluate_and_save_model(dataset: data_preprocess.Dataset, nn_model: model.HGSocialRec,
                             optimizer: torch.optim.Optimizer, weight_dir: str, epoch: int, max_keep_weights: int = 0,
                             n_predict_steps: int = 1, summary_writer: Optional[tb.SummaryWriter] = None,
                             target_step: int = 2):
    evaluate_result = evaluate(dataset, nn_model, epoch, n_predict_steps=n_predict_steps, summary_writer=summary_writer,
                               target_step=target_step)
    is_best, (loss_val, auc_val, loss_test, auc_test) = evaluate_result
    now = datetime.now().strftime('%y%m%d_%H%M%S')
    latest_file_save_path = os.path.join(weight_dir, 'latest.pth')
    t = min(target_step, n_predict_steps) - 1
    if is_best:
        save_name = f'{now}_epoch{epoch}_lVal{format(loss_val[t],".3f")}_aucVal{format(auc_val[t],".3f")}_lTest' \
                    f'{format(loss_test[t],".3f")}_aucTest{format(auc_test[t],".3f")}.pth'
        save_path = os.path.join(weight_dir, save_name)
        utils.save_model(save_path, [nn_model, optimizer, _best_result], epoch, include_random_state=True,
                         max_keep_weights=max_keep_weights, ignore_file_regex='latest.pth')
        shutil.copy(save_path, latest_file_save_path)
    else:
        utils.save_model(latest_file_save_path, [nn_model, optimizer, _best_result], epoch, include_random_state=True,
                         max_keep_weights=max_keep_weights, ignore_file_regex='latest.pth')
    return evaluate_result


def train(dataset: data_preprocess.Dataset, nn_model: model.HGSocialRec, epochs: int, lr: float, l2_reg: float,
          evaluate_per_epochs: int, weight_dir: str, summary_writer: Optional[tb.SummaryWriter] = None,
          load_weight_file_or_dir: Optional[str] = None, max_keep_weights: int = 0, n_predict_steps: int = 1,
          log_embedding_histogram: bool = False):
    logger = logging.getLogger('train')
    params = list(nn_model.parameters())
    logger.debug('Model parameters:')
    for name, value in nn_model.state_dict().items():
        logger.debug('%s: %s', name, str(value.size()))
    adam = opt.AdamW(params, lr, weight_decay=l2_reg)
    # load weights
    begin_epoch = 0
    if load_weight_file_or_dir:
        params_from_ckpt = []
        assert os.path.exists(load_weight_file_or_dir), \
            'Could not load weight: path "%s" not found' % load_weight_file_or_dir
        if os.path.isdir(load_weight_file_or_dir):
            begin_epoch = utils.load_last_saved_model(load_weight_file_or_dir, params_from_ckpt,
                                                      use_saved_random_state=True)
            assert begin_epoch is not None,\
                'Failed to load last saved model from directory %s' % load_weight_file_or_dir
        else:
            begin_epoch = utils.load_model(load_weight_file_or_dir, params_from_ckpt, use_saved_random_state=True)
        if len(params_from_ckpt) == 3:
            global _best_result
            nn_model_param, adam_param, _best_result = params_from_ckpt
        else:
            nn_model_param, adam_param = params_from_ckpt  # previous saved format, not used in newer version
        nn_model.load_state_dict(nn_model_param)
        adam.load_state_dict(adam_param)

    dataset_loader = data_preprocess.DatasetLoader(dataset, unique_pos_edge=False)
    for epoch in range(begin_epoch + 1, epochs + 1):
        nn_model.train()
        epoch_loss = 0
        begin_t = time()
        for args in dataset_loader:
            adam.zero_grad()
            if log_embedding_histogram and epoch % evaluate_per_epochs == 0:
                embeddings, logits = _feed_forward(nn_model, summary_writer, epoch, *args)
            else:
                embeddings, logits = _feed_forward(nn_model, None, None, *args)
            pos_logits, neg_logits = logits
            # loss function: binary cross entropy
            cross_entropy = _cross_entropy(pos_logits, neg_logits)
            loss = cross_entropy / (1 + dataset.neg_sample_size) / args[0].shape[0]
            torch.nn.utils.clip_grad_value_(params, 1)
            loss.backward(retain_graph=True)
            adam.step()
            epoch_loss += model.to_cpu(loss.detach()).numpy()
        t_used = time() - begin_t
        logger.info('Epoch %d/%d: loss: %f, time: %f', epoch, epochs, epoch_loss, t_used)

        if summary_writer:
            summary_writer.add_scalar('loss/train', epoch_loss, epoch)
            summary_writer.add_scalar('time/train', t_used, epoch)

        # evaluation
        if epoch % evaluate_per_epochs == 0:
            _evaluate_and_save_model(dataset, nn_model, adam, weight_dir, epoch, max_keep_weights, n_predict_steps,
                                     summary_writer)
    if epochs % evaluate_per_epochs != 0:
        _evaluate_and_save_model(dataset, nn_model, adam, weight_dir, epochs, max_keep_weights, n_predict_steps,
                                 summary_writer)


def main():
    now_str = datetime.now().strftime('%y%m%d_%H%M%S')
    args = parse_args()
    utils.configure_logging(args.verbose)
    logger = logging.getLogger('arg_parser')
    logger.info('Running args: %s', str(args.__dict__))
    utils.fix_seed(args.seed)
    utils.set_target_gpu(args.gpu)
    if args.reset_log_dir:
        utils.reset_dir(args.log_dir)
    if args.reset_weight_dir:
        utils.reset_dir(args.weight_dir)
    # torch.autograd.set_detect_anomaly(True)
    p = data_preprocess.Dataset(args.dataset, batch_size=args.batch_size,
                                neg_sample_size=args.neg_sample_size, test_neg_sample_size=args.test_neg_sample_size,
                                undirected_graph=not args.directed_graph, predict_time=args.predict_time,
                                predict_step=args.n_predict_steps)
    postfix_str = f'l2reg{args.l2_reg}_lr{args.lr}_neg{args.neg_sample_size},{args.test_neg_sample_size}' \
                  f'_bs{args.batch_size}{"" if args.directed_graph else "_ug"}_step{p.predict_step}'
    if args.extra_log_postfix is not None and len(args.extra_log_postfix) > 0:
        postfix_str += f'_{args.extra_log_postfix}'
    prefix_str = args.dataset
    if args.extra_log_prefix is not None and len(args.extra_log_prefix) > 0:
        prefix_str += f'_{args.extra_log_prefix}'
    nn = model.HGSocialRec(n_users=p.n_users, embedding_size=args.embedding_size, subgraphs=p.subgraphs,
                           n_hgcn_layers=args.n_hgcn_layers, dropout=args.dropout, dropout_graph=args.dropout_graph,
                           predict_time=p.predict_time, n_transformer_layers=args.n_transformer_layers,
                           use_positional_embedding=not args.no_positional_embedding,
                           n_line_graph_layers=args.n_line_conv_layers,
                           include_line_graph_weight=args.include_line_conv_weight,
                           model_name_prefix=prefix_str, model_name_postfix=postfix_str)
    nn = model.to_cuda(nn)
    if args.load_weight is not None:
        # restore previous run time
        assert os.path.exists(args.load_weight), 'Parameter load_weight not exists'
        if os.path.isfile(args.load_weight):
            par_dir = os.path.dirname(args.load_weight)
        else:
            par_dir = args.load_weight
        ptn = re.compile(r'^(\d{6}_\d{6})')
        match = re.search(ptn, os.path.basename(par_dir))
        if match is not None:
            now_str = match.group(1)
        else:
            logger.warning('Could not restore previous run time from "load_weight" path, use current time')
    run_id = f'{now_str}_{nn.model_name}'
    os.makedirs('running', exist_ok=True)  # prevents collecting running logs and weights in bash script
    open(os.path.join('running', run_id), 'wb').close()
    summary_writer = tb.SummaryWriter(os.path.join(args.log_dir, run_id)) if not args.disable_log else None
    weight_dir = os.path.join(args.weight_dir, run_id)
    logger.info('Model weights will be saved to "%s"', weight_dir)
    try:
        train(p, nn, args.epochs, args.lr, args.l2_reg, args.evaluate_per_epochs, weight_dir, summary_writer,
              args.load_weight, args.max_keep_weights, p.predict_step, args.log_embedding_histogram)
    except Exception:
        raise
    finally:
        if os.path.isfile(os.path.join('running', run_id)):
            os.remove(os.path.join('running', run_id))


if __name__ == '__main__':
    main()
