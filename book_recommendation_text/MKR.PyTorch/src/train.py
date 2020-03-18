#import tensorflow as tf
import numpy as np
from tqdm import tqdm
from model import MKR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter

def train(args, rs_dataset, kg_dataset):

    show_loss = args.show_loss
    show_topk = args.show_topk

    # Get RS data
    n_user = rs_dataset.n_user
    n_item = rs_dataset.n_item
    train_data, eval_data, test_data = rs_dataset.data
    train_indices, eval_indices, test_indices = rs_dataset.indices

    # Get KG data
    n_entity = kg_dataset.n_entity
    n_relation = kg_dataset.n_relation
    kg = kg_dataset.kg

    # Init train sampler
    train_sampler = SubsetRandomSampler(train_indices)

    # Init MKR model
    model = MKR(args, n_user, n_item, n_entity, n_relation)

    # Init Sumwriter
    writer = SummaryWriter(args.summary_path)

    # Top-K evaluation settings
    user_num = 100
    k_list = [1, 2, 5, 10, 20, 50, 100]
    train_record = get_user_record(train_data, True)
    test_record = get_user_record(test_data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    if len(user_list) > user_num:
        user_list = np.random.choice(user_list, size=user_num, replace=False)
    item_set = set(list(range(n_item)))
    step = 0
    for epoch in range(args.n_epochs):
        print("Train RS")
        train_loader = DataLoader(rs_dataset, batch_size=args.batch_size,
                                  num_workers=args.workers, sampler=train_sampler)
        for i, rs_batch_data in enumerate(train_loader):
            loss, base_loss_rs, l2_loss_rs = model.train_rs(rs_batch_data)
            writer.add_scalar("rs_loss", loss.cpu().detach().numpy(), global_step=step)
            writer.add_scalar("rs_base_loss", base_loss_rs.cpu().detach().numpy(), global_step=step)
            writer.add_scalar("rs_l2_loss", l2_loss_rs.cpu().detach().numpy(), global_step=step)
            step += 1
            if show_loss:
                print(loss)

        if epoch % args.kge_interval == 0:
            print("Train KGE")
            kg_train_loader = DataLoader(kg_dataset, batch_size=args.batch_size,
                                         num_workers=args.workers, shuffle=True)
            for i, kg_batch_data in enumerate(kg_train_loader):
                rmse, loss_kge, base_loss_kge, l2_loss_kge = model.train_kge(kg_batch_data)
                writer.add_scalar("kge_rmse_loss", rmse.cpu().detach().numpy(), global_step=step)
                writer.add_scalar("kge_loss", loss_kge.cpu().detach().numpy(), global_step=step)
                writer.add_scalar("kge_base_loss", base_loss_kge.cpu().detach().numpy(), global_step=step)
                writer.add_scalar("kge_l2_loss", l2_loss_kge.cpu().detach().numpy(), global_step=step)
                step += 1
                if show_loss:
                    print(rmse)


        # CTR evaluation
        train_auc, train_acc = model.eval(train_data)
        eval_auc, eval_acc = model.eval(eval_data)
        test_auc, test_acc = model.eval(test_data)

        print('epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
              % (epoch, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))

        # top-K evaluation
        if show_topk:
            precision, recall, f1 = model.topk_eval(user_list, train_record, test_record, item_set, k_list)
            print('precision: ', end='')
            for i in precision:
                print('%.4f\t' % i, end='')
            print()
            print('recall: ', end='')
            for i in recall:
                print('%.4f\t' % i, end='')
            print()
            print('f1: ', end='')
            for i in f1:
                print('%.4f\t' % i, end='')
            print('\n')

def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict
