import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np


def train(train_data, test_data, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    for epoch in range(1, args.epochs+1):
        print('\nEpoch %d' % epoch)
        for batch in train_data:
            model.train()
            feature, target = batch
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            logit = model(feature)

            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = np.sum((target - torch.max(logit, 1)[1]).cpu().numpy() == 0)
                accuracy = 100.0 * corrects/target.shape[0]
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             accuracy,
                                                                             corrects,
                                                                             target.shape[0]))
            if steps % args.test_interval == 0:
                dev_acc = eval(test_data, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss, size = 0, 0, 0
    for batch in data_iter:
        feature, target = batch
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)
        avg_loss += loss.item()
        corrects += np.sum((target - torch.max(logit, 1)[1]).cpu().numpy() == 0)
        size += target.shape[0]
        # 400 tests are enough
        if size > 400:
            break

    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
