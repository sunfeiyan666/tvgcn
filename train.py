import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
import sys
import configparser


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='./data/toy/timeini',
                    help='data path')
parser.add_argument('--adj_filename', type=str, default='data/PEMS08/distance.csv', help='adj filename path')
parser.add_argument('--save', type=str, default='data/toy', help='save path')
parser.add_argument('--gcn_bool', action='store_false', default='true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', default='true', help='whether only adaptive adj')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=25, help='inputs dimension')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--print_every', type=int, default=200, help='')
parser.add_argument('--static_feature_bool', action='store_true', default='false', help='whether to add static feature')
parser.add_argument('--num_of_vertices',type=int,default=20,help='number of vertices/variables')
parser.add_argument('--ksize',type=int,default=20,help='the nearest k nodes as neighbors')
parser.add_argument('--tanhalpha',type=float,default=3,help='adj of original feature alpha')
parser.add_argument('--time_embedding_bool', action='store_true', default='true', help='whether time embedding')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

args = parser.parse_args()

def main(expid):
    device = torch.device(args.device)

    gcn_bool = args.gcn_bool

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    num_of_vertices = args.num_of_vertices
    if args.aptonly != "true":
        adp = util.get_adjacency_matrix(args.adj_filename, num_of_vertices)

        adp = torch.Tensor(adp).to(device)
        supports = [torch.tensor(adp).to(device)]  # physical connection
        # print(adp.size())
    else:
        supports = []
        print("0", len(supports))
    # print(args)

    scaler = dataloader['scaler']

    engine = trainer(scaler, args.in_dim, args.seq_length, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, args.num_of_vertices, args.ksize, args.tanhalpha, device, supports, gcn_bool, args.time_embedding_bool)
    print(sum(p.numel() for p in engine.model.parameters() if p.requires_grad))
    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []

    #torch.load("pems04-24best_exp_0_best_epoch_89_20.72.pth")

    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3) #(b,t,n,f)->(b,f,n,t)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)  #(b,t,n,f)->(b,f,n,t)
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
        t2 = time.time()
        train_time.append(t2 - t1)
        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
              flush=True)
        torch.save(engine.model.state_dict(),
                   args.save + "exp_" + str(expid) + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    #testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(
        torch.load(args.save + "exp_" + str(expid) + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))

    # engine.model.load_state_dict(
    #     torch.load("pems08-24best_exp_0_best_epoch_85_16.83.pth"))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]
    print("realy",realy.shape)#3389,307,24
    engine.model.eval()
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        #print("testx",testx.shape)  #(32,3,307,24)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
            #print("preds", preds.shape)  # 32,1,137,12
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]
    #print("yhat", yhat.shape)

    print("Training finished")
    # print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    amae = []
    amape = []
    armse = []
    for i in range(args.seq_length):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        #print("pred",pred.shape)
        #print("real", real.shape)
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    print('best epoch: ', str(bestid))
    torch.save(engine.model.state_dict(),
               args.save + 'best_exp_' + str(expid) + '_best_epoch_' + str(bestid) +
               '_' + str(round(his_loss[bestid], 2)) + ".pth")

if __name__ == "__main__":
    t1 = time.time()
    # for i in range(3):
    main(0)
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
