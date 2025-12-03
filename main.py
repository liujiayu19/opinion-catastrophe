import torch
import numpy as np
from predata import read_data
from models import Linear, GCN, HGCN
from Utils import _generate_HG_from_H, _generate_G_from_H, evaluation, ROC
from sklearn.svm import SVC

lr = 0.1
epochs = 1000
model_name = "HGCN"
x, y, h, labeled = read_data()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if model_name == "SVM":
    model = SVC(probability=True)
elif model_name == "Linear":
    hidden_dim = 64
    model = Linear.Linear(11, 64, 128, 2).to(device)
    is_graph = False
elif model_name == "GCN":
    lr = 0.1
    g = torch.tensor(_generate_G_from_H(h), dtype=torch.float32)
    hidden_dim = 64
    model = GCN.GCN(11, 64, 256, 2).to(device)
    is_graph = True
elif model_name == "HGCN":
    g = torch.tensor(_generate_HG_from_H(h), dtype=torch.float32)
    hidden_dim = 64
    model = HGCN.HGNN(11, 2, 64).to(device)
    is_graph = True
criterion = torch.nn.CrossEntropyLoss()
if model_name!="SVM":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

best_P, best_R, best_F1, best_accuracy = 0, 0, 0, 0
if model_name == "GCN" or model_name == "HGCN":
    for i in range(epochs):
        pre = model(x.to(device), g.to(device))
        pre = torch.cat((pre, labeled.to(device)), 1)
        y = torch.cat((y.to(device), labeled.to(device)), 1)
        a = pre[pre[:, 2] == 0][:, :2]
        b = y[y[:, 1] == 0][:, :1]
        loss = criterion(a, b.long().squeeze())
        P, R, F1, accuracy = evaluation(pre[pre[:, 2] == 1][:, :2].argmax(1).cpu(), y[y[:, 1] == 1][:, :1].cpu())
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            print("best_P: ", P, "best_R: ", R, "best_F1: ", F1, "best_accuracy: ", accuracy)
            torch.save({'NN': model.state_dict()}, f'./savefile/{model_name}/{model_name}.pt')
       # print(loss)
        optimizer.zero_grad()  # 重置模型参数的梯度,默认情况下梯度会迭代相加
        loss.backward()  # 反向传播预测损失，计算梯度
        optimizer.step()  # 梯度下降，w = w - lr * 梯度。随机梯度下降是迭代的，通过随机噪声能避免鞍点的出现
    # 计算ROC曲线
    ROC(model_name, y[y[:, 1] == 1][:, :1].cpu().detach().numpy(), pre[pre[:, 2] == 1][:, :2].cpu().detach().numpy()[:, 1])

elif model_name=="Linear" or model_name=="SVM":
    x = torch.cat((x.to(device), labeled.to(device)), 1)
    y = torch.cat((y.to(device), labeled.to(device)), 1)
    train_x = x[x[:, -1] == 0][:, :11].to(device)
    train_y = y[y[:, -1] == 0][:, :1].to(device)
    test_x = x[x[:, -1] == 1][:, :11].to(device)
    test_y = y[y[:, -1] == 1][:, :1].to(device)
    if model_name == "Linear":
        for i in range(epochs):
            # train
            pre = model(train_x)
            loss = criterion(pre, train_y.long().squeeze())
            optimizer.zero_grad()  # 重置模型参数的梯度,默认情况下梯度会迭代相加
            loss.backward()  # 反向传播预测损失，计算梯度
            optimizer.step()  # 梯度下降，w = w - lr * 梯度。随机梯度下降是迭代的，通过随机噪声能避免鞍点的出现

            # test
            with torch.no_grad():
                pre = model(test_x)
            #   print(pre, test_y)
                P, R, F1, accuracy = evaluation(pre.argmax(1).cpu(), test_y.cpu().long().squeeze())
                if best_accuracy < accuracy:
                    best_accuracy = accuracy
                    print("best_P: ", P, "best_R: ", R, "best_F1: ", F1, "best_accuracy: ", accuracy)
                    torch.save({'NN': model.state_dict()}, f'./savefile/{model_name}/{model_name}.pt')
        ROC(model_name, test_y.cpu(), pre.cpu()[:, 1])

    elif model_name == "SVM":
        model.fit(train_x.cpu(), train_y.cpu())
        pre = model.predict(test_x.cpu())
        P, R, F1, accuracy = evaluation(pre, test_y.cpu().long().squeeze())
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            print("best_P: ", P, "best_R: ", R, "best_F1: ", F1, "best_accuracy: ", accuracy)







