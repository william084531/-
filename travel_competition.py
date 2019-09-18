import pandas as pd
import numpy as np
from csv import reader
from pandas.core.frame import DataFrame
#from itertools import chain
import torch
import torch.utils.data as Data
import torch.nn as nn
from torch.autograd import Variable
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import os
postsite = []
post_n = []
position = []
title = []
site = ['order','airline','cache_map','day_schedule','group','training-set']

def search(path, word):
    for filename in os.listdir(path):
        fp = os.path.join(path, filename)
        if os.path.isfile(fp) and word in filename:
            postsite.append(fp)
        elif os.path.isdir(fp):
            search(fp, word)
for i in range(len(site)):
    search('D:\參賽資料\dataset',site[i])
def process_position():
  for t in range(len(postsite)):
    with open (postsite[t],"r",encoding ="utf-8") as f:
      data = reader(f,delimiter=',')
      post = []#  在此處在做[]可以讓post內的數值不斷更新
      for row in data:
        post.append(row)
      post_n.append(post[1:len(post)])
      title.append(post[0])
#  post_ch = list(chain(*post_n))
  for i in range(len(post_n)):
    c = DataFrame(post_n[i],columns = title[i])
#  c = c.drop(c.columns[0:30:3], axis=1).fillna('-90')
    position.append(c)
process_position()
filter = (position[0]['order_id'].isin(position[5]['order_id']))
filt_notin = [not i for i in filter] #True,False 翻轉
train_set = position[0][filter]
test_set = position[0][filt_notin]
train_set = pd.merge(position[4],train_set)
test_set = pd.merge(position[4],test_set)
#%%
order_da = []
order_od = []
train_set = train_set.replace(regex={r'^subline_value_':'','area_value_':'','src1_value_':'','src2_value_':'','unit_value_':''})
train_data = train_set.drop(['product_name','promotion_prog','order_id','group_id'], axis=1)
test_set = test_set.replace(regex={r'^subline_value_':'','area_value_':'','src1_value_':'','src2_value_':'','unit_value_':''})
test_data = test_set.drop(['product_name','promotion_prog','order_id','group_id'], axis=1)
dd = train_data['begin_date']
od = test_data['begin_date']
for i in range(len(dd)):
  ord_B = dd.iloc[i]+' 17:58:28 +0700'
  order_da.append(ord_B)
for i in range(len(od)):
  ord_D = od.iloc[i]+' 17:58:28 +0700'
  order_od.append(ord_D)
#%%
from email.utils import parsedate_tz
#from datetime import datetime
order_date_full = []#begin_date
order_date_fuls = []#order_date
order_date_use = []#begin_use
order_date_uss = []#order_use
#delta_date = []
for i in range(len(order_da)):
  order_dat = parsedate_tz(order_da[i])
  order_date_full.append(order_dat)
for i in range(len(order_od)):
  order_dats = parsedate_tz(order_od[i])
  order_date_fuls.append(order_dats)
for i in range(len(order_date_full)):
  order = order_date_full[i][0:3]
  order_date_use.append(list(order))
for i in range(len(order_date_fuls)):
  orders = order_date_fuls[i][0:3]
  order_date_uss.append(list(orders))
label = position[5]['deal_or_not']
labels = np.array([label],float)
#for i in range(len(order_date_uss)):
#  date_delta = datetime(order_date_use[i][0],order_date_use[i][1],order_date_use[i][2])-datetime(order_date_uss[i][0],order_date_uss[i][1],order_date_uss[i][2])
#  delta_date.append(str(date_delta)[0:2])
#for i in range(len(delta_date)):
#  if delta_date[i] =='0:':
#    delta_date[i] ='0'
#delta_date = DataFrame(delta_date,columns = ['delta_date'])
order_date_use = DataFrame(order_date_use,columns = ['begin_year','begin_month','begin_date'])
order_date_uss = DataFrame(order_date_uss,columns = ['begin_year','begin_month','begin_date'])
#%%
train_data = train_data.drop(['order_date','begin_date'], axis=1)
#train_data = train_data.drop(['order_date'], axis=1)
train_data = pd.concat([train_data,order_date_use,label],axis = 1)
train_data_1 = train_data[train_data['deal_or_not']!='0']
train_data_0 = train_data[train_data['deal_or_not']!='1']
train_data_try = DataFrame.sample(train_data_0,n=len(train_data_1), frac=None, replace=False, weights=None, random_state=None, axis=None)
train_data_u = pd.concat([train_data_1,train_data_try],axis = 0)
test_set_data = test_data.drop(['order_date','begin_date'], axis=1)
#train_data = train_data.drop(['order_date'], axis=1)
test_set_data = pd.concat([test_set_data,order_date_uss],axis = 1)
test_set_data = test_set_data.drop(['sub_line','area','price'], axis=1)
#%%
training_use = DataFrame.sample(train_data_u,n=None, frac=0.7, replace=False, weights=None, random_state=None, axis=None)
indexs = list(training_use.index)
test_validation = train_data_u.drop(index=indexs, axis=1, inplace=False)
test_use = DataFrame.sample(test_validation,n=None, frac=0.5, replace=False, weights=None, random_state=None, axis=None)
indexs1 = list(test_use.index)
validation = test_validation.drop(index=indexs1, axis=1, inplace=False)
#%%
training_target = training_use['deal_or_not']
test_target = test_use['deal_or_not']
validation_target = validation['deal_or_not']
train_price = training_use['price'].astype(float)/10000
test_price = test_use['price'].astype(float)/10000
validation_price = validation['price'].astype(float)/10000
test_set_price = test_data['price'].astype(float)/10000
training_use = training_use.drop(['deal_or_not','begin_year','price'], axis=1)
test_use = test_use.drop(['deal_or_not','begin_year','price'], axis=1)
validation = validation.drop(['deal_or_not','begin_year','price'], axis=1)
test_set_data = pd.concat([test_set_data,test_set_price],axis = 1)
test_set_data = test_set_data.drop(['begin_year','price'], axis=1)
training_use = pd.concat([training_use,train_price],axis = 1)
test_use = pd.concat([test_use,test_price],axis = 1)
validation = pd.concat([validation,validation_price],axis = 1)
training_use = training_use.drop(['sub_line','area','price'], axis=1)
test_use = test_use.drop(['sub_line','area','price'], axis=1)
validation = validation.drop(['sub_line','area','price'], axis=1)
#training_use = training_use.drop(['deal_or_not','order_year'], axis=1)
#test_use = test_use.drop(['deal_or_not','order_year'], axis=1)
#validation = validation.drop(['deal_or_not','order_year'], axis=1)
#%%
training_use = torch.from_numpy(training_use.values.astype(float)).type(torch.FloatTensor)
training_targets = np.array([training_target],float).T
training_target_ = torch.from_numpy(training_targets.reshape(len(training_target))).type(torch.LongTensor)
test_use = torch.from_numpy(test_use.values.astype(float)).type(torch.FloatTensor)
test_targets = np.array([test_target],float).T
test_target_ = torch.from_numpy(test_targets.reshape(len(test_target))).type(torch.LongTensor)
validation = torch.from_numpy(validation.values.astype(float)).type(torch.FloatTensor)
validation_targets = np.array([validation_target],float).T
validation_target_ = torch.from_numpy(validation_targets.reshape(len(validation_target))).type(torch.LongTensor)
test_set_data = torch.from_numpy(test_set_data.values.astype(float)).type(torch.FloatTensor)
training_data = Data.TensorDataset(training_use,training_target_)
test_data = Data.TensorDataset(test_use,test_target_)
validation_data = Data.TensorDataset(validation,validation_target_)
train_loader = Data.DataLoader(dataset=training_data, batch_size=100, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=100, shuffle=True)
validation_loader = Data.DataLoader(dataset=validation_data, batch_size=100, shuffle=True)
#%%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(7, 80) 
#        self.norm = nn.BatchNorm1d(80)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(80, 80)  
#        self.norm = nn.BatchNorm1d(80)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc3 = nn.Linear(80, 80)  
#        self.norm = nn.BatchNorm1d(80)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc4 = nn.Linear(80, 80)  
#        self.norm = nn.BatchNorm1d(80)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc5 = nn.Linear(80, 80)  
#        self.norm = nn.BatchNorm1d(80)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc6 = nn.Linear(80, 2)  
#        self.relu = nn.ReLU()
#        self.fc7 = nn.Linear(80, 80)  
#        self.relu = nn.ReLU()
#        self.fc8 = nn.Linear(80, 80)  
#        self.relu = nn.ReLU()
#        self.fc9 = nn.Linear(80, 80)  
#        self.relu = nn.ReLU()
#        self.fc10 = nn.Linear(80, 2)  
#        self.relu = nn.ReLU()
#        self.fc11 = nn.Linear(250, 250)  
#        self.relu = nn.ReLU()
#        self.fc12 = nn.Linear(250, 125)  
#        self.relu = nn.ReLU()
#        self.fc13 = nn.Linear(125, 125)  
#        self.relu = nn.ReLU()
#        self.fc14 = nn.Linear(125, 75)  
#        self.relu = nn.ReLU()
#        self.fc6 = nn.Linear(80, 15)  
#        self.fc15 = nn.Linear(75, 2)  
    def forward(self, x):
        out = self.fc1(x)
#        out = self.norm(out)
        out = self.relu(out)
#        out = self.drop(out)
        out = self.fc2(out)
#        out = self.norm(out)
        out = self.relu(out)
#        out = self.drop(out)
        out = self.fc3(out)
#        out = self.norm(out)
        out = self.relu(out)
#        out = self.drop(out)
        out = self.fc4(out)
#        out = self.norm(out)
        out = self.relu(out)
#        out = self.drop(out)
        out = self.fc5(out)
#        out = self.norm(out)
        out = self.relu(out)
#        out = self.drop(out)
        out = self.fc6(out)
        return out
#        out = self.relu(out)
#        out = self.fc7(out)
#        out = self.relu(out)
#        out = self.fc8(out)
#        out = self.relu(out)
#        out = self.fc9(out)
#        out = self.relu(out)
#        out = self.fc10(out)
#        out = self.relu(out)
#        out = self.fc11(out)
#        out = self.relu(out)
#        out = self.fc12(out)
#        out = self.relu(out)
#        out = self.fc13(out)
#        out = self.relu(out)
#        out = self.fc14(out)
#        out = self.relu(out)
#        out = self.fc15(out)

#net = nn.Sequential(
#    nn.Linear(8, 40),
#    nn.ReLU(),
#    nn.Linear(40, 20),
#    nn.ReLU(),
#    nn.Linear(20, 10),
#    nn.ReLU(),
#    nn.Linear(10, 15)
#)
net = Net()
#net.load_state_dict(torch.load('params_post_try.pkl')
#net.load_state_dict(torch.load('參賽_try_mlp_3 (3).pkl'))
#net.load_state_dict(torch.load('params_參賽_try_mlp_1.pkl'))
#net.load_state_dict(torch.load('params_post_較好的模型.pkl'))
#net.load_state_dict(torch.load('參賽_mlp.pkl'))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001) # 使用随机梯度下降，学习率 0.1
#%%
losses = []
acces = []
eval_losses = []
eval_acces = []
vali_acces = []
ys = []
yl = []
pre = []
lab = []
for e in range(300):
    train_loss = 0
    train_acc = 0
    net.train()
    for im, label in train_loader:
        im = Variable(im)
        label = Variable(label)
        # 前向传播
        out = net(im)
        loss = criterion(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        train_acc += acc
        
    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
#    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    net.eval() # 将模型改为预测模式
    for im, label in test_loader:
        im = Variable(im)
        label = Variable(label)
        out = net(im)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        eval_acc += acc
        ys.append(out.data.numpy())
        yl.append(label.cpu().numpy())
        pre.append(pred)
        lab.append(label)
    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))
    vali_loss = 0
    vali_acc = 0
    net.eval() # 将模型改为预测模式
    for im, label in validation_loader:
        im = Variable(im)
        label = Variable(label)
        out = net(im)
        loss = criterion(out, label)
        # 记录误差
        vali_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        vali_acc += acc
        ys.append(out.data.numpy())
        yl.append(label.cpu().numpy())
        pre.append(pred)
        lab.append(label)
#    eval_losses.append(eval_loss / len(test_loader))
    vali_acces.append(vali_acc / len(validation_loader))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f},Vali Acc: {:.6f}'
          .format(e, train_loss / len(train_loader), train_acc / len(train_loader), 
                     eval_loss / len(test_loader), eval_acc / len(test_loader),vali_acc/len(validation_loader)))
#torch.save(net.state_dict(), 'params_參賽_try_mlp_1.pkl')
#%%
pred_as = []
sacces = []
test_output = net(test_set_data)
_, pred = test_output.max(1)
#num_correct = (pred == training_target_).sum().item()
#acc = num_correct / training_use.shape[0]
pred = pred.numpy()
pred = torch.max(test_output, 1)[1].data.numpy().squeeze()
predss = pd.DataFrame(pred)
pred_as.append(pred)
#%%
file_path = r'D:/python/python/參賽.xlsx'
writer = pd.ExcelWriter(file_path)
predss.to_excel(writer,index=True,encoding='utf-8')
writer.save()
#%%
order1_da = []
test_set = test_set.replace(regex={r'^subline_value_':'','area_value_':'','src1_value_':'','src2_value_':'','unit_value_':''})
test_data = test_set.drop(['product_name','promotion_prog','order_id','group_id'], axis=1)
dd1 = test_data['order_date']
for i in range(len(dd1)):
  ord_D1 = dd1.iloc[i]+' 17:58:28 +0700'
  order1_da.append(ord_D1)
#%%
from email.utils import parsedate_tz
order_date_full1 = []
order_date_use1 = []
for i in range(len(order1_da)):
  order_dat1 = parsedate_tz(order1_da[i])
  order_date_full1.append(order_dat1)
for i in range(len(order_date_full1)):
  order = order_date_full1[i][0:3]
  order_date_use1.append(list(order))
#label = position[5]['deal_or_not']
#labels = np.array([label],float)
order_date_use1 = DataFrame(order_date_use1,columns = ['order_year','order_month','order_date'])
test_data = test_data.drop(['order_date','begin_date'], axis=1)
test_data = pd.concat([test_data,order_date_use1],axis = 1)
test_data = torch.from_numpy(test_data.values.astype(float)).type(torch.FloatTensor)






#%%
from scipy import interp
yls = yl[0]
yss = ys[0]
for i in range(len(yl)-1):
  yls = np.concatenate((yls, yl[i+1]), axis=0)
  yss = np.concatenate((yss, ys[i+1]), axis=0)
yls = label_binarize(yls, classes=[0, 1])
n_classes = yls.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = metrics.roc_curve(yls[:, i], yss[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])
fpr["micro"], tpr["micro"], _ = metrics.roc_curve(yls.ravel(), yss.ravel())
roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))


# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
#%%
import matplotlib.pyplot as plt
#看單一分類的ROC曲線
#plt.figure()
#lw = 2
#plt.plot(fpr[2], tpr[2], color='darkorange',
#         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
#plt.show()
#%%
from itertools import cycle
plt.figure()
#單獨看micro 微平均的數值(看模型的整體預測能力)
#plt.plot(fpr["micro"], tpr["micro"],
#         label='micro-average ROC curve (area = {0:0.2f})'
#               ''.format(roc_auc["micro"]),
#         color='deeppink', linestyle=':', linewidth=4)
##單獨看macro 宏平均的數值(看模型在個別類別預測能力的平均)
#plt.plot(fpr["macro"], tpr["macro"],
#         label='macro-average ROC curve (area = {0:0.2f})'
#               ''.format(roc_auc["macro"]),
#         color='darkorange', linestyle=':', linewidth=4)
#可以看到每個類別的roc曲線和AUC數值
colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','green','blue','yellow','black','pink'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
#%%
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
pres = pre[0]
labs = lab[0]
for i in range(len(pre)-1):
  pres = np.concatenate((pres, pre[i+1]), axis=0)
  labs = np.concatenate((labs, lab[i+1]), axis=0)
macro_F1 = f1_score(labs, pres, average='macro') 
micro_F1 = f1_score(labs, pres, average='micro') 
weighted_F1 = f1_score(labs, pres, average='weighted') 
none_F1 = f1_score(labs, pres, average=None) 

macro_preci = precision_score(labs, pres, average='macro') 
micro_preci = precision_score(labs, pres, average='micro') 
weighted_preci = precision_score(labs, pres, average='weighted') 
none_preci = precision_score(labs, pres, average=None) 

macro_recall = recall_score(labs, pres, average='macro') 
micro_recall = recall_score(labs, pres, average='micro') 
weighted_recall = recall_score(labs, pres, average='weighted') 
none_recall = recall_score(labs, pres, average=None) 
#%%
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(yls[:, i],
                                                        yss[:, i])
    average_precision[i] = average_precision_score(yls[:, i], yss[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(yls.ravel(),
    yss.ravel())
average_precision["micro"] = average_precision_score(yls, yss,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))
#%%
plt.figure()
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
         where='post')
plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    .format(average_precision["micro"]))
#%%
from itertools import cycle
# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(7, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision["micro"]))

for i, color in zip(range(n_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
plt.show()

#%%
from sklearn.metrics import confusion_matrix
import itertools
class_names = ['0601','0603','0605','0615','0618','1202','1213','1220','1222']
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

cnf_matrix = confusion_matrix(labs, pres)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
