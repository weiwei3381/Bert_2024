import torch
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from transformers import AdamW
from transformers.optimization import get_scheduler

token = BertTokenizer.from_pretrained('google-bert/bert-base-chinese')  # 加载编码工具
# 定义计算设备，默认使用CPU进行计算
device = 'cpu'
if torch.cuda.is_available():
    device = 'CUDA'


class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        """数据初始化函数

        :param string split: 表示取哪个分片，split的常用取值为训练集“train”，验证集“validation”和测试集“test”
        """
        # 文件路径写死，表示就加载这个数据集
        self.dataset = load_dataset("./data/ChnSentiCorp")[split]

    def __len__(self):
        """
        确保实例能使用len()的魔术方法
        """
        return len(self.dataset)
        
    def __getitem__(self, i):
        """
        确保实例能使用[20]的魔术方法，也可以使用for x in X进行遍历
        """
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']
        return text, label

# 定义数据整理函数
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    
    # 编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True, # 截断
        padding='max_length',  # [PAD]到最大长度
        max_length=500, # 最大长度
        return_tensors='pt',  # 返回pytorch张量
        return_length=True  # 返回长度
    )
    
    # input_ids：编码之后的数字
    input_ids = data['input_ids'].to(device)  #把数据移动到计算设备上
    # attention_mask：补零的位置是0, 其他位置是1
    attention_mask = data['attention_mask'].to(device)  #把数据移动到计算设备上
    # token_type_ids：第1个句子和特殊符号的位置是0, 第2个句子的位置是1
    token_type_ids = data['token_type_ids'].to(device)  #把数据移动到计算设备上
    labels = torch.LongTensor(labels).to(device)  #把数据移动到计算设备上
    return input_ids, attention_mask, token_type_ids, labels

# 定义数据集加载器
dataset = Dataset('train')
loader = torch.utils.data.DataLoader(
    dataset=dataset,  # 表示要加载的数据集，此处使用了之前定义好的训练数据集，所以此处的加载器为训练数据集加载器，区别于测试数据集加载器
    batch_size=16,  # 每个批次中包括16条数据
    collate_fn=collate_fn,  # 要使用的数据整理函数
    shuffle=True,  # 打乱各个批次之间的顺序
    drop_last=True  # 表示当剩余的数据不足16条时，丢弃这些尾数
)

# 加载预训练模型
pretrained = BertModel.from_pretrained('google-bert/bert-base-chinese')
# 冻结bert-base-chinese模型的参数
for param in pretrained.parameters():
    param.requires_grad_(False) # 不训练预训练模型，不需要计算梯度
# 设定计算设备
pretrained.to(device)

# 定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768,2)  # 768到2的全连接层

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
            # 对抽取的特征只取第1个字的结果做分类即可
            out = self.fc(out.last_hidden_state[:, 0])
            out = out.softmax(dim=1)
            return out
        
model = Model()
# 设定计算设备
model.to(device)

# 模型训练
def train():
    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=5e-4)
    # 定义loss函数
    criterion = torch.nn.CrossEntropyLoss()
    # 定义学习率调节器
    scheduler = get_scheduler(name="linear", 
                              num_warmup_steps=0, 
                              num_training_steps=len(loader),
                              optimizer=optimizer)
    # 将模型切换到训练模式
    model.train()
    # 按批次遍历训练其中的数据
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
        # 模型计算
        out = model(input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    token_type_ids=token_type_ids)
        # 计算loss并使用梯度下降法优化模型参数
        loss = criterion(out, labels)
        print(loss)
        loss.requires_grad_(True)
        loss.backward()  # 误差反向传播
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        # 输出各项数据情况
        if i % 10 == 0:
            out = out.argmax(dim=1)
            accuracy = (out == labels).sum().item() / len(labels)
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print(i, loss.item(), lr, accuracy)

if __name__ == "__main__":
    train()
