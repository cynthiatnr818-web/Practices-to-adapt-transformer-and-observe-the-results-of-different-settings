import math
import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device("cpu")
print(f"当前运行设备: {device}")

# ==========================================
# 1. 准备微型英译中数据集与词典
# ==========================================
# 特殊符号：<PAD>用于补齐长度，<BOS>句子开始，<EOS>句子结束
PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2

data = [
    ("i love cats", "我 爱 猫"),
    ("i love dogs", "我 爱 狗"),
    ("you love cats", "你 爱 猫"),
    ("he likes apples", "他 喜欢 苹果"),
    ("she likes cats", "她 喜欢 猫"),
    ("i ate an apple", "我 吃 了 一个 苹果"),
    ("you ate an apple", "你 吃 了 一个 苹果"),
    ("he is happy", "他 很 开心"),
    ("she is happy", "她 很 开心"),
    ("the cat is happy", "这 只 猫 很 开心"),
]

def build_vocab(sentences):
    vocab = {"<PAD>": PAD_IDX, "<BOS>": BOS_IDX, "<EOS>": EOS_IDX}
    for s in sentences:
        for word in s.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

src_vocab = build_vocab([pair[0] for pair in data])
tgt_vocab = build_vocab([pair[1] for pair in data])
tgt_idx2word = {idx: word for word, idx in tgt_vocab.items()}

def encode_sentence(sentence, vocab):
    return [BOS_IDX] + [vocab[w] for w in sentence.split()] + [EOS_IDX]

# 将所有句子转为数字索引，并补齐到相同长度
src_seqs = [encode_sentence(pair[0], src_vocab) for pair in data]
tgt_seqs = [encode_sentence(pair[1], tgt_vocab) for pair in data]

def pad_sequences(seqs):
    max_len = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD_IDX] * (max_len - len(s)) for s in seqs], dtype=torch.long)

src_tensor = pad_sequences(src_seqs).to(device)
tgt_tensor = pad_sequences(tgt_seqs).to(device)


# ==========================================
# 2. 位置编码（完全按照论文 3.5 节的正弦/余弦公式）
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term) # 偶数维用 sin
        pe[:, 1::2] = torch.cos(position * div_term) # 奇数维用 cos
        self.register_buffer("pe", pe.unsqueeze(0))  # 形状: (1, max_len, d_model)

    def forward(self, x):
        # 将词嵌入与位置编码直接相加
        return x + self.pe[:, :x.size(1), :]

# ==========================================
# 3. Transformer 翻译模型
# ==========================================
class SimpleTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.d_model = d_model
        
        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # PyTorch 内置的标准 Encoder-Decoder Transformer 结构
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True  # 让输入形状为 (batch_size, seq_len, d_model)
        )
        
        # 最后的线性层：将 d_model 维映射回目标词典大小
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        # 1. 生成因果掩码（Causal Mask），防止解码器偷看后面的词（设为 -inf）
        tgt_seq_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
        
        # 2. 生成 Padding 掩码，告诉模型忽略用来补齐长度的 <PAD>
        src_pad_mask = (src == PAD_IDX)
        tgt_pad_mask = (tgt == PAD_IDX)
        
        # 3. 词嵌入（按论文 3.4 节乘以 sqrt(d_model)）并加上位置编码
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        # 4. 通过编码器和解码器
        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )
        return self.fc_out(out)


# ==========================================
# 4. 训练模型
# ==========================================
model = SimpleTransformer(len(src_vocab), len(tgt_vocab)).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX) # 计算损失时忽略 <PAD>
optimizer = optim.Adam(model.parameters(), lr=0.005)

print("开始训练...")
model.train()
for epoch in range(1, 301):
    optimizer.zero_grad()
    
    # 对应论文图 1 的 "Outputs (shifted right)"：
    # 解码器输入去掉最后一个词，预测目标去掉第一个 <BOS>
    tgt_input = tgt_tensor[:, :-1]
    tgt_expected = tgt_tensor[:, 1:]
    
    logits = model(src_tensor, tgt_input)
    
    # 计算损失并反向传播更新权重
    loss = criterion(logits.reshape(-1, len(tgt_vocab)), tgt_expected.reshape(-1))
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")


# ==========================================
# 5. 自回归推理（逐词翻译测试）
# ==========================================
def translate(sentence, max_len=15):
    model.eval()
    src = torch.tensor([encode_sentence(sentence, src_vocab)], dtype=torch.long).to(device)
    
    # 解码器一开始只有 <BOS>
    tgt_indices = [BOS_IDX]
    
    with torch.no_grad():
        for _ in range(max_len):
            tgt = torch.tensor([tgt_indices], dtype=torch.long).to(device)
            logits = model(src, tgt)
            
            # 取出最后一个位置概率最大的词
            next_token = logits[0, -1, :].argmax().item()
            if next_token == EOS_IDX:
                break
            tgt_indices.append(next_token)
            
    return " ".join(tgt_idx2word[idx] for idx in tgt_indices[1:])

print("\n=== 翻译测试 ===")
test_sentences = [
    "i love cats",
    "he likes apples",
    "i ate an apple",
    "the cat is happy",
    "she love dogs", # 测试训练集里没见过的组合
]

for s in test_sentences:
    print(f"英文: {s:18s} -> 中文: {translate(s)}")
# 保存模型参数和词典

torch.save({
    "model_state_dict": model.state_dict(),
    "src_vocab": src_vocab,
    "tgt_vocab": tgt_vocab,
}, "my_transformer.pt")

print("模型已保存至 my_transformer.pt")