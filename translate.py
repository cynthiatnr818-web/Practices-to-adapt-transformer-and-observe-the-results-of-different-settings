import math
import torch
import torch.nn as nn

device = torch.device("cpu") # 推理极快，直接用 CPU 避免兼容问题
PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2

# ==========================================
# 1. 必须保留与训练时完全一致的模型结构定义
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SimpleTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=0.1, batch_first=True
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        src_pad_mask = (src == PAD_IDX)
        tgt_pad_mask = (tgt == PAD_IDX)
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        out = self.transformer(
            src=src_emb, tgt=tgt_emb, tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )
        return self.fc_out(out)

# ==========================================
# 2. 加载权重和词典
# ==========================================
checkpoint = torch.load("my_transformer.pt", map_location=device, weights_only=False)
src_vocab = checkpoint["src_vocab"]
tgt_vocab = checkpoint["tgt_vocab"]
tgt_idx2word = {idx: word for word, idx in tgt_vocab.items()}

# 实例化空模型，并把训练好的参数填进去
model = SimpleTransformer(len(src_vocab), len(tgt_vocab)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval() # 切换到评估模式（关闭 Dropout）

# ==========================================
# 3. 翻译函数与交互循环
# ==========================================
def translate(sentence, max_len=15):
    words = sentence.strip().lower().split()
    # 检查是否有词典里没见过的生词
    unknown_words = [w for w in words if w not in src_vocab]
    if unknown_words:
        return f"[报错] 词典里没有这些词: {unknown_words}"

    src_indices = [BOS_IDX] + [src_vocab[w] for w in words] + [EOS_IDX]
    src = torch.tensor([src_indices], dtype=torch.long).to(device)
    tgt_indices = [BOS_IDX]

    with torch.no_grad(): # 推理时关闭梯度计算，省内存且更快
        for _ in range(max_len):
            tgt = torch.tensor([tgt_indices], dtype=torch.long).to(device)
            logits = model(src, tgt)
            next_token = logits[0, -1, :].argmax().item()
            if next_token == EOS_IDX:
                break
            tgt_indices.append(next_token)

    return " ".join(tgt_idx2word[idx] for idx in tgt_indices[1:])

if __name__ == "__main__":
    print("模型加载完成！输入英文句子进行翻译（输入 q 退出）：")
    while True:
        text = input("\n请输入英文: ")
        if text.strip().lower() == "q":
            break
        print("中文翻译:", translate(text))