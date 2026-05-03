from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_INDEX = 0
UNK_INDEX = 1


def build_char_vocab(code_samples, min_freq=1):
    counter = Counter()
    for code in code_samples:
        counter.update(code)

    vocab = {PAD_TOKEN: PAD_INDEX, UNK_TOKEN: UNK_INDEX}
    for char, count in sorted(counter.items()):
        if count >= min_freq:
            vocab[char] = len(vocab)

    return vocab


def build_label_map(labels):
    return {label: index for index, label in enumerate(sorted(set(labels)))}


def encode_text(text, vocab, max_len):
    token_ids = [vocab.get(char, UNK_INDEX) for char in text[:max_len]]

    if len(token_ids) < max_len:
        token_ids.extend([PAD_INDEX] * (max_len - len(token_ids)))

    return token_ids


class CodeDataset(Dataset):
    def __init__(self, code_samples, labels, vocab, label_to_id, max_len):
        self.code_samples = code_samples
        self.labels = labels
        self.vocab = vocab
        self.label_to_id = label_to_id
        self.max_len = max_len

    def __len__(self):
        return len(self.code_samples)

    def __getitem__(self, index):
        token_ids = encode_text(self.code_samples[index], self.vocab, self.max_len)
        label_id = self.label_to_id[self.labels[index]]
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(label_id, dtype=torch.long),
        )


class CharCNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        embedding_dim=64,
        num_filters=128,
        kernel_sizes=(3, 5, 7),
        dropout=0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_INDEX)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        embedded = embedded.transpose(1, 2)

        pooled_outputs = []
        for conv in self.convs:
            features = torch.relu(conv(embedded))
            pooled = torch.max(features, dim=2).values
            pooled_outputs.append(pooled)

        features = torch.cat(pooled_outputs, dim=1)
        features = self.dropout(features)
        return self.classifier(features)


def make_data_loader(code_samples, labels, vocab, label_to_id, max_len, batch_size, shuffle):
    dataset = CodeDataset(code_samples, labels, vocab, label_to_id, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_cnn(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for token_ids, labels in data_loader:
            token_ids = token_ids.to(device)
            labels = labels.to(device)

            logits = model(token_ids)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total if total else 0.0


def train_cnn(
    train_x,
    train_y,
    valid_x,
    valid_y,
    checkpoint_path,
    max_len=1000,
    batch_size=64,
    epochs=5,
    learning_rate=1e-3,
    min_char_freq=1,
    device_name=None,
):
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    vocab = build_char_vocab(train_x, min_freq=min_char_freq)
    label_to_id = build_label_map(train_y)
    id_to_label = {index: label for label, index in label_to_id.items()}

    train_loader = make_data_loader(
        train_x, train_y, vocab, label_to_id, max_len, batch_size, shuffle=True
    )
    valid_loader = make_data_loader(
        valid_x, valid_y, vocab, label_to_id, max_len, batch_size, shuffle=False
    )

    model = CharCNNClassifier(vocab_size=len(vocab), num_classes=len(label_to_id)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for token_ids, labels in train_loader:
            token_ids = token_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(token_ids)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        valid_accuracy = evaluate_cnn(model, valid_loader, device)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"valid_accuracy={valid_accuracy:.4f}"
        )

        if valid_accuracy >= best_accuracy:
            best_accuracy = valid_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "label_to_id": label_to_id,
                    "id_to_label": id_to_label,
                    "max_len": max_len,
                    "config": {
                        "embedding_dim": 64,
                        "num_filters": 128,
                        "kernel_sizes": (3, 5, 7),
                        "dropout": 0.3,
                    },
                    "valid_accuracy": valid_accuracy,
                },
                checkpoint_path,
            )
            print(f"Saved CNN checkpoint to {checkpoint_path}")

    print(f"Best validation accuracy: {best_accuracy:.4f}")


def load_cnn_checkpoint(checkpoint_path, device_name=None):
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = CharCNNClassifier(
        vocab_size=len(checkpoint["vocab"]),
        num_classes=len(checkpoint["label_to_id"]),
        embedding_dim=config["embedding_dim"],
        num_filters=config["num_filters"],
        kernel_sizes=config["kernel_sizes"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, device


def predict_cnn(model, checkpoint, device, code):
    token_ids = encode_text(code, checkpoint["vocab"], checkpoint["max_len"])
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, label_id = torch.max(probabilities, dim=0)

    language = checkpoint["id_to_label"][int(label_id.item())]
    return language, float(confidence.item())
