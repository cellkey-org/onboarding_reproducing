import pandas as pd
import torch
import torch.nn.functional as F


class EarlyStopping:
    def __init__(self, patience=7, delta=0, save_path="best_model.pth"):
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.save_path = save_path
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_wts = model.state_dict()
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(self.best_model_wts, self.save_path)

    def load_best_model(self, model):
        model.load_state_dict(torch.load(self.save_path))
        return model


def batch_pearsonr(x, y):
    mean_x = torch.mean(x, dim=1, keepdim=True)
    mean_y = torch.mean(y, dim=1, keepdim=True)
    xm = x - mean_x
    ym = y - mean_y
    r_num = torch.sum(xm * ym, dim=1)
    r_den = torch.sqrt(torch.sum(xm**2, dim=1) * torch.sum(ym**2, dim=1))
    r = r_num / r_den
    return r


def train_model(model, criterion, optimizer, num_epochs, train_dl, val_dl, device, tb_writer, patience) -> dict:
    res = {"train": list(), "validataion": list()}

    early_stopping = EarlyStopping(patience=patience, delta=0.001, save_path=f"{tb_writer.get_logdir()}/best_model.pth")

    # train/validation
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        cosine_similarities = []
        pearson_coefficients = []

        for sequences, intensities in train_dl:
            sequences = sequences.float().unsqueeze(2).to(device)  # (batch_size, sequence_length, input_size)
            intensities = intensities.float().to(device)

            outputs = model(sequences)

            loss = criterion(outputs, intensities)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cosine_similarity = F.cosine_similarity(outputs, intensities, dim=1)
            cosine_similarities.extend(cosine_similarity.cpu().detach().numpy())

            pearson_coefficient = batch_pearsonr(outputs, intensities)
            pearson_coefficients.extend(pearson_coefficient.cpu().detach().numpy())

        train_loss /= len(train_dl)
        mean_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities)
        mean_pearson_coefficient = sum(pearson_coefficients) / len(pearson_coefficients)

        model.eval()
        val_loss = 0.0
        cosine_similarities_val = []
        pearson_coefficients_val = []
        with torch.no_grad():
            for sequences, intensities in val_dl:
                sequences = sequences.float().unsqueeze(2).to(device)
                intensities = intensities.float().to(device)

                outputs = model(sequences)
                loss = criterion(outputs, intensities)
                val_loss += loss.item()

                # 코사인 유사도 계산 (Validation)
                cosine_similarity_val = F.cosine_similarity(outputs, intensities, dim=1)
                cosine_similarities_val.extend(cosine_similarity_val.cpu().detach().numpy())

                # 벡터화된 피어슨 상관 계수 계산 (Validation)
                pearson_coefficient_val = batch_pearsonr(outputs, intensities)
                pearson_coefficients_val.extend(pearson_coefficient_val.cpu().detach().numpy())

        val_loss /= len(val_dl)
        mean_cosine_similarity_val = sum(cosine_similarities_val) / len(cosine_similarities_val)
        mean_pearson_coefficient_val = sum(pearson_coefficients_val) / len(pearson_coefficients_val)

        res["train"].append([epoch, train_loss, mean_cosine_similarity, mean_pearson_coefficient])
        res["validataion"].append([epoch, val_loss, mean_cosine_similarity_val, mean_pearson_coefficient_val])

        tb_writer.add_scalar("Loss/Train", train_loss, epoch)
        tb_writer.add_scalar("Loss/Validation", val_loss, epoch)
        tb_writer.add_scalar("Cosine Similarity/Train", mean_cosine_similarity, epoch)
        tb_writer.add_scalar("Cosine Similarity/Validation", mean_cosine_similarity_val, epoch)
        tb_writer.add_scalar("PCC/Train", mean_pearson_coefficient, epoch)
        tb_writer.add_scalar("PCC/Validation", mean_pearson_coefficient_val, epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"Train Cosine Similarity: {mean_cosine_similarity:.4f}, Train PCC: {mean_pearson_coefficient:.4f}")
        print(
            f"Validation Cosine Similarity: {mean_cosine_similarity_val:.4f}, Validation PCC: {mean_pearson_coefficient_val:.4f}"
        )

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Stop Early")
            break

    print("Training complete.")

    model = early_stopping.load_best_model(model)

    return model, res


def test_model(model, criterion, test_dl, device, tb_writer, detail: str):
    res = {"test": list()}
    # test
    model.eval()
    test_loss = 0.0
    cosine_similarities = []
    pearson_coefficients = []

    with torch.no_grad():
        for sequences, intensities in test_dl:
            sequences = sequences.float().unsqueeze(2).to(device)
            intensities = intensities.float().to(device)

            outputs = model(sequences)
            loss = criterion(outputs, intensities)
            test_loss += loss.item()

            # 코사인 유사도 계산 (Validation)
            cosine_similarity = F.cosine_similarity(outputs, intensities, dim=1)
            cosine_similarities.extend(cosine_similarity.cpu().detach().numpy())

            # 벡터화된 피어슨 상관 계수 계산 (Validation)
            pearson_coefficient = batch_pearsonr(outputs, intensities)
            pearson_coefficients.extend(pearson_coefficient.cpu().detach().numpy())

    # 평균 코사인 유사도 및 피어슨 상관 계수 계산
    test_loss /= len(test_dl)
    mean_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities)
    mean_pearson_coefficient = sum(pearson_coefficients) / len(pearson_coefficients)

    res["test"].append([test_loss, mean_cosine_similarity, mean_pearson_coefficient])
    print(f"Test: Mean Cosine Similarity: {mean_cosine_similarity:.4f}")
    print(f"Test: Mean Pearson Correlation Coefficient: {mean_pearson_coefficient:.4f}")

    tb_writer.add_scalar(f"Loss/Test {detail}", test_loss)
    tb_writer.add_scalar(f"Cosine Similarity/Test {detail}", mean_cosine_similarity)
    tb_writer.add_scalar(f"PCC/Test {detail}", mean_pearson_coefficient)

    return res
