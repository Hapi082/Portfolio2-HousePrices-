import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import torch
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn

#データ読み込み
train = pd.read_csv("Data/train.csv")
test = pd.read_csv("Data/test.csv")

#不要な列を削除
train.drop(["Id"],axis=1,inplace=True)
test.drop(["Id"],axis=1,inplace=True)

#目的変数を対数変換
y = np.log1p(train.pop("SalePrice"))
X = train.copy()

#特徴量を型ごとに区別
num_cols = X.select_dtypes(include=["int64","Float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns


# 前処理パイプライン
numeric_pipe = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])

categorical_pipe = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="constant",fill_value="Missing")),
    ("encoder",OneHotEncoder(handle_unknown="ignore",sparse=False))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num",numeric_pipe,num_cols),
        ("cat",categorical_pipe,cat_cols)
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

#学習データを前処理
X_processed = preprocess.fit_transform(X)

#学習データ to Tensor
X_tensor = torch.tensor(X_processed,dtype=torch.float32)
y_tensor = torch.tensor(y.values,dtype=torch.float32).view(-1,1)

#Dataloaderを使うためDatasetを作成
dataset = TensorDataset(X_tensor,y_tensor)
train_set,val_set = train_test_split(dataset,test_size=0.2,random_state=42)

train_loader = DataLoader(train_set,batch_size=64,shuffle=True)
val_loader = DataLoader(val_set,batch_size=64,shuffle=False)

#テスト用データも同様の処理を行う
X_test_processed = preprocess.transform(test)
X_test_tensor = torch.tensor(X_test_processed,dtype=torch.float32)

#ニューラルネットワークモデルの定義
class HousePrices (nn.Module):
    def __init__(self,features:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features,256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32,1)
        )
    
    def forward(self,x):
        return self.net(x)

inputDim = X_tensor.shape[1]
model = HousePrices(inputDim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,mode="min",factor=0.5
,patience=3,verbose=True)

#学習ループ
EPOCHS = 60
for epoch in range(1, EPOCHS + 1):
    # ---- Training ----
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss  = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)

    train_loss = running_loss / len(train_loader.dataset)

    # ---- Validation ----
    model.eval()
    val_running = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss  = criterion(preds, yb)
            val_running += loss.item() * xb.size(0)

    val_loss = val_running / len(val_loader.dataset)
    scheduler.step(val_loss)

    if epoch % 5 == 0 or epoch == 1:
        print(f"[{epoch:02d}/{EPOCHS}] "
              f"Train MSE: {train_loss:.5f}  |  Val MSE: {val_loss:.5f}")
        

#推論と提出ファイル作成
model.eval()
with torch.no_grad():
    test_preds_log = model(X_test_tensor.to(device)).cpu().squeeze()

#log1p の逆変換
test_preds = torch.expm1(test_preds_log).numpy()

# submission.csv 出力
sub_df = pd.DataFrame({
    "Id": pd.read_csv("Data/test.csv")["Id"],   # 元の Id 列を復活
    "SalePrice": test_preds
})
sub_df.to_csv("submission_nn.csv", index=False)
