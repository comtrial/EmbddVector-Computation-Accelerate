

import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from Embedding_Autoencoder.data_set import EmbeddDataset
from Embedding_Autoencoder.model import Autoencoder
from searcher import VectorEngine

def get_encodings(model, dl):
    model.eval()
    with torch.no_grad():
        encodings = [model.encoder(x) for x, _ in dl]
    return torch.cat(encodings, dim=0)

def main():
    
    start = time.time()
    # DataLoader
    print('Data 로딩 중..') 
    EMBD_PICK_DIR = 'data/embedding.pkl'
    dataset = EmbeddDataset(EMBD_PICK_DIR)
    data_loader = DataLoader(dataset, batch_size=32, shuffle = False)

    # Autoencoder 
    print('차원축소 모델 설정 중..')
    target_dim = 20
    device = torch.device('cpu')
    model_path = "saved_model/AE20D.ckpt"
    model = Autoencoder(target_dim)
    model.load_state_dict(torch.load(model_path, map_location=device)["state_dict"])

    # 학습된 Autoencoder를 통한 차원축소된 20차원 vector
    print('차원축소 중..')
    encoded_vector = get_encodings(model, data_loader).numpy()
    
    del dataset, data_loader
    
    # faiss 벡터 검색엔진
    print('검색 중..')
    engine = VectorEngine()
    df = pd.read_excel('data/aihub_or_kr-sports_ko.xlsx')
    index, db_vector = engine.indexer(encoded_vector)
    result = engine.searcher(df, db_vector, index)
    
    
    result = result.drop_duplicates(['similarity'], keep='first')
    result.sort_values(by = 'similarity', ascending=False).to_csv('result.csv')
    
    print(f'프로그램 실행 시간 {time.time() - start}, 찾은 결과{len(result.shape)}')
    return

if __name__ == '__main__':
    main()