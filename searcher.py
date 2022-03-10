import faiss
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm

class VectorEngine:

    def _cos_sim(self, A, B):
        
        return dot(A, B)/(norm(A)*norm(B))

    def _vector_search(self, query_vector, index, num_results=3):
        D, I = index.search(np.array([query_vector]), k=num_results)

        return I

    def indexer(self, encoded_vector):

        db_vector = encoded_vector

        # 벡터차원 설정
        index = faiss.IndexFlatL2(db_vector.shape[1])

        index = faiss.IndexIDMap(index)

        # 벡터와 id 매핑
        index.add_with_ids(db_vector, np.arange(len(db_vector)))

        return index, db_vector

    def searcher(self, df, db_vector, index):

        result = []
        target_acc = 0.9
        target_columns = ['id1', 'id2', 'text1', 'text2',  'similarity']
        
        def _cos_sim(A, B):
            return dot(A, B)/(norm(A)*norm(B))

        for query_id in df.index:

            if query_id % 10000 == 0:
                print(f'{int((query_id / len(df))*100)}% 검색 완료')
            
            D, target_id_list = index.search(np.array([db_vector[query_id]]), k=3)

            for target_id in target_id_list[0][1:]:
                sub_result = []
                similarity = _cos_sim(db_vector[query_id], db_vector[target_id])

                if similarity > target_acc:
                    sub_result.append([
                                    df.loc[query_id].id, 
                                    df.loc[target_id].id, 
                                    df.loc[query_id].text, 
                                    df.loc[target_id].text, 
                                    similarity
                                    ])
                    result.append(sub_result)
        
        result_df = pd.DataFrame(np.array(result).reshape(-1, 5), columns= target_columns)
        # result_df.sort_values(by = target_columns[-1], ascending=False)

        return result_df