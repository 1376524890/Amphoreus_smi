"""
余弦相似度收敛检测
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict

def texts_to_vec(texts:List[str])->np.ndarray:
    vect = TfidfVectorizer(stop_words=None, token_pattern=r"(?u)\b\w+\b")
    return vect.fit_transform(texts).toarray()

def cosine_sim(a:np.ndarray, b:np.ndarray)->float:
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

def is_converged(logs:List[Dict], window:int=3, threshold:float=0.95)->bool:
    """检测最近 window 轮行为文本是否相似"""
    if len(logs) < window:
        return False
    recent = [l["action"] for l in logs[-window:]]
    vecs   = texts_to_vec(recent)
    # 两两相似度
    for i in range(len(vecs)-1):
        if cosine_sim(vecs[i], vecs[i+1]) < threshold:
            return False
    return True