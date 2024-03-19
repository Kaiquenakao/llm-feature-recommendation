import pandas as pd
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np


def recomendacao(search_text):
    data = [['Modelo de titanic é um modelo que prediz se um passageiro sobreviveu ou não', ['idade', 'passageiro_id', 'pclass', 'valor_casa']],
    ['Modelo de predicao de casa é um modelo que prediz o valor de uma casa', ['valor_casa', 'cor_casa', 'quantidade_quarto']],
    ]
    df = pd.DataFrame(data, columns = ['descricao', 'features_utilizada'])


    text = df['descricao']
    encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
    vectors = encoder.encode(text)


    vector_dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(vector_dimension)
    faiss.normalize_L2(vectors)
    index.add(vectors)

    search_vector = encoder.encode(search_text)
    _vector = np.array([search_vector])
    faiss.normalize_L2(_vector)

    k = index.ntotal
    distances, ann = index.search(_vector, k=k)

    results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})

    merge = pd.merge(results, df, left_on="ann", right_index=True)

    labels  = df['features_utilizada']
    category = labels[ann[0][0]]
    distance_09 = merge[merge["distances"] < 0.9]

    features_list = list(set(sum(distance_09['features_utilizada'], [])))
    return features_list

def main():
    question = st.text_input("Digite sua pergunta")
    if st.button('Buscar'):
        with st.spinner('Buscando....'):
            rec = recomendacao(question)
            st.text_area("Features recomendadas: ", value=rec)

if __name__ == "__main__":
    main()