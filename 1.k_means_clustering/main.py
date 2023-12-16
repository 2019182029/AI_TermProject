import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import matplotlib.font_manager as fm

if __name__ == "__main__":
    font_path = 'malgunsl.ttf'
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)

    # NLPRK_STA.csv 파일을 읽어서 처음 데이터 확인
    csv_data_df = pd.read_csv('NLPRK_STA.csv', encoding='cp949')

    # Pandas의 Dataframe에서 국립공원 육지면적, 탐방객수 컬럼만 선택해서 X로 정의한다.
    X = csv_data_df.iloc[:, [1, 2]].values
    m = X.shape[0]
    n = X.shape[1]

    K = 3  # cluster 개수 k = 3
    n_iter = 100  # iteration = 100

    Centroids = np.array([]).reshape(n, 0)
    for i in range(K):
        rand = rd.randint(0, m - 1)
        Centroids = np.c_[Centroids, X[rand]]

    Output = {}
    EucDistance = np.array([]).reshape(m, 0)
    tempDist = np.sum((X - Centroids[:, 0]) ** 2, axis=1)

    for i in range(n_iter):
        EucDistance = np.array([]).reshape(m, 0)
        for k in range(K):
            tempDist = np.sum((X - Centroids[:, k]) ** 2, axis=1)
            EucDistance = np.c_[EucDistance, tempDist]
        C = np.argmin(EucDistance, axis=1) + 1

        Y = {}
        for k in range(K):
            Y[k + 1] = np.array([]).reshape(2, 0)
        for i in range(m):
            Y[C[i]] = np.c_[Y[C[i]], X[i]]
        for k in range(K):
            Y[k + 1] = Y[k + 1].T
        for k in range(K):
            Centroids[:, k] = np.mean(Y[k + 1], axis=0)
        Output = Y

    color = ['green', 'blue', 'red']
    labels = ['cluster1', 'cluster2', 'cluster3']

    for k in range(K):
        plt.scatter(Output[k + 1][:, 0], Output[k + 1][:, 1], c=color[k], label=labels[k])
    plt.scatter(Centroids[0, :], Centroids[1, :], s=300, c='yellow', label='Centroids')
    plt.xlabel('육지면적 (㎢)')  # x축은 '육지면적'
    plt.ylabel('탐방객수 (명)')  # y축은 '탐방객수'
    plt.grid(True)
    plt.legend()
    plt.show()
