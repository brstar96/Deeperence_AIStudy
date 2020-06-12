# KpopCelebRetrieval
Deeperence 멱살스터디 세 번째 시간입니다! 이번 시간에는 지난 스터디에 이은 심화 버전으로 연예인 이미지 검색기를 준비했습니다. 이미지 검색은 softmax 레이어를 거쳐 확률을 뽑아 내는 분류 문제와는 달리 FC 레이어를 활용해 두 이미지 간 유사도를 측정합니다. FC 레이어는 Image retrieval, GAN을 비롯한 다양한 방면에 활용되므로 이번 스터디를 통해 기존까지 알고 계시던 지식을 확장하셨으면 좋겠습니다. 

<blockquote>
<b>Deeperence 멱살 스터디는...</b><br>
숭실대학교 머신러닝 소모임 Deeperence에서 진행하는 'Vision AI 멱살 스터디'는 처음 비전 인공지능에 입문하신 분들을 대상으로 한달간 다양한 태스크를 속성으로 경험시켜 드리는 스터디입니다. 이름 그대로 멱살을 잡아끄는 듯한 초밀착 멘토링으로 가려운 곳을 시원하게 긁어 드립니다. (이 튜토리얼은 Deeperence, 제 <a href = "https://brstar96.github.io/">개인 블로그</a>에 연재됩니다.)<br><br>
- 지난 스터디 복습하기<br>
  - 1. MNIST: https://colab.research.google.com/drive/1ygxE9jzh3PtU05O9zCaQ8Sc-WvuXSOpn <br>
  - 2. KaKR 3rd 자동차 분류대회: https://colab.research.google.com/drive/1p5GEx8UzGcBu-Nxjd8XxYQTBYh0Xlk8h  
 <br><br>
<i>Written by Myeong-Gyu.LEE, 2020-01-02</i>
</blockquote>
</blockquote>
<br>

이 코드는 다음과 같은 순서로 연예인 유사 이미지 검색을 수행합니다. <br>
1. Train set으로 softmax가 없는 분류기를 학습합니다.
2. Query image와 모든 reference image들 간에 Cosine similarity를 구한 후 유사도가 높은 순으로 sort합니다. 
```python
sim_matrix = np.dot(query_vecs, reference_vecs.T)
indices = np.argsort(sim_matrix, axis=1)
indices = np.flip(indices, axis=1)
```

## Dependencies
- Python 3.6.x
- pytorch 1.3
- scikit-learn 0.21.2
- pandas 0.24.2
- numpy 1.16.4
- pillow 6.2.1

## Usage
학습 수행 시는 `args.mode`를 `train`으로, 테스트 수행 시엔 `args.mode`를 `test`로 설정하고 실행해 주세요.<br><br>
run `main.py` with following arguments:
> (Linux) $python3 main.py <br>
> (Windows) $python main.py
