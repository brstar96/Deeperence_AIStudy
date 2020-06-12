# MNIST Classification
Deeperence 멱살스터디 첫 번째 시간입니다! 갓 선형회귀와 로지스틱 회귀를 마치고 오신 분들이 제일 먼저 접하는 튜토리얼이 바로 이 Deep learning의 'Hello, world!'와도 같은 MNIST CNN 튜토리얼이죠. 하지만 대부분의 튜토리얼이 코드 설명에만 치중해 있다 보니 깊은 의미에 대해 잘 알려주지 않는 경우가 많습니다. 이번 스터디를 통해 베이즈 이론 관점에서의 신경망을 배워 봅니다.<br>  

선형 회귀를 갓 마치고 오신 분들께서 가장 헷갈려 하시는 부분이 고차원 이미지 데이터 핸들링인데요, 이번 시간에 ①MNIST 데이터를 다뤄 보며 1차원 그레이 스케일 이미지에 대해 이해하고, 다음 시간에 ②KaKR 3rd 차종분류 대회의 데이터셋을 활용해 3차원 RGB 이미지에 대해 이해하는 시간을 가져 보겠습니다.<br>

MNIST 데이터셋은 국내는 물론 해외 머신러닝 커뮤니티에서 오랜 시간 사랑받고 연구에 활용되어 왔습니다. MNIST는 1980년대 미국 국립표준기술연구소에서 수집한 6만 개 훈련 이미지, 그리고 1만 개의 테스트 이미지로 구성되어 있습니다. MNIST 데이터셋은 또한 Numpy array 형태로 다양한 머신러닝 프레임워크들에서 기본 제공하고 있습니다.<br>

MNIST 데이터셋을 활용한 손글씨 분류 문제는 0부터 9까지의 클래스를 가진 다량의 손글씨 이미지 데이터들을 학습한 후 들어오는 테스트 이미지에 대해 0~9 사이의 분류 결과를 예측하는 것입니다. 이번 노트북에서는 이미지 데이터의 구조와 함께 CNN 훈련을 위해 어떻게 이 데이터를 핸들링할지에 초점을 맞추어 스터디를 진행해 보도록 하겠습니다.


<blockquote>
<b>Deeperence 멱살 스터디는...</b><br>
숭실대학교 머신러닝 소모임 Deeperence에서 진행하는 'Vision AI 멱살 스터디'는 처음 비전 인공지능에 입문하신 분들을 대상으로 한달간 다양한 태스크를 속성으로 경험시켜 드리는 스터디입니다. 이름 그대로 멱살을 잡아끄는 듯한 초밀착 멘토링으로 가려운 곳을 시원하게 긁어 드립니다. (이 튜토리얼은 Deeperence, 제 <a href = "https://brstar96.github.io/">개인 블로그</a>에 연재됩니다.)<br><br>

<i>Written by Myeong-Gyu.LEE, 2020-11-20</i>
</blockquote>
</blockquote>
<br>



<table align="center">
<tbody><tr><td>
  <center>
  <a target="_blank" href="https://colab.research.google.com/drive/1ygxE9jzh3PtU05O9zCaQ8Sc-WvuXSOpn?usp=sharing">
    <img src="https://camo.githubusercontent.com/dfbf50eed8dd2dea5f3e0beaaf2001eeca77f314/68747470733a2f2f7777772e74656e736f72666c6f772e6f72672f696d616765732f636f6c61625f6c6f676f5f333270782e706e67" data-canonical-src="https://www.tensorflow.org/images/colab_logo_32px.png"><br>Google Colab에서 열기
  </a>
  </center>
</td>
    
<td>
  <a target="_blank" href="https://github.com/brstar96/Deeperence_AIStudy/blob/master/01_MNIST_Classification/01_MNIST_Classification.ipynb">
    <img width="32px" src="https://camo.githubusercontent.com/9a6bfd119aeed95f13553a994f2d1cd97e033768/68747470733a2f2f7777772e74656e736f72666c6f772e6f72672f696d616765732f4769744875622d4d61726b2d333270782e706e67" data-canonical-src="https://www.tensorflow.org/images/GitHub-Mark-32px.png"><br>GitHub에서 노트북 열기</a>
</td>
</tr></tbody></table>

<br>

* 멱살스터디 커리큘럼 (커리큘럼의 주제는 스터디 진행 상황에 따라 바뀔 수 있습니다.)
  1. 깊게 배우는 Pytorch CNN MNIST Tutorial: 단순히 모델과 데이터를 불러와 돌리는 학습이 아닌, 베이즈 이론 관점에서의 MNIST 튜토리얼을 다뤄 봅니다. 
  2. KaKR_ML Month Car Model Classification: 3채널 RGB 이미지를 활용해 차종 분류 문제를 해결해 봅니다. Kaggle을 비롯한 다양한 대회에서 사용되는 실전 테크닉(EDA, Cross Validation, Image Augmentation)과 Optimizer, Learning rate scheduler를 통한 loss surface 안정화 방법에 대해 알아봅니다.
  3. Kpop Celeb Retrieval: 두 이미지 간의 유사도를 비교하는 Image Retrieval 문제를 해결해 봅니다. 나와 닮은 꼴 연예인은 누구!?
  4. (진행예정) Recycling garbage Detection: 이미지 속에서 페트병을 찾아라! 이미지 속에서 특정 물체의 위치와 클래스를 검출하는 Detection 문제를 해결해 봅니다. 
  5. (진행예정) Aerial Semantic Segmentation: 특정 사물의 영역 마스크를 검출하는 semantic segmentation 문제를 해결해 봅니다. 
