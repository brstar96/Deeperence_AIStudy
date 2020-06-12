# KaKR3rd Car Model Classificaiton
Deeperence 멱살스터디 두 번째 시간입니다! <br>

이번 시간엔 캐글 코리아라는 비영리 페이스북 그룹의 세 번째 대회인 2019 3rd ML month with KaKR 베이스라인을 함께 보도록 하겠습니다. MNIST를 마치고 오신 분들이 3채널 RGB 이미지를 처음 마주하게 되면 멘탈이 흔들리실 수 있는데요, 흔히 RGB 이미지를 2차원 배열로 알고 계신 분들이 많기 때문에 이번 튜토리얼에서 지금껏 우리가 보아 왔던 이미지의 실체(?)를 파악하고 어떻게 RGB 이미지를 핸들링해 분류 문제를 해결할 수 있을지 고민해 보겠습니다.<br>

더 나아가 Kaggle을 비롯한 다양한 대회에서 사용되는 실전 테크닉(EDA, Cross Validation, Image Augmentation)과 Optimizer, Learning rate scheduler를 통한 loss surface 안정화 방법에 대해 알아봅니다.

<blockquote>
<b>Deeperence 멱살 스터디는...</b><br>
숭실대학교 머신러닝 소모임 Deeperence에서 진행하는 'Vision AI 멱살 스터디'는 처음 비전 인공지능에 입문하신 분들을 대상으로 한달간 다양한 태스크를 속성으로 경험시켜 드리는 스터디입니다. 이름 그대로 멱살을 잡아끄는 듯한 초밀착 멘토링으로 가려운 곳을 시원하게 긁어 드립니다. (이 튜토리얼은 Deeperence, 제 <a href = "https://brstar96.github.io/">개인 블로그</a>에 연재됩니다.)<br><br>
- 지난 스터디 복습하기<br>
  - 1. MNIST: https://colab.research.google.com/drive/1ygxE9jzh3PtU05O9zCaQ8Sc-WvuXSOpn <br>  
 <br>
<i>Written by Myeong-Gyu.LEE, 2019-12-20</i>
</blockquote>
</blockquote>
<br>

<center>
<table align="center">
<tbody><tr><td><center>
  <a target="_blank" href="https://colab.research.google.com/drive/1p5GEx8UzGcBu-Nxjd8XxYQTBYh0Xlk8h?usp=sharing">
    <img src="https://camo.githubusercontent.com/dfbf50eed8dd2dea5f3e0beaaf2001eeca77f314/68747470733a2f2f7777772e74656e736f72666c6f772e6f72672f696d616765732f636f6c61625f6c6f676f5f333270782e706e67" data-canonical-src="https://www.tensorflow.org/images/colab_logo_32px.png"><br>Google Colab에서 열기
  </a></center>
</td>
    
<td>
  <a target="_blank" href="https://github.com/brstar96/Deeperence_AIStudy/blob/master/02_KaKR3rd_CarModelClassificaiton/02_KaKR3rd_CarModelClassificaiton.ipynb">
    <img width="32px" src="https://camo.githubusercontent.com/9a6bfd119aeed95f13553a994f2d1cd97e033768/68747470733a2f2f7777772e74656e736f72666c6f772e6f72672f696d616765732f4769744875622d4d61726b2d333270782e706e67" data-canonical-src="https://www.tensorflow.org/images/GitHub-Mark-32px.png"><br>GitHub에서 노트북 열기</a>
</td>
</tr></tbody></table>
</center>

<br>

* 멱살스터디 커리큘럼 (커리큘럼의 주제는 스터디 진행 상황에 따라 바뀔 수 있습니다.)
  1. 깊게 배우는 Pytorch CNN MNIST Tutorial: 단순히 모델과 데이터를 불러와 돌리는 학습이 아닌, 베이즈 이론 관점에서의 MNIST 튜토리얼을 다뤄 봅니다. 
  2. KaKR_ML Month Car Model Classification: 3채널 RGB 이미지를 활용해 차종 분류 문제를 해결해 봅니다. Kaggle을 비롯한 다양한 대회에서 사용되는 실전 테크닉(EDA, Cross Validation, Image Augmentation)과 Optimizer, Learning rate scheduler를 통한 loss surface 안정화 방법에 대해 알아봅니다.
  3. Kpop Celeb Retrieval: 두 이미지 간의 유사도를 비교하는 Image Retrieval 문제를 해결해 봅니다. 나와 닮은 꼴 연예인은 누구!?
  4. (진행예정) Recycling garbage Detection: 이미지 속에서 페트병을 찾아라! 이미지 속에서 특정 물체의 위치와 클래스를 검출하는 Detection 문제를 해결해 봅니다. 
  5. (진행예정) Aerial Semantic Segmentation: 특정 사물의 영역 마스크를 검출하는 semantic segmentation 문제를 해결해 봅니다. 
