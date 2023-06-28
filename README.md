# Purpose
## Computer-Vision-Courses-in-Kaggle 강의 번역
(Create image classifiers with TensorFlow and Keras, and explore convolutional neural networks.)

캐글(Kaggle)에서 제공하는 Computer Vision 내용을 번역하였습니다.
(오역 및 잘못된 부분이 많으니 참고 바랍니다.)

---

# 목차

1. The Convolutional Classifier (In Korean)
   - https://www.kaggle.com/piantic/the-convolutional-classifier-in-korean
2. Convolution and ReLU (In Korean)
   - https://www.kaggle.com/piantic/convolution-and-relu-in-korean
3. Maximum Pooling (In Korean) (will be updated.)
4. The Sliding Window (In Korean) (will be updated.)
5. Custom Convnets (In Korean) (will be updated.)
6. Data Augmentation (In Korean) (will be updated.)


* 다른 곳에 활용하실 때에는 출처를 남겨주시면 감사드리겠습니다.

  (사용에 특별한 제한은 없습니다.)

# 기타(임시)

- 데이터 전처리
    - 데이터 수집 (Collection)
    - 데이터 정제 (Clean Labels/Samples)
    - 데이터 증강 (Data Up/Down)
- AI 모델 개발(설계-평가-튜닝)
    - AI 모델 아키텍쳐 설계
    - XAI (Explainable AI)
    - AI 모델 학습 및 평가 (Learning curve 해석/검증)
    - AI 모델 튜닝 (HPO, Class Imbalanced Class)
- AI 시스템 구축(설계-배포-모니터링-자동화-최적화)
    - AI 시스템(ML PipeLine) 설계 및 배포
    - AI 시스템 모니터링 및 자동화 (지속운영)
    - AI 시스템 최적화
- 최신 트렌드
    - 최근 주요 AI 기술 발전 트렌드

- 학습자료
    - Full Stack Deep Learning
        - Spring2021
        - course-2022
    - Software with AI 방법론
        - 01 개발 프로세스
        - 03 DevMLOps
        - 04 인프라
     
- 1번
    - minority, accuracy, interpolation, overfitting, outlier
    - decision boundary

- 2번
- 3번
- 4번
    - Direct multi-step
    - Single-shot
    - Recursive model

- 5번
- 6번
    - 멀티 label - binary cross-entropy
    - element-wise 방식으로 계산한 accuracy는 주어진 문제 상황에 적합하지 않다.
        - row-wise, f-beta score

- 8번
    - 추론 시 batch norm layer는 스킵 X. 저장해 놓은 평균과 분산값 사용
    - Softmax 함수는 exponential 함수가 포함되어 있어서 실수배를 해도 결과값 동일하지 않을 수 있음
    - Split을 하기 전에 정규화를 수행하면 test의 정보가 학습 단계에 유입될 수 있다.
    - Early stopping은 regularization의 일종으로 variance를 낮추는 경향이 있다.

- 9번
    - L1 regularization은 L2와 다르게 일부 weight를 0으로 만들 수 있다.
    - L1과 L2는 loss가 최소값에 가까워지는 것을 어렵게 하여 regularizer의 역할 수행
    - L1의 경우 동일한 폭으로 감

- 11번
    - Dropout - bias 증가, Variance 감소
    - 더 많은 학습 데이터 추가 - biase 변화 없음, Variance 감소
    - Weight decay는 dropout과 유사한 측면을 가짐
        - bias 증가, variance 감
        
- 15
    - 트랜스포머의 Positional encoding은 절대적인 정보를 입력
    - multi-head 입력 차원을 head 개수로 나눠진 차원에 projection 하여 나눠서 입력하여 각 head는 입력 sequence를 모두 볼 수 있다
    - position wise feed forward nn은 입력 sequence 각각 따로 feed forward nn을 통과한다.
 
- - 1번
    - standardization은 feature의 분포가 gaussian distribution을 따른다는 가정이 성립할 때 사용하는 방법
- 2번
    - 데이터 증강 - 데이터를 증가시켜 모델의 분산(variance)를 감소시킨다
    - training loss가 처음은 높게 나올 수 있다
    - 이미지 일부 채워넣는 것도 증강 방식
    - 자언어 - 문장의 단어를 비슷한 의미의 단어로 대체하여 증강, 증강 과정에서 미리 학습된 word embedding vector의 코사인 거리(cosine distance)를 이용할 수 있음

- 4번
    - inductive bias가 약한 모델은 강항 모델에 비해 상대적으로 필요한 학습 데이터셋의 양이 많다
    - 입력 데이터를 순차적으로 처리하는 RNN의 특성은 inductive bias라고 볼 수 있고, 매 step 동일한 함수를 재귀적으로(recursively) 적용하는 것도 ib라고 볼 수 있다
    - Vision Transformer 모델은 입력부에 가까운 lower layers와 출력부에 가까운 higher layers 사이의 representation의 유사도(similarity)가 CNN 모델과 비교하여 더 높다.
    - Vision Transformer 모델은 CNN 모델과 비교하여 입력 이미지 상의 특정한 위치에 대한 위치 정보(positional information)을 마지막 layer까지 더 잘 유지한다.
    - 트포에서 self-attention을 통해 계산된 입력 데이터의 임의의 두 토큰 사이의 attention 값은 Query와 Key에 따라 달라진다.

- 6번 - 여기서는 (가)를 feature extraction으로 (나)를 fine-tuning으로 정의함
    - fine-tuning은 새로운 task에 weight 값을 좀 더 최적화 하기 위해 수행하나 모델 복잡도의 문제를 해결하지는 않는다.
    - feature extraction은 base model과 새로운 task가 유사함을 전제로 적용되는 방법, base model에서 추출한 feature들이 적용할 문제의 classifier에서 중요한 구분자가 될 때 효과가 높다
    - feature extraction은 base model을 동결하기 때문에 해당 layer에 대해 gradient를 계산할 필요가 없으므로 학습은 빠를 수 있으나 추론 속도는 변하지 않는다.
    - fine-tuning은 데이터가 적을 때 overfitting 가능성이 높아 feature extraction만 사용하는 것보다 일반화 성능이 하락할 수 있다.
    - Transfer learning은 추론 속도를 개선하지 않는다.

- 8번 - 언더피팅, 오버피팅 그래프 보고 설명 맞추기
    - underfitting - 네트워크의 capacity를 늘리거나 regularization 파라미터의 값을 줄임, epoch을 늘려 학습을 더 시켜봄
    - overfitting - regularization 방법들(dropout, batch normalization 등)을 적용하거나 network의 capacity를 줄이는 방법 등이 있음

- 9번
    - 모델의 예측값과 실제 정답의 차이의 평균 - bias
    - 입력 데이터에 대해 예측값이 얼마만큼 변화할 수 있는지에 대한 양(amount)의 개념 - variance
    - 딥러닝 알고리즘은 training set에서 random error에 대해 견고하므로(robust) 꼭 수정할 필요는 없지만 (systematic error)에 대해서는 민감
    - validation set에 존재하는 잘못 레이블된 데이터를 수정하는 경우에는 validation set과 test set이 동일한 분포를 가질 수 있도록 (test set)에 동일하게 error correcting 작업을 수행해줘야 한다.
        - 솔직히 이건… test set은 숨김처리하고 하는게 맞지 않나…

- 10번 - Bagging과 Boosting 기법 설명
    - Bagging 기법은 각 독립 모델의 결과를 집계하여 평가하기 때문에 variance error를 줄이느 효과가 있다
    - Boosting 기법은 오분류된 샘플에 대해서 가중치를 더해 하나의 모델이 학습되므로 모델의 정확도를 높여 bias error를 줄이는 효과가 있다.
    - Bagging
        - 각 개별 독립 모델마다 raw data에서 복원 추출로 뽑힌 데이터로 학습 진행, 학습 데이터에 뽑히지 않은 데이터를 활용하여 평가 및 검증을 진행한다.
    - Bagging 기법은 개별 독립 모델이 학습되므로 분산 computing에 좀 더 용이할 수 있지만, bossting 기법은 sequential하게 진행되므로 분산 computing이 어렵다.
    - Bagging 기법에서 개별 독립 모델 결과를 집계할 때, categorical 데이터에서는 투표 방식을 집계하고 continuous 데이터에서는 평균으로 집계한다.
    - Boosting은 오분류된 샘플에 더 많은 가중치를 부여하여 모델의 성능을 높인다.
    
- 12번 - Dropout과 이후 인퍼런스 때 가중치에 보정하는 문제
    - 오차 계산 - 기울기를 backpropagation으로 찾기 - 파라미터 업데이트
    - 0.25 드롭아웃 주면 보정할 때는 inference score에 (1-0.25)의 값을 곱해야 한다. 그래서 3/4
    
- 13번 - 모델 드리프트(Model Drift) 관련
    - 데이터 x의 분포의 변화(P(x)의 변화)는 데이터 드리프트를 의미하며 모델은 모든 x 에 대해 정확히 y를 예측할 수 없고 데이터 편향은 모델 성능에 영향을 미치므로 이러한 변화는 모델의 예측력에 변화를 준다.

- 14번 - GPT3 관련
    - 이건 지금도 유효한지 알 수 없음. GPT-4는 다를 지도

---

인공지능 (Artificial Intelligence): 인간의 지능을 모방하고 구현하는 기술과 분야입니다.

딥러닝 (Deep Learning): 인공신경망을 기반으로 한 머신러닝 알고리즘의 한 종류로, 다층 신경망을 사용하여 데이터를 학습하고 예측하는 방법입니다.

신경망 (Neural Network): 생물학적 뉴런의 동작 원리를 모방한 기계학습 모델로, 입력층, 은닉층, 출력층 등으로 구성되어 있습니다.

입력층 (Input Layer): 모델에 입력되는 데이터의 초기층으로, 외부로부터 입력 신호를 받습니다.

은닉층 (Hidden Layer): 입력층과 출력층 사이에 위치한 중간층으로, 입력 데이터를 처리하고 중간 표현을 만듭니다.

출력층 (Output Layer): 최종적으로 모델에서 예측되는 결과를 출력하는 층입니다.

가중치 (Weights): 각 뉴런 또는 노드 사이의 연결 강도를 나타내는 값으로, 입력 신호에 곱해져 중요도를 조절합니다.

편향 (Bias): 각 뉴런 또는 노드에 추가되는 상수로, 활성화 함수의 선형 변환에 영향을 줍니다.

활성화 함수 (Activation Function): 각 뉴런의 출력을 결정하는 비선형 함수로, 네트워크의 비선형성과 표현력을 증가시킵니다.

역전파 (Backpropagation): 신경망에서 가중치와 편향을 학습하기 위해 사용되는 알고리즘으로, 출력 값과 정답 사이의 오차를 역으로 전파하여 가중치를 조정합니다.

경사하강법 (Gradient Descent): 손실 함수의 기울기를 이용하여 최적의 가중치를 찾기 위한 최적화 알고리즘입니다.

손실 함수 (Loss Function): 모델의 출력과 정답 사이의 오차를 계산하는 함수로, 학습 중에 최소화되어야 하는 목표입니다.

최적화 (Optimization): 손실 함수를 최소화하기 위해 모델의 가중치와 편향을 조정하는 과정입니다.

과적합 (Overfitting): 모델이 학습 데이터에 너무 맞추어져서 새로운 데이터에 대한 일반화 성능이 떨어지는 현상입니다.

드롭아웃 (Dropout): 모델의 일부 뉴런을 임의로 제거하여 과적합을 방지하는 정규화 기법입니다.

배치 정규화 (Batch Normalization): 학습 과정 중에 입력 데이터를 정규화하여 학습을 안정화시키고 속도를 향상시키는 방법입니다.

합성곱 신경망 (Convolutional Neural Network, CNN): 이미지와 같은 그리드 형태의 데이터에 적합한 신경망 구조로, 합성곱과 풀링 계층을 포함합니다.

풀링 (Pooling): 합성곱 계층의 출력을 간소화하고 위치 불변성을 제공하기 위해 사용되는 다운샘플링 기법입니다.

순환 신경망 (Recurrent Neural Network, RNN): 시퀀스 형태의 데이터를 처리하는데 특화된 신경망으로, 내부에 순환 구조를 가지고 있습니다.

장단기 메모리 (Long Short-Term Memory, LSTM): RNN의 변형 구조로, 시퀀스 데이터의 장기 의존성을 학습하기 위해 고안된 메모리 셀입니다.

게이트 (Gate): LSTM에서 사용되는 게이트로, 정보의 흐름을 제어하여 시퀀스 데이터의 학습과 예측을 개선합니다.

변이형 오토인코더 (Variational Autoencoder, VAE): 생성 모델 중 하나로, 데이터의 잠재 변수를 학습하여 새로운 샘플을 생성하는 모델입니다.

생성적 적대 신경망 (Generative Adversarial Network, GAN): 생성자와 판별자라는 두 개의 모델이 서로 경쟁하며 데이터를 생성하고 평가하는 모델입니다.

전이 학습 (Transfer Learning): 한 작업에서 학습한 모델을 다른 작업에 적용하여 학습 속도와 성능을 향상시키는 방법입니다.

사전 훈련된 모델 (Pretrained Model): 대규모 데이터셋에서 미리 학습된 모델로, 일반적인 특징을 잘 추출할 수 있습니다.

파인튜닝 (Fine-tuning): 사전 훈련된 모델을 다른 작업에 맞게 조정하고 추가로 학습시키는 과정입니다.

데이터 전처리 (Data Preprocessing): 원시 데이터를 모델에 적합한 형태로 가공하고 정제하는 과정입니다.

데이터 증강 (Data Augmentation): 원본 데이터를 변형하여 데이터셋을 인위적으로 확장시키는 방법으로, 모델의 일반화 성능을 향상시킵니다.

하이퍼파라미터 (Hyperparameter): 모델 아키텍처나 학습 과정을 제어하는 매개변수로, 사람이 직접 설정해야 합니다.

학습률 (Learning Rate): 경사 하강법에서 한 번에 이동하는 거리를 결정하는 매개변수로, 모델의 학습 속도를 조절합니다.

배치 크기 (Batch Size): 한 번의 업데이트마다 사용되는 학습 데이터의 개수입니다.

에포크 (Epoch): 전체 데이터셋을 한 번 학습하는 것을 의미하며, 에포크 수는 학습 과정에서 데이터셋을 반복하는 횟수를 의미합니다.

초기화 (Initialization): 모델의 가중치와 편향을 초기 값으로 설정하는 과정입니다.

정규화 (Regularization): 모델의 복잡도를 제한하고 과적합을 방지하기 위해 가중치에 추가되는 페널티를 설정합니다.

엔트로피 (Entropy): 정보 이론에서 불확실성의 척도로 사용되며, 모델의 예측 확률 분포의 불확실성을 나타냅니다.

미니배치 (Mini-batch): 한 번의 학습 과정에서 사용되는 일부 데이터로 구성된 작은 데이터 그룹입니다.

원-핫 인코딩 (One-hot Encoding): 범주형 변수를 이진 형태로 표현하는 방법으로, 각 범주에 대해 하나의 요소만 1로 표현하고 나머지는 0으로 표현합니다.

자연어 처리 (Natural Language Processing, NLP): 인간의 언어를 컴퓨터가 이해하고 처리할 수 있는 형태로 변환하는 분야입니다.

텍스트 분류 (Text Classification): 텍스트를 사전 정의된 범주로 분류하는 과정입니다.

이미지 분류 (Image Classification): 이미지를 사전 정의된 범주로 분류하는 과정입니다.

객체 감지 (Object Detection): 이미지에서 특정 객체의 위치와 종류를 찾는 작업입니다.

시계열 예측 (Time Series Prediction): 시간에 따라 변화하는 데이터의 패턴을 학습하여 미래 값을 예측하는 작업입니다.

강화 학습 (Reinforcement Learning): 환경과 상호작용하며 보상을 최대화하는 의사 결정 전략을 학습하는 방법입니다.

모델 저장 및 로드 (Model Saving and Loading): 학습한 모델을 저장하고 나중에 재사용하기 위한 과정입니다.

파라미터 (Parameter): 모델이 학습하여 조정하는 가중치와 편향 등의 값입니다.

튜닝 (Tuning): 모델의 하이퍼파라미터를 조정하여 최적의 성능을 달성하는 과정입니다.

모델 복잡도 (Model Complexity): 모델이 학습할 수 있는 함수의 복잡성을 나타내는 개념입니다.

모델 성능 평가 (Model Evaluation): 학습된 모델의 예측 성능을 평가하고 분석하는 과정입니다.

정확도 (Accuracy): 모델의 예측 결과가 실제 값과 얼마나 일치하는지를 나타내는 지표입니다.

오차 행렬 (Confusion Matrix): 분류 모델의 성능을 평가하기 위해 실제 클래스와 예측 클래스의 일치 여부를 행렬로 나타낸 것입니다.

이러한 키워드들은 AI 모델 아키텍처 설계와 관련된 핵심적인 개념들을 포함하고 있습니다. 이러한 개념들을 이해하고 활용하여 모델을 설계하고 개발하는 것이 중요합니다.

