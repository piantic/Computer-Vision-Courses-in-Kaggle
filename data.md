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

물론, 키워드를 하나씩 주제별로 잡고 알려드리겠습니다. HPO (Hyperparameter Optimization)에 대한 핵심 키워드 50가지는 다음과 같습니다:

Hyperparameter
Optimization
Search Space
Objective Function
Grid Search
Random Search
Bayesian Optimization
Genetic Algorithms
Simulated Annealing
Evolutionary Strategies
Tree of Parzen Estimators (TPE)
Sequential Model-Based Optimization (SMBO)
Surrogate Model
Exploration-Exploitation Tradeoff
Cross-Validation
Overfitting
Underfitting
Early Stopping
Learning Rate
Batch Size
Dropout Rate
Weight Decay
Activation Function
Number of Hidden Layers
Number of Neurons per Layer
Convolutional Kernel Size
Pooling Size
Learning Rate Schedule
Momentum
Regularization
Gradient Descent
Stochastic Gradient Descent (SGD)
Mini-Batch Gradient Descent
Adam Optimizer
RMSprop Optimizer
Learning Rate Decay
Momentum Decay
Network Architecture
Model Complexity
Feature Selection
Dimensionality Reduction
Early Stopping Criteria
Model Ensemble
Dropout Regularization
L1 Regularization (Lasso)
L2 Regularization (Ridge)
Hyperparameter Tuning Frameworks
Hyperopt
Optuna
Sklearn GridSearchCV


Hyperparameter: AI 모델의 학습 과정을 제어하는 변수로, 모델의 구조, 학습률, 배치 크기 등이 여기에 해당합니다.
Optimization: 목표 함수를 최적화하여 최상의 결과를 얻기 위한 과정입니다.
Search Space: Hyperparameter의 가능한 값들의 집합을 의미합니다.
Objective Function: 최적화 과정에서 최대화 또는 최소화하려는 함수로, 일반적으로 모델의 성능 지표를 나타냅니다.
Grid Search: 주어진 Search Space에서 모든 가능한 조합을 시도하여 최적의 Hyperparameter 조합을 찾는 방법입니다.
Random Search: 무작위로 선택된 Hyperparameter 값의 조합을 사용하여 최적의 조합을 찾는 방법입니다.
Bayesian Optimization: 확률적인 모델을 사용하여 더 효율적으로 Hyperparameter를 조정하는 방법입니다.
Genetic Algorithms: 생물학적 진화 원리를 모방하여 Hyperparameter 탐색 과정을 진행하는 방법입니다.
Simulated Annealing: 높은 온도에서 시작하여 시스템의 에너지를 점차 낮춰가면서 최적의 해를 탐색하는 방법입니다.
Evolutionary Strategies: 유전 알고리즘 기반의 Hyperparameter 최적화 방법입니다.
Tree of Parzen Estimators (TPE): 기존 시행 착오에 의존하는 베이지안 최적화 방법으로, Hyperparameter 조합의 우도를 추정합니다.
Sequential Model-Based Optimization (SMBO): Surrogate 모델을 사용하여 이전 반복에서 얻은 정보를 활용하여 Hyperparameter를 조정하는 방법입니다.
Surrogate Model: 실제 목적 함수를 대신하여 Hyperparameter 성능을 근사하는 모델입니다.
Exploration-Exploitation Tradeoff: 탐색과 활용 사이의 균형을 유지하며 Hyperparameter 탐색을 진행하는 것을 의미합니다.
Cross-Validation: 데이터를 여러 부분집합으로 나누어 모델의 일반화 성능을 평가하는 방법입니다.
Overfitting: 모델이 학습 데이터에 너무 과도하게 적합되어 새로운 데이터에서 성능이 저하되는 현상입니다.
Underfitting: 모델이 학습 데이터를 충분히 반영하지 못해 새로운 데이터에서도 성능이 낮은 현상입니다.
Early Stopping: 검증 데이터의 성능 지표가 더 이상 향상되지 않을 때 모델의 학습을 조기 종료하는 전략입니다.
Learning Rate: 모델이 각 학습 단계에서 업데이트되는 정도를 결정하는 하이퍼파라미터입니다.
Batch Size: 한 번의 업데이트에서 사용되는 학습 데이터의 샘플 개수를 의미합니다.

Dropout Rate: 신경망에서 일부 뉴런을 임의로 제거하여 과적합을 줄이는 정규화 기법입니다.
Weight Decay: 모델의 가중치를 줄이는 데 사용되는 정규화 기법으로, 오버피팅을 방지합니다.
Activation Function: 신경망의 뉴런에서 들어온 입력을 변환하여 출력을 계산하는 함수입니다.
Number of Hidden Layers: 신경망에서 입력 레이어와 출력 레이어 사이에 있는 숨겨진 레이어의 개수입니다.
Number of Neurons per Layer: 신경망의 레이어에 포함된 뉴런의 개수입니다.
Convolutional Kernel Size: 컨볼루션 레이어에서 사용되는 필터의 크기입니다.
Pooling Size: 풀링 레이어에서 사용되는 윈도우의 크기로, 이미지를 압축하여 공간 차원을 줄입니다.
Learning Rate Schedule: 학습률을 조절하는 데 사용되는 전략입니다. 학습의 진행에 따라 학습률을 동적으로 조정합니다.
Momentum: 경사 하강법에서 이전 업데이트의 영향을 고려하는 가중치입니다.
Regularization: 모델의 복잡도를 제어하여 과적합을 방지하는 기법입니다.
Gradient Descent: 최적화 알고리즘으로, 기울기를 사용하여 손실 함수를 최소화하는 방향으로 모델을 업데이트합니다.
Stochastic Gradient Descent (SGD): 각 학습 단계에서 무작위로 선택된 샘플을 사용하여 경사 하강법을 수행하는 방법입니다.
Mini-Batch Gradient Descent: 일부 샘플을 사용하여 경사 하강법을 수행하는 방법으로, 배치 크기보다 작은 크기의 미니배치를 사용합니다.
Adam Optimizer: Adaptive Moment Estimation의 약자로, 경사 하강법의 변형으로 널리 사용되는 최적화 알고리즘입니다.
RMSprop Optimizer: Root Mean Square Propagation의 약자로, 경사 하강법의 변형으로 학습률을 조정하는 최적화 알고리즘입니다.
Learning Rate Decay: 학습률을 점차적으로 감소시키는 전략입니다.
Momentum Decay: 모멘텀 값을 점차적으로 감소시키는 전략입니다.
Network Architecture: 모델의 구조를 결정하는 요소들의 집합입니다.
Model Complexity: 모델이 학습 데이터에 적합해질 수 있는 복잡성의 정도입니다.
Feature Selection: 모델에 가장 유용한 특성을 선택하는 과정입니다.
Dimensionality Reduction: 고차원 데이터의 차원을 줄이는 방법으로, 주성분 분석 (PCA) 등이 있습니다.
Early Stopping Criteria: 학습을 조기 종료하기 위해 사용되는 기준입니다.
Model Ensemble: 여러 모델을 조합하여 예측 결과를 향상시키는 기법입니다.
Dropout Regularization: 신경망에서 뉴런의 임의적인 제거를 통해 과적합을 방지하는 정규화 기법입니다.
L1 Regularization (Lasso): 가중치의 L1 노름을 추가하여 모델의 복잡성을 제한하는 정규화 방법입니다.
L2 Regularization (Ridge): 가중치의 L2 노름을 추가하여 모델의 복잡성을 제한하는 정규화 방법입니다.
Hyperparameter Tuning Frameworks: HPO를 수행하기 위해 사용되는 프레임워크들입니다.
Hyperopt: 베이지안 최적화를 기반으로 한 HPO를 지원하는 프레임워크입니다.
Optuna: 베이지안 최적화와 트리 기반 파젠 추정기를 활용한 HPO를 지원하는 프레임워크입니다.
Sklearn GridSearchCV: 사이킷런에서 제공하는 그리드 서치 기반의 HPO 도구입니다.

Hyperparameter: 모델의 학습 과정을 제어하는 변수로, 모델의 구조, 학습률, 배치 크기 등이 여기에 해당합니다.
Optimization: 목표 함수를 최적화하여 최상의 결과를 얻기 위한 과정입니다.
Search Space: Hyperparameter의 가능한 값들의 집합을 의미합니다.
Objective Function: 최적화 과정에서 최대화 또는 최소화하려는 함수로, 일반적으로 모델의 성능 지표를 나타냅니다.
Grid Search: 주어진 Search Space에서 모든 가능한 조합을 시도하여 최적의 Hyperparameter 조합을 찾는 방법입니다.
Random Search: 무작위로 선택된 Hyperparameter 값의 조합을 사용하여 최적의 조합을 찾는 방법입니다.
Bayesian Optimization: 확률적인 모델을 사용하여 더 효율적으로 Hyperparameter를 조정하는 방법입니다.
Surrogate Model: 실제 목적 함수를 대신하여 Hyperparameter 성능을 근사하는 모델입니다.
Cross-Validation: 데이터를 여러 부분집합으로 나누어 모델의 일반화 성능을 평가하는 방법입니다.
Overfitting: 모델이 학습 데이터에 너무 과도하게 적합되어 새로운 데이터에서 성능이 저하되는 현상입니다.
Underfitting: 모델이 학습 데이터를 충분히 반영하지 못해 새로운 데이터에서도 성능이 낮은 현상입니다.
Learning Rate: 모델이 각 학습 단계에서 업데이트되는 정도를 결정하는 하이퍼파라미터입니다.
Batch Size: 한 번의 업데이트에서 사용되는 학습 데이터의 샘플 개수를 의미합니다.
Dropout Regularization: 신경망에서 뉴런의 임의적인 제거를 통해 과적합을 방지하는 정규화 기법입니다.
Model Complexity: 모델이 학습 데이터에 적합해질 수 있는 복잡성의 정도입니다.
Feature Selection: 모델에 가장 유용한 특성을 선택하는 과정입니다.
Early Stopping: 검증 데이터의 성능 지표가 더 이상 향상되지 않을 때 모델의 학습을 조기 종료하는 전략입니다.
Gradient Descent: 최적화 알고리즘으로, 기울기를 사용하여 손실 함수를 최소화하는 방향으로 모델을 업데이트합니다.


HPO는 무엇을 의미하나요?
a) 모델의 학습 과정을 제어하는 변수
b) 최적화 과정에서 최대화 또는 최소화하려는 함수
c) Hyperparameter를 최적화하는 과정
d) 모델의 구조를 결정하는 요소들의 집합

Grid Search와 Random Search의 차이점은 무엇인가요?
a) Grid Search는 Hyperparameter 값을 무작위로 선택하고, Random Search는 가능한 모든 조합을 시도합니다.
b) Grid Search는 가능한 모든 조합을 시도하고, Random Search는 Hyperparameter 값을 무작위로 선택합니다.
c) Grid Search는 확률적인 모델을 사용하고, Random Search는 실제 목적 함수를 사용합니다.
d) Grid Search와 Random Search는 동일한 방법입니다.

어떤 Hyperparameter 최적화 방법이 확률적인 모델을 사용하여 Hyperparameter 성능을 근사화하는 방법인가요?
a) Grid Search
b) Random Search
c) Bayesian Optimization
d) Cross-Validation

Early Stopping의 목적은 무엇인가요?
a) 모델의 학습 과정을 제어하는 변수를 최적화하는 것입니다.
b) 최적화 과정에서 최대화 또는 최소화하려는 함수를 찾는 것입니다.
c) 모델이 학습 데이터에 과적합되는 것을 방지하기 위해 학습을 조기 종료하는 것입니다.
d) 모델의 구조를 결정하는 요소들의 집합을 결정하는 것입니다.

Gradient Descent는 무엇을 최소화하기 위한 알고리즘인가요?
a) 모델의 학습 과정을 제어하는 변수
b) 최적화 과정에서 최대화 또는 최소화하려는 함수
c) Hyperparameter를 최적화하는 과정
d) 모델의 구조를 결정하는 요소들의 집합

정답:

c) Hyperparameter를 최적화하는 과정
b) Grid Search는 가능한 모든 조합을 시도하고, Random Search는 Hyperparameter 값을 무작위로 선택합니다.
c) Bayesian Optimization
c) 모델이 학습 데이터에 과적합되는 것을 방지하기 위해 학습을 조기 종료하는 것입니다.
b) 최적화 과정에서 최대화 또는 최소화하려는 함수



HPO (Hyperparameter Optimization)의 개념에 대해 설명해주세요. HPO가 왜 중요한지 그 이유도 함께 서술해주세요.
HPO는 모델의 성능을 향상시키기 위해 Hyperparameter 값을 조정하는 과정입니다. Hyperparameter는 모델의 학습 과정이나 구조를 제어하는 변수들을 의미합니다. HPO의 목표는 최적의 Hyperparameter 조합을 찾아 모델의 성능을 극대화하는 것입니다. HPO가 중요한 이유는 다음과 같습니다:
Hyperparameter는 모델의 성능에 직접적인 영향을 미치기 때문에 적절한 조정이 필요합니다.
최적의 Hyperparameter 조합을 찾으면 모델의 성능이 향상되고, 일반화 능력이 향상됩니다.
HPO를 통해 시간과 자원을 효율적으로 사용할 수 있으며, 실험적인 접근보다 더 좋은 결과를 얻을 수 있습니다.
Grid Search와 Random Search는 각각 어떤 방식으로 Hyperparameter를 탐색하는지 설명해주세요. 두 방법 각각의 장단점을 비교하고, 어떤 상황에서 어느 방법을 선택해야 하는지 알려주세요.
Grid Search는 가능한 모든 Hyperparameter 조합을 탐색하는 방법입니다. 예를 들어, 주어진 범위에서 모든 조합을 시도하여 최적의 조합을 찾습니다.
Random Search는 Hyperparameter 값을 무작위로 선택하여 탐색하는 방법입니다. 무작위로 선택되는 값들을 통해 최적의 조합을 찾습니다.
Grid Search의 장점은 모든 조합을 시도하기 때문에 전체적인 탐색 범위를 잘 커버할 수 있다는 것입니다. 하지만 계산 비용이 크고, 차원이 높아질수록 탐색 공간이 기하급수적으로 커지는 단점이 있습니다.
Random Search의 장점은 계산 비용이 낮고, 탐색 공간을 무작위로 선택하기 때문에 다양한 영역을 빠르게 탐색할 수 있다는 것입니다. 하지만 전체적인 탐색 범위를 제대로 커버하지 못할 수 있습니다.
Grid Search는 탐색 공간이 작고 Hyperparameter 조합의 상호작용이 적을 때 유리하며, Random Search는 대규모 탐색 공간이나 Hyperparameter 조합의 상호작용이 큰 경우 유리합니다.
Bayesian Optimization은 HPO에서 어떤 원리를 기반으로 동작하는지 설명해주세요. Surrogate Model과 Acquisitions Function이 어떤 역할을 하는지도 함께 서술해주세요.
Bayesian Optimization은 확률적인 모델을 사용하여 Hyperparameter 성능을 근사화하는 방법입니다. 다음과 같은 원리로 동작합니다:
Surrogate Model: Bayesian Optimization은 목적 함수를 직접 사용하지 않고, 목적 함수의 근사를 위한 Surrogate Model을 구축합니다. Surrogate Model은 Hyperparameter 공간을 모델링하고, 이를 기반으로 다음으로 탐색할 위치를 선택합니다.
Acquisitions Function: Acquisitions Function은 Surrogate Model을 활용하여 어떤 위치에서 탐색을 진행할지 결정합니다. Acquisitions Function은 탐색-이용 트레이드오프를 고려하여 새로운 위치를 제안합니다. 대표적으로 Expected Improvement(EI)와 Upper Confidence Bound(UCB)가 있습니다.
Early Stopping이 HPO에서 어떤 역할을 하는지 설명해주세요. Early Stopping을 사용하면 어떤 이점이 있는지 설명하고, 주의해야 할 사항도 함께 언급해주세요.
Early Stopping은 모델이 학습 데이터에 과적합되는 것을 방지하기 위해 학습을 조기 종료하는 기법입니다. 학습 과정에서 모델의 성능이 더 이상 향상되지 않을 때 학습을 종료합니다.
Early Stopping의 이점은 다음과 같습니다:
과적합을 방지하여 모델의 일반화 능력을 향상시킵니다.
학습 시간과 자원을 절약할 수 있습니다.
주의해야 할 사항은 Early Stopping을 적용할 때 다음과 같은 사항을 고려해야 합니다:
적절한 조기 종료 기준을 설정해야 합니다. 일반적으로 검증 데이터의 성능이 향상되지 않는 동안 일정 기간 동안 종료하지 않는 것이 일반적입니다.
과적합이 발생하지 않는지 확인하기 위해 모델의 성능을 모니터링해야 합니다.
Gradient Descent는 HPO에서 어떤 방식으로 사용되는지 설명해주세요. Gradient Descent를 통해 어떤 값을 최적화하는지 그 예시를 들어 설명해주세요.
Gradient Descent는 최적화 알고리즘으로, HPO에서 모델의 성능을 최대화하거나 손실을 최소화하기 위해 사용됩니다. Gradient Descent는 다음과 같은 방식으로 사용됩니다:
모델의 손실 함수를 정의하고, 이를 최소화하는 방향으로 Hyperparameter 값을 조정합니다.
Gradient Descent는 현재 위치에서의 기울기(gradient)를 계산하고, 이를 이용하여 다음 위치를 결정합니다. 기울기의 반대 방향으로 이동하면서 점진적으로 최적값에 수렴합니다.
예를 들어, Neural Network의 학습에서 Learning Rate는 Hyperparameter 중 하나입니다. Gradient Descent를 사용하여 Learning Rate를 조정하면 모델의 학습 과정을 최적화할 수 있습니다. 적절한 Learning Rate 값을 찾기 위해 Gradient Descent를 사용할 수 있습니다.