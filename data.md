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

크롤링 (Web scraping): 웹 페이지에서 데이터를 자동으로 추출하는 과정 또는 기술입니다.
API (Application Programming Interface): 소프트웨어 애플리케이션 간 상호작용을 위해 정의된 인터페이스로, 데이터를 요청하고 응답을 받을 수 있는 방법을 제공합니다.
데이터셋 (Dataset): 일련의 관련 데이터를 모아놓은 구조로, 분석이나 모델 학습을 위해 사용됩니다.
스크래핑 (Scraping): 웹 페이지에서 정보를 수집하거나 추출하는 과정을 의미합니다.
수집 정책 (Collection policy): 데이터 수집 시 준수해야 할 규칙이나 지침을 정의한 정책입니다.
웹 크롤러 (Web crawler): 자동화된 방식으로 웹을 탐색하고 정보를 수집하는 프로그램 또는 스크립트입니다.
데이터 소스 (Data source): 데이터를 얻을 수 있는 원본이나 출처를 의미합니다.
데이터 필터링 (Data filtering): 데이터에서 원하는 부분만 추출하거나 선택하는 과정을 말합니다.
크롤링 도구 (Crawling tools): 웹 크롤링 작업을 돕기 위해 개발된 소프트웨어나 라이브러리입니다.
웹 스크랩링 (Web scraping): 웹 페이지에서 데이터를 추출하는 과정을 의미합니다.
인증 (Authentication): 데이터에 접근하기 위해 신원을 확인하는 과정입니다.
데이터 수집 방법 (Collection methods): 데이터를 수집하는 다양한 방법이나 절차를 의미합니다.
웹 데이터 (Web data): 웹 페이지에서 추출한 데이터로, HTML, XML, JSON 등의 형식을 가질 수 있습니다.
크롤링 제한 (Crawling restrictions): 웹 사이트에서 크롤러의 접근을 제한하는 규칙이나 정책을 말합니다.
데이터 정합성 (Data integrity): 데이터의 일관성, 정확성, 유효성을 의미합니다.
위의 핵심 키워드를 활용하여 데이터 수집에 관한 문제를 만들어보겠습니다:

문제: 크롤링을 통해 주어진 웹 사이트에서 책 정보를 수집하려고 합니다. 어떤 데이터 수집 방법을 사용하면 좋을까요? 그리고 데이터 수집 시 주의해야 할 점은 무엇일까요?

답안:
책 정보를 수집하기 위해 웹 크롤링을 사용할 수 있습니다. 웹 크롤러를 이용하여 웹 페이지를 탐색하고, 필요한 데이터를 추출할 수 있습니다. 크롤링 시에는 다음과 같은 주의사항을 고려해야 합니다:

크롤링 정책을 준수해야 합니다. 웹 사이트에는 크롤링에 대한 제한이나 정책이 설정될 수 있으므로, 로봇 배제 표준(Robots.txt)을 확인하고 지켜야 합니다.
적절한 딜레이를 설정해야 합니다. 웹 사이트에 과도한 요청을 보내면 서버에 부하를 줄 수 있으므로, 적절한 딜레이를 설정하여 서버에 대한 예의를 지켜야 합니다.
인증이 필요한 경우 인증 절차를 수행해야 합니다. 접근 권한이 필요한 웹 페이지의 경우, 로그인 정보나 API 키 등을 사용하여 인증 절차를 거쳐야 합니다.
데이터 필터링을 통해 필요한 정보만 추출해야 합니다. 크롤링된 데이터 중에서 필요한 부분만 선택하여 수집해야 하며, 필터링 작업을 통해 불필요한 정보를 제거할 수 있습니다.


이상치 (Outliers): 일반적인 데이터 분포에서 벗어나는 값으로, 데이터 분석과 예측에 부정적인 영향을 줄 수 있습니다.

결측치 (Missing values): 데이터셋에서 값이 빠져 있는 부분을 말하며, 이를 적절히 처리해야 데이터의 왜곡을 방지할 수 있습니다.

중복 데이터 (Duplicate data): 중복된 데이터 샘플은 모델 학습에 중복 정보를 제공하므로, 이를 식별하고 제거하여 데이터의 품질을 향상시킵니다.

잡음 (Noise): 데이터에 포함된 불필요한 혹은 잘못된 정보를 의미하며, 정제 과정에서 이를 제거하여 모델의 성능을 향상시킵니다.

오류 데이터 (Erroneous data): 잘못된 값이나 부적절한 형식으로 입력된 데이터를 의미합니다. 이를 찾아 수정하거나 제거하여 데이터의 신뢰성을 높일 수 있습니다.

이상한 값 (Anomalies): 데이터셋에서 예상치 못한 패턴이나 값으로, 이상치와 유사하지만 다른 개념입니다. 이상한 값은 데이터 분석 시 고려해야 할 중요한 측면입니다.

라벨 오류 (Label errors): 데이터의 라벨이 잘못 부여되거나 오류가 있는 경우를 말합니다. 이를 확인하고 수정하여 모델의 학습과 예측 결과를 개선할 수 있습니다.

이상한 패턴 (Anomalous patterns): 데이터셋에서 예상치 못한 패턴이나 규칙을 가진 데이터를 의미합니다. 이러한 패턴을 식별하고 조치를 취함으로써 데이터의 정확성을 향상시킬 수 있습니다.

데이터 정합성 (Data integrity): 데이터의 일관성과 정확성을 의미하며, 데이터 정제 과정에서 이를 유지하거나 개선하는 것이 중요합니다.

데이터 품질 (Data quality): 데이터의 정확성, 완전성, 일관성, 신뢰성 등을 포함한 전반적인 특성을 의미합니다. 데이터 품질을 향상시키는 것은 모델의 신뢰도를 높이는 데 도움이 됩니다.

데이터 정제 방법 (Cleaning methods): 데이터 정제에 사용되는 다양한 방법과 기법을 의미합니다. 이는 결측치 대체, 이상치 제거, 중복 데이터 처리 등을 포함합니다.

라벨 정제 (Label cleaning): 데이터의 라벨에 대한 오류를 수정하거나 라벨의 일관성을 확보하는 과정을 말합니다.

불완전한 데이터 (Incomplete data): 일부 데이터가 누락되거나 부족한 경우를 의미하며, 이를 처리하여 데이터의 완전성을 유지하거나 개선할 수 있습니다.

데이터 완전성 (Data completeness): 데이터가 완전하고 필요한 정보를 충분히 포함하고 있는지를 의미합니다. 데이터 완전성을 유지하는 것은 모델의 학습과 예측의 신뢰성을 높이는 데 도움이 됩니다.

불균형 데이터 (Imbalanced data): 클래스 간 불균형이 발생하는 경우를 말합니다. 일부 클래스의 샘플 수가 다른 클래스에 비해 현저히 적은 경우, 적절한 처리를 통해 불균형을 해소할 필요가 있습니다.




이상치(Outliers)란 무엇인가요?

정답: 데이터 집합에서 일반적인 패턴과 동떨어진 값으로, 다른 데이터와 비교했을 때 통계적으로 유의미한 차이를 가지는 값입니다.
결측치(Missing values)란 무엇인가요?

정답: 데이터 집합에서 일부 변수 또는 관측치가 비어있는 값이거나 존재하지 않는 값을 가지는 것을 의미합니다.
중복 데이터(Duplicate data)란 무엇인가요?

정답: 데이터 집합에서 동일한 값이 중복되어 나타나는 것을 말합니다.
잡음(Noise)이란 무엇인가요?

정답: 데이터에 포함된 불필요한 정보나 무작위로 발생한 오류로 인해 데이터의 정확성이나 품질이 저하되는 것을 의미합니다.
오류 데이터(Erroneous data)란 무엇인가요?

정답: 부정확하거나 잘못된 값 또는 정보를 가지고 있는 데이터를 말합니다.
이상한 값(Anomalies)과 이상치(Outliers)의 차이점은 무엇인가요?

정답: 이상치는 일반적인 패턴에서 벗어난 값으로, 통계적으로 유의미한 차이를 가지는 것을 말합니다. 이상한 값은 데이터 집합 내에서 이상한 패턴이나 특이한 동작을 보이는 값입니다.
라벨 오류(Label errors)란 무엇인가요?

정답: 데이터 집합의 라벨 또는 클래스가 잘못 라벨링되어 있는 오류를 말합니다.
이상한 패턴(Anomalous patterns)이란 무엇인가요?

정답: 데이터 집합 내에서 다른 데이터와는 다른 패턴을 가진 관측치 또는 데이터 그룹을 의미합니다.
데이터 정합성(Data integrity)이란 무엇인가요?

정답: 데이터가 정확하고 일관성이 있으며 신뢰할 수 있는 상태를 유지하는 것을 의미합니다.
데이터 품질(Data quality)을 향상시키기 위한 몇 가지 방법을 예를 들어 설명해주세요.

정답: 데이터 품질을 향상시키기 위한 방법으로는 이상치 제거, 결측치 처리, 중복 데이터 제거, 오류 데이터 수정, 라벨 정제, 데이터 완전성 확인 등이 있습니다.


데이터 수집 (Collection):
데이터 수집은 다양한 방법을 통해 데이터를 수집하는 과정입니다. 예를 들어, 웹 크롤링이나 API를 통해 데이터를 수집할 수 있습니다. 데이터 수집 과정에서는 데이터 소스를 선정하고 수집 정책을 설정하는 등의 결정을 내려야 합니다. 또한 데이터 필터링을 통해 원하는 데이터만을 추출할 수도 있습니다.

데이터 정제 (Clean Labels/Samples):
데이터 정제는 수집한 데이터에서 이상치, 결측치, 중복 데이터, 잡음 등을 제거하고 데이터의 정합성과 품질을 향상시키는 과정입니다. 이상치는 다른 데이터와 동떨어진 값으로, 이를 제거하여 데이터의 일관성을 유지할 수 있습니다. 결측치는 누락된 데이터 값으로, 이를 적절한 방법으로 채워넣을 수 있습니다. 중복 데이터는 중복된 정보를 제거하여 데이터의 중복성을 줄입니다. 잡음은 데이터에 포함된 불필요한 정보로, 이를 제거하거나 보정함으로써 데이터 품질을 개선할 수 있습니다.

데이터 증강 (Data Up/Down):
데이터 증강은 기존의 데이터를 변형하거나 새로운 데이터를 생성하여 데이터의 양을 늘리는 과정입니다. 이를 통해 데이터의 다양성을 확보하고 모델의 성능을 향상시킬 수 있습니다. 데이터 증강 방법으로는 이미지 변환, 증식, 보간, 회전, 가중치 조정 등 다양한 기법이 있습니다. 이를 적절히 활용하여 데이터를 확대하고 변형함으로써 모델의 학습에 도움을 줄 수 있습니다.



이상치(Outliers)에 대해 설명하고, 이상치를 처리하는 방법을 소개해주세요.

결측치(Missing values)가 데이터에 포함되어 있는 경우, 결측치 처리를 위해 사용할 수 있는 몇 가지 기법을 설명하고, 각 기법의 장단점을 언급해주세요.

중복 데이터(Duplicate data)가 데이터 집합에 존재할 때, 중복 데이터 처리를 위해 사용할 수 있는 방법을 설명하고, 중복을 판단하고 처리하는 기준에 대해서도 언급해주세요.

잡음(Noise)이 데이터에 영향을 미치는 경우, 잡음 감소를 위해 사용할 수 있는 몇 가지 방법을 설명하고, 각 방법의 특징과 적용 가능한 상황에 대해서도 언급해주세요.

라벨 오류(Label errors)를 처리하기 위해 사용할 수 있는 방법과 절차를 설명하고, 라벨 오류를 식별하고 수정하기 위한 일반적인 접근 방식에 대해서도 언급해주세요.

답변:

이상치(Outliers)는 데이터 집합에서 다른 값들과 동떨어진 극단적인 값들을 의미합니다. 이상치는 데이터 분석 결과를 왜곡시킬 수 있으므로 적절한 처리가 필요합니다. 이상치를 처리하는 방법으로는 다음과 같은 방법들이 있습니다:
통계적 기법: 평균, 중앙값, 표준편차 등의 통계적 기준을 사용하여 이상치를 탐지하고 제거합니다.
시각화 기법: 상자 그림(box plot)이나 산점도(scatter plot)를 통해 이상치를 시각적으로 탐지하고 제거합니다.
도메인 지식 활용: 도메인 지식을 활용하여 이상치를 판단하고 처리합니다.
결측치(Missing values)는 데이터에서 값이 비어 있는 경우를 말합니다. 결측치 처리를 위해 사용되는 몇 가지 기법은 다음과 같습니다:
제거: 결측치가 있는 행이나 열을 제거하여 데이터를 처리합니다. 단점으로는 데이터 손실이 발생할 수 있습니다.
대체: 평균, 중앙값, 최빈값 등의 대표값이나 보간법을 사용하여 결측치를 대체합니다. 단점으로는 대체된 값이 원래 데이터의 특성을 완벽하게 대표하지 못할 수 있습니다.
중복 데이터(Duplicate data)는 데이터 집합에서 동일한 정보를 포함하는 중복된 데이터를 말합니다. 중복 데이터 처리를 위해 사용할 수 있는 방법은 다음과 같습니다:
제거: 중복된 데이터를 제거하여 데이터를 정제합니다. 중복을 판단하는 기준으로는 모든 특성 값이 동일한 경우나 주요 특성 값을 기준으로 판단할 수 있습니다.
잡음(Noise)은 데이터에 포함된 불필요한 정보나 오차로 인해 데이터 분석 결과를 왜곡시키는 요소입니다. 잡음을 감소시키기 위해 사용할 수 있는 몇 가지 방법은 다음과 같습니다:
필터링: 필터링 기법을 사용하여 데이터에서 잡음을 제거합니다. 예를 들어, 이동 평균 필터링, 가우시안 필터링 등이 있습니다.
정규화: 데이터를 정규화하여 잡음을 감소시킵니다. 예를 들어, Min-Max 정규화나 Z-score 정규화를 사용할 수 있습니다.
라벨 오류(Label errors)를 처리하기 위해 사용할 수 있는 방법은 다음과 같습니다:
검토 및 수정: 라벨이 잘못되었을 가능성이 있는 데이터를 식별하고, 도메인 전문가나 추가 검토를 통해 라벨을 수정합니다.
일괄적인 수정: 전체 데이터 세트에서 일괄적으로 오류가 있는 라벨을 수정합니다. 예를 들어, 오류가 있는 라벨을 모두 제거하거나 다른 올바른 라벨로 대체합니다.


이상치(Outliers) 처리:
문제: 주어진 데이터셋에서 이상치를 탐지하고 제거하는 PyTorch 코드를 작성해보세요.

python
Copy code
import torch
import torch.nn as nn
import torch.optim as optim

# 데이터셋 로드
dataset = torch.tensor([1.2, 2.1, 0.8, 3.9, 100.0, 2.5, 1.0, 0.9])

# 이상치 탐지
mean = torch.mean(dataset)
std = torch.std(dataset)
threshold = mean + 3 * std  # 통계적 기준을 사용하여 이상치 임계값 설정
outliers = dataset[dataset > threshold]

# 이상치 제거
clean_dataset = dataset[dataset <= threshold]

print("Original dataset:", dataset)
print("Detected outliers:", outliers)
print("Cleaned dataset:", clean_dataset)
결측치(Missing values) 처리:
문제: 주어진 데이터셋에서 결측치를 평균값으로 대체하는 PyTorch 코드를 작성해보세요.

python
Copy code
import torch
import torch.nn as nn
import torch.optim as optim

# 데이터셋 로드 (결측치 포함)
dataset = torch.tensor([1.2, 2.1, float("nan"), 3.9, 2.5, float("nan"), 1.0, 0.9])

# 결측치 대체
mean = torch.nanmean(dataset)  # 결측치를 제외한 평균 계산
clean_dataset = torch.where(torch.isnan(dataset), mean, dataset)  # 결측치를 평균값으로 대체

print("Original dataset:", dataset)
print("Cleaned dataset:", clean_dataset)
중복 데이터(Duplicate data) 처리:
문제: 주어진 데이터셋에서 중복 데이터를 제거하는 PyTorch 코드를 작성해보세요.

python
Copy code
import torch
import torch.nn as nn
import torch.optim as optim

# 데이터셋 로드 (중복 데이터 포함)
dataset = torch.tensor([1, 2, 3, 2, 4, 3, 5, 1])

# 중복 데이터 제거
unique_dataset = torch.unique(dataset)

print("Original dataset:", dataset)
print("Dataset without duplicates:", unique_dataset)
잡음(Noise) 감소:
문제: 주어진 데이터셋에 가우시안 잡음을 추가하고, 잡음을 감소시키는 PyTorch 코드를 작성해보세요.

python
Copy code
import torch
import torch.nn as nn
import torch.optim as optim

# 데이터셋 로드
dataset = torch.tensor([1.2, 2.1, 0.8, 3.9, 2.5, 1.0, 0.9])

# 가우시안 잡음 추가
noise = torch.randn(dataset.shape)  # 잡음 생성
noisy_dataset = dataset + noise

# 잡음 감소
denoised_dataset = torch.median(noisy_dataset)

print("Original dataset:", dataset)
print("Noisy dataset:", noisy_dataset)
print("Denoised dataset:", denoised_dataset)
