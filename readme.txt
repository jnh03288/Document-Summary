softvote
├── config
├── data
├── model
├── modules
├── output
├── pred_elec.py
├── pred_funnel.py
├── predict.py
├── pred_kobert.py
├── requirements.txt
├── results
├── train_elec.py
├── train_funnel.py
├── train_kobert.py
└── train.py

25 directories

config
각 모델 별 학습 및 추론 파라매터가 저장된 yml 파일이 존재하는 폴더
data
train, test 데이터 경로로부터 각 모델별로 사용가능한 형태로 전처리하여 저장하는 폴더
model
문장 요약 모델 3종에 대한 파이썬 코드가 있는 폴더
modules
학습 및 추론에 필요한 여러 기능들을 구현한 파이썬 코드가 존재하는 폴더
pred_elec.py
electra 모델을 통해 추론하는 코드
pred_funnel.py
funnel모델을 통해 추론하는 코드
predict.py
python3 predict.py
위 명령어를 통해 수행하면 kobert, funnel, electra 세 모델의 추론 결과를 soft voting 방식으로 결과를 출력한다.
pred_kober.py
kobert 모델을 통해 추론하는 코드
results
학습된 모델이 저장되는 폴더

train.py 
python3 train.py
위 명령어를 통해 수행하면 실행 되어 kobert, funnel, electra 세 모델을 학습시켜 results 폴더에 그 결과를 저장한다.
train_elec.py
electra 모델을 학습하는 코드
train_funnel.py
funnel 모델을 학습하는 코드
train_kobert.py
kobert 모델을 학습하는 코드

output
└── result.json

output
predict.py가 실행되어 나온 요약 json 파일을 저장하는 폴더
result.json
문서 요약 결과가 json 형태로 기록된 파일

weights
├── elec.pt
├── funnel.pt
└── kobert.pt

elec.pt
학습된 electra 모델 가중치 파일
funnel.pt
학습된 funnel 모델 가중치 파일
kobert.pt
학습된 kobert 모델 가중치 파일

학습된 가중치 파일로 예측하기 위해서 각 모델 파일을 
softvote/results/train/elec/
softvote/results/train/funnel/
softvote/results/train/kobert/
로 이동한후 각 가중치 파일들의 이름을 모두 best.pt로 설정하여 predict.py 코드를 실행한다.