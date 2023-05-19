# Kaggle_Competition
## Store Sales - Time Series Forecasting

[Streamlit 링크](https://maestroyongseok-kaggle-storesales-app-1yt5gg.streamlit.app/) 
[발표 영상](https://youtu.be/0oyoY8s4few) 
[데모 시연](https://github.com/MaestroYongseok/Kaggle_SalesStore/blob/main/video/dashboard.avi) 
[발표 PPT](https://github.com/MaestroYongseok/Kaggle_SalesStore/blob/main/storeSales.pdf) 
![screensh](C:\Users\YONSAI\Desktop\Kaggle_StoreSales\img\main.jpg)<br/><br/>

## 1.프로젝트의 시작 (2023.04.20 ~ 2023.05.17)
- Kaggle Competition : Store Sales - Time Series Forecasting ( Use machine learning to predict grocery sales )
- 링크 : [Kaggle page](https://www.kaggle.com/c/store-sales-time-series-forecasting)
 
## 2. 대회 목표 : 식료품 소매업체의 데이터로 매장 매출을 예측 ( 시계열 예측)
- 여러 매장에서 판매되는 수천 가지 품목의 판매 단가를 더 정확하게 예측하는 모델을 구축
- 날짜, 매장 및 품목 정보, 프로모션, 판매 단가로 구성된 접근하기 쉬운 학습 데이터 세트를 통해 머신 러닝 기술을 연습

## 3. 참여 목적 :
- Python에 대해 공부한 내용을 바탕으로 머신러닝 실습 및 실력향상
- 대시보드 생성능력 향상.
- 시계열 분석 및 회귀 분석 등 통계 지식 향상

## 4. 데이터
- \[Kaggle Competition : Store Sales - Time Series Forecasting] 에서 제공하는 데이터를 이용하였습니다.
- [데이터 링크](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

## 5. ERD (개체 관계 모델)
![screensh](https://github.com/MaestroYongseok/Kaggle_SalesStore/blob/main/img/erd.jpg)

## 6. 팀 구성
- 사용언어 : Python : 3.9.13v
- 작업툴 : Pycharm
- 인원 : 3명
- 주요 업무 : Streamlit 라이브러리를 이용한 웹개발 구현 코드 작성 및 머신러닝을 활용한 매장 매출 예측
- 기간 : 2023.04.20 ~ 2023.05.17
***

## 7. 주요 기능
- Home
  + 멤버소개 및 프로젝트 개요
- Description
  + 프로젝트 목표 및 수행단계 소개
- Data
  + 데이터 설명 및 데이터 정의서 
- EDA
  + 유가 그래프, 판매량 그래프  
  + 각 품목의 기간 및 매장 별 판매량
  + 지진이 판매에 미치는 영향
  + 유가가 판매에 미치는 영향
- STAT
  + 시계열 분석 (ADF, Moving Average, Seasonal Forecast)
  + 두 평균의 비교
    * 각 품목 간 평균 비교 
- ML
  + 랜덤 포레스트 회귀를 통한 머신러닝
  + 데이터 예측
***

## 8. 설치 방법
### Windows
+ 버전 확인
    - vscode : 1.74.1
    - python : 3.9.13
    - 라이브러리 : pandas (1.5.3), numpy (1.23.5), plotly (5.14.1), matplotlib (3.7.1), streamlit (1.21.0), seaborn (0.12.2), pingouin (0.5.3), statsmodels (0.13.2), scikit-learn (1.2.2), xgboost (1.7.5), pandas-profiling (3.6.3), streamlit-option-menu (0.3.2), streamlit_pandas_profiling (0.1.3), scipy(1.9.1), 


- 프로젝트 파일을 다운로드 받습니다. 

```bash
git clone https://github.com/MaestroYongseok/Kaggle_SalesStore.git
```

- 프로젝트 경로에서 가상환경 설치 후 접속합니다. (Windows 10 기준)
```bash
virtualenv venv
source venv/Scripts/activate
```

- 라이브러리를 설치합니다. 
```bash
pip install -r requirements.txt
```

- streamlit 명령어를 실행합니다. 
```bash
streamlit run app.py
```

