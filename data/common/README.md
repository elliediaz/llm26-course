# common: 도메인과 상관없이 쓰는 데이터

| 폴더·파일 | 회차 | 내용 | 출처·라이선스 |
|---|---|---|---|
| `fashion_mnist/` | 1 | 28×28 흑백 옷 사진 10종, 학습 60,000·시험 10,000(IDX gz 원본 4개) | Zalando Research Fashion-MNIST, MIT. https://github.com/zalandoresearch/fashion-mnist |
| `nsmc/train.tsv`, `test.tsv` | 1 | 네이버 영화 리뷰 감성(긍정 1·부정 0) 부분 집합 30,000·10,000건(긍정·부정 같은 수). 만든 방법 `nsmc/make_subset.py` | Naver Sentiment Movie Corpus v1.0(e9t), CC0 1.0. https://github.com/e9t/nsmc |
| `numerals/` | 1 | 한글 금액 수사 → 숫자 변환 합성 데이터(첫 실행 때 `make_numerals.py`가 만듦) | 규칙 생성 |
| `sentences.json` | 2 | 토큰 수 비교 문장 | 자체 작성 |
| `sandbox_questions.txt` | 5 · 6 · 7 | 일상 질문 460건(한 줄에 하나). 7회차 학습 데이터를 만들 때와 5회차 형식 실험의 기본 질문으로 씀 | 자체 작성(가상). 실존 인물 · 상호 · 연락처 없음 |

- NSMC는 실제 사용자 리뷰라 맞춤법이 틀리거나 거친 표현이 섞여 있음. 리뷰 id 열은 뺐음
- 두 공개 데이터 모두 개인정보가 없고 해외 서버에 올려도 되는 조건임(AIHub 데이터는 이 저장소에 넣지 않음)
- `sandbox_questions.txt`는 주제가 고르게 섞이도록 손으로 쓴 질문이라 실제 사용자 질문보다 문장이 반듯함. 오타나 줄임말이 섞인 입력은 5회차 노트북의 '문제 입력' 단계에서 따로 넣어 봄
