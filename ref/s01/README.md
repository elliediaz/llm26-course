# 1회차 참고 결과

휴식 시간 안에 학습이 끝나지 않았거나 학습이 실패했을 때 실습 2·3을 이어 가기 위한 결과 파일임.

```bash
bash ref/s01/use_ref.sh      # results/·runs/에 없는 파일만 복사
```

| 파일 | 만든 명령 |
|---|---|
| `numerals_gru.json`, `numerals_gru_attn.json`, `numerals_transformer.json`, `numerals_by_length.png`, `numerals_loss.png` | `python src/tiny_seq2seq_train.py --arch all --minutes 2.5` |
| `numerals_gru_attn.pt` | 같은 학습의 gru_attn 체크포인트(`viz_attention.py tiny --arch gru_attn`이 읽음) |
| `cls_vs_reg.json`, `cls_vs_reg.png` | `python src/cls_vs_reg.py` |
| `attn_tiny_gru_attn.png` | `python src/viz_attention.py tiny --arch gru_attn --text "삼억 칠천이백만 오천 원"` |

- CPU에서 학습한 값이라 A100에서 각자 잰 값과 조금 다를 수 있음. 비교할 때는 내 결과를 기준으로 함
