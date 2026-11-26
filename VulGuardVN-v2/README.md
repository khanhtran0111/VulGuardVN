# VulGuard v2 — pipeline chạy local

Cài đặt đầy đủ của [VulGuard_v2_implementation_plan.md](VulGuard_v2_implementation_plan.md), chạy hoàn toàn trên
máy local (Quadro P2000 4 GB, 12 nhân, 30 GB RAM). Chỉ tầng LLM gọi API OpenRouter (`openai/gpt-oss-20b`,
provider `coreweave/fp4`). Không cần Kaggle.

## 1. Cài đặt (đã làm sẵn trên máy này)

```bash
cd VulGuardVN-v2
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu126 torch==2.14.1
uv pip install --python .venv/bin/python -r requirements.txt
```

| Thành phần | Vị trí | Ghi chú |
|---|---|---|
| Python 3.12 + torch 2.14.1 cu126 | `.venv/` | bản cu126 còn kernel sm_60 cho GPU Pascal; bản cu128+ đã bỏ Pascal |
| JDK 21 (Temurin 21.0.13.1) | `tools/jdk-21/` | không cần Java hệ thống |
| Joern v4.0.674 | `tools/joern-cli/` | chốt version, checksum sha512 đã kiểm |
| UniXcoder `microsoft/unixcoder-base-nine` | `models/unixcoder-base-nine/` | |
| Dữ liệu | `../data/{primevul,bigvul,devign,reveal}` | |
| API key | `.env` → `OPENROUTER_API_KEY=` | **bạn tự điền**; nhớ đặt trần 100 USD trên OpenRouter |

Kiểm tra môi trường: `.venv/bin/python -m v2.scripts.run_p0_check`.

## 2. Chạy

```bash
./run_all.sh local     # P0-P3: dữ liệu, Joern, feature, 25 model prefilter/dataset, ablation A1/A3   (không tốn API)
./run_all.sh dev       # P4 bước 4: vòng phát triển prompt trên dev set 300 hàm (~1-2 USD) -> dev_report.json
./run_all.sh freeze    # P4 bước 5: đóng băng hash prompt (D6). Chỉ chạy khi dev_report.decision_point_step4.ok_to_freeze
./run_all.sh llm       # P4: A-pool -> test (+ zero-shot B3), A7/A8, RQ5 (gpt-oss-120b), kiểm tra độ bất định
./run_all.sh offline   # P5 kiểm chứng CPG, P6 tầng quyết định, P7 thí nghiệm + độ bền                (không tốn API)
./run_all.sh b7        # (tùy chọn) baseline B7: fine-tune UniXcoder trên GPU local
./run_all.sh status    # token / USD đã dùng
./run_all.sh smoke     # toàn bộ pipeline trên tập con nhỏ với LLM mock (artifacts_smoke/, không tốn API)
```

### Chạy từng dataset, từng seed

`./run_<dataset>.sh <phase> [--reps ...]` (= `./run_dataset.sh <dataset> <phase> ...`), dataset ∈
`primevul | bigvul | devign | reveal`:

| Phase | Làm gì | API |
|---|---|---|
| `check` | trạng thái từng bước + lệnh nên chạy tiếp | không |
| `prep` | P1 dữ liệu, P2 Joern, P3 feature (UniXcoder lâu, resume được) | không |
| `train --reps 1,2` | train prefilter cho các rep chọn, rồi chấm điểm + calibrate | không |
| `ablations` | retrain A1 (bỏ từng view) và A3 (bỏ loss theo cặp) | không |
| `estimate` | kiểm tra điều kiện + ước tính USD, không gọi API | không |
| `llm` | P4 A-pool → test (+ zero-shot); với PrimeVul thêm A7/A8, RQ5, độ bất định | **có** |
| `offline` | P5 + P6 cho dataset này, P7 gộp mọi dataset đã xong | không |
| `b7 --seed 42` | baseline B7 (fine-tune UniXcoder trên GPU local) | không |
| `all` | prep → train → llm → offline | **có** |

**Seed = rep.** Rep `r` dùng ensemble seed `{10r, …, 10r+4}`, đồng thời là seed của router (LightGBM, mẫu con 80%
của A) và của baseline định tuyến ngẫu nhiên (§7.5). Các rep độc lập, resume được, chạy theo thứ tự bất kỳ:

```bash
./run_devign.sh prep
./run_devign.sh train --reps 1        # hôm nay 1 rep
./run_devign.sh train --reps 2,3      # hôm sau thêm rep
./run_devign.sh offline               # P6/P7 luôn dùng mọi rep đã train + chấm điểm
```

Tầng LLM không phụ thuộc seed (D5), nên chỉ chạy **một lần** cho mỗi dataset. Thêm rep sau đó không tốn API.
Trước lần `llm` đầu tiên của bất kỳ dataset nào: `./run_all.sh dev` (dev set PrimeVul) → `./run_all.sh freeze`.
`llm` luôn tự kiểm tra trước khi gọi API (Joern đủ, feature store có, prompt đã đóng băng, API key, ước tính USD
so với trần chi tiêu) và dừng nếu thiếu.

A7/A8, RQ5, kiểm tra độ bất định và thí nghiệm độ bền chỉ chạy cho PrimeVul. `results/`, `prefilter_metrics.csv`,
`p6_summary.json` luôn gộp mọi dataset đã chạy; phép chuyển giao router giữa dataset (RQ3) chỉ có khi ≥ 2 dataset xong P6.
`./run_all.sh` vẫn dùng được để chạy nhiều dataset cùng lúc.

`DATASETS=primevul,devign ./run_all.sh local` để chạy nhiều dataset cùng lúc. Mọi bước đều resume được: chạy lại sẽ bỏ qua
phần đã có (chunk UniXcoder, seed đã train, batch Joern, khóa cache LLM). Lệnh chạy lâu nên tách khỏi terminal:

```bash
setsid nohup ./run_all.sh local > v2/artifacts/logs/local.log 2>&1 &
```

Từng script cũng chạy riêng được: `.venv/bin/python -m v2.scripts.run_pN_... --help`.

## 3. Cấu trúc (mục 4 của plan)

```
v2/
  configs/     datasets.yaml, prefilter.yaml, llm.yaml, decision.yaml, experiments.yaml, cpg.yaml
  data/        loaders (primevul, bigvul, devign, reveal), dedup, splits, A/B, dev set, demo bank
  features/    heuristic graph + 24 numeric, token/skeleton ids, UniXcoder feature store
  cpg/         extract.sc (Joern script) + runner theo lô
  prefilter/   model 4 view (PyTorch), loss theo cặp, ensemble, calibrator có trọng số
  llm/         client OpenRouter + cache, prompts/*.j2, schemas/*.json, retrieval, runner
  verify/      cpg_decide.py (V1-V3, bảng quyết định 6.5)
  decision/    conformal, bayes, router (LightGBM), cost_model, budget, frame
  eval/        metrics (VD-S, pair metrics, F1, MCC...), simulate, stats
  experiments/ transforms.py (đổi tên biến, chèn dead code)
  scripts/     run_p0_check ... run_p7_experiments, run_p7_robustness, run_b7_unixcoder_ft
  localizer.py
  artifacts/   (gitignore) records, features, cpg_cache, prefilter, llm (llm_cache.jsonl), decision, predictions, results
```

Output chính: `v2/artifacts/data_card.md`, `{ds}/cpg_cache/stats.json`, `prefilter_metrics.csv`,
`{ds}/llm/dev_report.json`, `{ds}/llm/verify_stats.json`, `{ds}/llm/manual_review_30.md`, `p6_summary.json`,
`results/tables.md`, `results/*.csv`, `results/figures/*.png`. Mỗi lần chạy ghi config hash vào `artifacts/run_log.jsonl`.

## 4. Thời gian đo trên máy này

| Việc | Đo được | Plan ước lượng |
|---|---|---|
| P1 dữ liệu (4 dataset) | ~3 phút (Big-Vul CSV 10.8 GB chiếm ~2 phút) | |
| Joern | ~50 hàm/giây với 2 JVM → ~60k hàm trong ~20-30 phút | 1-2 ngày |
| UniXcoder embedding | 15-30 hàm/giây (fp32) → ~460k hàm trong **~6-9 giờ** | 1-2 giờ |
| Heuristic feature | < 1 phút / 30k hàm (10 tiến trình) | vài giờ |
| Prefilter 1 model | Devign 19.5k hàm train: 7.4 s/epoch, 12 epoch ≈ 1.5 phút, ~1 GB GPU | |
| Prefilter 25 model (ngoại suy) | PrimeVul/Big-Vul ≈ 12 phút/model → ~5 giờ; Devign/ReVeal ≈ 40 phút | 1 đêm/dataset |

GPU P2000 là Pascal: fp16 chậm gấp ~3 lần fp32 (đã đo), nên mọi thứ chạy fp32.

## 5. Khác biệt so với plan (và lý do)

1. **View AST-like = skeleton của code** (keyword/toán tử/`id`/`call`/`lit`, hàm `build_skeleton` của v1). Chuỗi
   AST từ đồ thị heuristic chỉ có ~10 ký hiệu, gần như không có tín hiệu. 24 đặc trưng numeric vẫn lấy từ đồ thị heuristic (D7).
2. **Nhãn router đổi tên thành `delta(x)`** (plan gọi là `u(x)`, trùng với phương sai ensemble `u(x)` đang là đặc trưng).
3. **Kiểm chứng V1-V3 offline**: Joern chạy một lần ở P2 và xuất sẵn mọi thông tin cần (lời gọi/deref/index theo dòng,
   biến bị taint từ tham số/nguồn qua DDG, điều kiện dominate hoặc control mỗi dòng). P5 không phải gọi lại Joern.
   V2 dùng DDG reachability trong hàm (xấp xỉ `reachableByFlows`, nhanh và tất định).
4. **Demo bank không cần Joern**: demo là vùng diff (dòng thay đổi ± 3 dòng) giữa bản lỗi và bản đã vá; embedding
   UniXcoder của vùng lỗi. Query = slice CPG quanh sink của giả thuyết mạnh nhất.
5. **Big-Vul dùng `MSR_data_cleaned.csv`** (188,636 hàng → test 18,398 sau dedup), không phải bản HF 217k mà v1 dùng.
   Cặp lỗi/đã vá: 8,284 (bản vá là record phụ `in_main=False`, chỉ dùng cho pair metrics và loss theo cặp).
6. **Không có `commit_date`** trong dữ liệu phát hành; split của PrimeVul vốn đã theo thời gian.
7. **Joern parse rate**: hàm C++ bị gán là C (file_name rỗng), hàm thiếu kiểu trả về và annotation kernel (`__user`...)
   làm c2cpg thất bại. Runner nhận diện C++, thay annotation bằng khoảng trắng cùng độ dài (giữ số dòng), thêm `int`
   khi thiếu kiểu trả về, và chạy lại lượt 2 với chế độ ngôn ngữ đảo cho hàm còn lỗi.
8. A-pool PrimeVul: 300 lỗi + 2,700 không lỗi như plan (B còn 291 lỗi → α nhỏ nhất 0.34%). ReVeal: B chỉ có 101 lỗi →
   α nhỏ nhất 0.98%, nên α = 0.5% không đạt được (ghi trong `conformal.json`).
9. A11 được cài như "gán thẳng top-p̂ ở cùng recall với v2" (so sánh FP ở cùng điểm vận hành).
10. **Chưa cài**: phần tùy chọn "1,000 hàm đã biến đổi đi qua tầng LLM" của P7 bước 5 (phần prefilter/skip đã có).
12. **View thứ 5 `finetune` + B7 (2026-11-23)**: UniXcoder fine-tune trên GPU local, **cross-fitting K=5 fold** trên
    train (fold theo cặp, phân tầng theo nhãn). Hàm train nhận logit out-of-fold, A/B/test nhận trung bình 5 model →
    không rò rỉ vào prefilter. Cùng lần chạy cho luôn baseline B7 (ensemble 5 model, ngưỡng F1 tốt nhất trên train_dev).
    Cấu hình `prefilter.yaml → finetune` (fp32, batch 8 × accum 4; từ 2026-11-25: 3 epoch, max_len 512, xem mục 17).
    Devign: 4.5 mẫu/s ở 320 token → ~11 giờ cho 5 fold × 2 epoch; resume theo epoch. Bỏ view: xoá `finetune` khỏi
    `model.views`.
13. **`reasoning.effort: medium`** (D2 cũ: low) và `max_tokens: 8192` để phần suy luận không cắt cụt JSON. Mọi lời gọi
    LLM chạy lại (khoá cache khác), vòng dev + đóng băng làm lại. `./run_<ds>.sh freeze --if-ok` chỉ đóng băng khi
    dev_report báo `ok_to_freeze`.
14. **Vòng dev 2026-11-24** (quay lại `reasoning: low`; medium cho v2 0.12 < zero-shot 0.14, 5.6% lời gọi bị cắt ở
    8,192 token, chi phí ×3):
    - `max_tokens` 16,384; ngân sách code trong prompt 400 dòng / 24,000 ký tự (trước 220 / 14,000), slice 40 dòng, 12 sink.
    - Prompt mới (bản cũ ở `v2/llm/prompts/history/2026-11-23_*`): Pass 1 chỉ đưa giả thuyết khi chỉ ra được check
      thiếu cụ thể; Pass 2 trung lập theo 3 bước (liệt kê check → tìm đường đi không có guard → so với cặp tham chiếu);
      Pass 2 bác bỏ dùng cặp tham chiếu. Prompt zero-shot (B3) giữ nguyên.
    - Luật §6.5 "bác bỏ + không có giả thuyết": plan = 1, nay = 0 (`decision.yaml: refute_without_hypothesis`), vì với
      Pass 1 mới "không có giả thuyết" là bằng chứng an toàn (dev: bác bỏ P-C 0.05 → 0.20). Luật gốc = ablation **A12**.
    - Kết quả dev: v2 trung lập 0.20, bác bỏ 0.20, zero-shot 0.16, JSON hợp lệ 100% → đóng băng.
11. **Vòng dev (P4 bước 4), 2026-11-23** — chỉ dùng dev set PrimeVul (lấy từ train), trước khi đóng băng:
    - Lần 1: v2 P-C 0.10 < zero-shot 0.12 → không đóng băng. Nguyên nhân: V3 coi mọi điều kiện nhắc tới biến là guard
      (62/105 lần bác), V1 không khớp sink dạng mô tả, V2 bác các biến global/field.
    - Sửa luật kiểm chứng (v2.1, `verify/cpg_decide.py`): V3 chỉ tính điều kiện khớp `required_guard` (≥ 2/3 định danh,
      đúng loại so sánh); V1 khớp tên sink mềm hơn; V2 chấp nhận field/global; chế độ bác bỏ chấp nhận thêm lệnh gán
      kẹp giới hạn. Sửa prompt `pass2_refute.j2` (liệt kê các kiểu bảo vệ cần tìm).
    - Lần 2: v2 trung lập P-C **0.18**, bác bỏ 0.11, zero-shot 0.12 → đóng băng hash prompt. Chi phí cả vòng dev ≈ 0.1 USD.
15. **Điểm vận hành chọn trên B (2026-11-25)**, `decision/operating.py`. Ngoài các cost ratio đặt trước (cr1/3/5/10,
    θ* = 1/(1+r)), thêm `f1B`: θ chọn trên B (tập hiệu chỉnh, không bao giờ nhìn test) để F1 của prefilter lớn nhất;
    tỷ lệ ngầm r = (1−θ)/θ dùng cho nhãn router `delta` và chọn chế độ LLM. Cấu hình chính (`experiments.yaml → main`)
    dùng `f1B`; `cr5` là luật cũ. Expected cost trong bảng baseline/ablation luôn báo ở c_FN/c_FP = 5 (`eval_cost_ratio`).
    Lý do: trên Devign θ* = 1/6 gắn "lỗi" cho gần hết (recall 0.98, specificity 0.17); θ trên B ≈ 0.24–0.35.
16. **Router guard (2026-11-25)**, `decision.yaml: router_guard`. Router học được chỉ được gửi hàm ở ngân sách B nếu lợi
    ích out-of-fold của nó trên A tại B là > 0; ngược lại B = 0 → v2 không bao giờ tệ hơn prefilter trên A. Router
    `learned_guard` trong lưới; ablation **A13** = không guard. Lý do: trên Devign tầng LLM gần như không có tín hiệu
    (P(sạch | prefilter "lỗi", LLM "sạch") = 0.48–0.53 so với nền 0.50), router học được làm giảm F1.
    Devign với prefilter cũ: v2 F1 0.658 → **0.671**, MCC 0.239 → **0.286** (= B7, p = 0.91).
17. **Fine-tune 512 token, 3 epoch (2026-11-25)**: 52% hàm Devign dài hơn 320 token (320 chỉ giữ 35% lượng code).
    max_len 320 → 512, 2 → 3 epoch, lịch LR tuyến tính (warm-up 6%), batch gom theo độ dài (sắp theo độ dài trong
    từng khối ngẫu nhiên rồi xáo các batch) → chi phí/epoch ≈ 1.1× bản 320 thay vì ~1.8×. Đỉnh GPU 2.4–2.8 GB. Resume
    nạp checkpoint vào CPU (trước đây giữ một bản trên GPU). Kết quả trước khi sửa: `v2/artifacts_snapshots/`.
    **Chốt 2026-11-26: 512 token, 2 epoch, LR cố định, giữ batch gom theo độ dài** cho mọi dataset. Lần chạy 512/3 epoch
    trên Devign: AUC view fine-tune 0.716 → 0.717 (không có ý nghĩa). Theo độ dài hàm: ≤ 320 token (cùng input, chỉ khác
    epoch/lịch LR) 0.778 → 0.780; 321–512 token 0.692 → 0.703; > 512 token 0.639 → 0.631. Model fold 0 sau 3 epoch: AUC
    0.845 trên dữ liệu train của nó, 0.745 trên fold giữ lại → epoch thứ 3 chỉ học thuộc thêm. (Artifacts Devign hiện tại
    vẫn là fine-tune 512/3 epoch cho đến khi chạy lại `b7` với cấu hình chốt.)
18. **Stacking view fine-tune (2026-11-26)**, `prefilter.yaml: model.fusion: stacked`, `prefilter/stacking.py`. Mạng
    multi-view train trên 4 view (token, AST, semantic, numeric; thư mục `base-r{r}`), rồi
    s(x) = c + a·s_mạng(x) + b·logit_finetune(x), hồi quy logistic fit trên **train_dev** (không phải A: calibrator/router,
    không phải B: conformal/θ, không phải test). Lý do: train chung, mạng đặt trọng số 0.73 vào view fine-tune (trông mạnh
    hơn trên train: AUC out-of-fold 0.728 so với test 0.717), trong khi trọng số tuyến tính tốt nhất ≈ 0.45–0.5; AUC prefilter
    5 view rơi 0.725 → 0.719. Kiểm tra (trọng số fit trên B): A 0.732 → 0.740, test 0.719 → 0.730. Cách cũ = ablation
    **A14** (`main-r1`, `experiments.yaml: ablations.joint_reps`). A1 "bỏ finetune" = mạng cơ sở của rep 1 (không stack).

## 6. Trạng thái (2026-11-22)

- `./run_all.sh smoke` và `./run_<dataset>.sh <phase> --smoke` (tập con nhỏ, LLM mock) chạy hết P0 → P7 + độ bền.
- Xem trạng thái thật bất cứ lúc nào: `./run_<dataset>.sh check` (hoặc `python -m v2.scripts.run_status`).
- 2026-11-23: artifacts xoá sạch lần nữa (sau khi thêm view `finetune` và đổi `reasoning: medium`); Devign đang chạy
  lại full trong tmux session `devign`. Dataset nào chạy thì dataset đó được xây.
- `dev` tự xây đúng phần nó cần (records PrimeVul + Joern cho 300 hàm dev set), nên có thể bắt đầu bằng bất kỳ dataset nào.

## 7. Việc cần bạn làm

1. Điền `OPENROUTER_API_KEY` vào `.env`, đặt trần 100 USD, rồi `python -m v2.scripts.run_p4_llm probe`.
2. Vòng prompt: `./run_all.sh dev` → đọc `v2/artifacts/primevul/llm/dev_report.json`, sửa `v2/llm/prompts/*.j2` nếu
   cần, chạy lại (chỉ các lời gọi mới tốn tiền) → `./run_all.sh freeze`. Sau khi đóng băng, sửa prompt sẽ bị chặn ở A-pool/test.
3. P5 bước 4: đánh dấu 30 trường hợp trong `v2/artifacts/{ds}/llm/manual_review_30.md`.
