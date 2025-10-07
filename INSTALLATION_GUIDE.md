# 멀티 GPU 버전 설치 및 업그레이드 가이드

## 🚀 빠른 시작

### 새로 설치하는 경우

```bash
# 1. 저장소 클론
git clone https://github.com/Jolrin-Saram/auto_label_trainer.git
cd auto_label_trainer

# 2. 멀티 GPU 개선 파일로 교체
# (이 폴더의 파일들을 복사)

# 3. 가상 환경 설정
install.bat

# 4. 프로그램 실행
START_TRAINER.bat
```

### 기존 버전 업그레이드

현재 GitHub 저장소의 파일을 아래 개선된 파일로 교체하세요:

#### 교체할 파일 (필수)

1. **process_and_train.py**
   - 멀티 GPU 학습 지원 추가
   - Device string parsing
   - DDP 자동 활성화

2. **auto_labeler.py**
   - GPU 지정 기능 추가
   - 더 많은 이미지 포맷 지원

3. **run_training_ui.py**
   - GPU 선택 드롭다운 UI
   - 자동 GPU 감지
   - 설정 저장/로드 개선

#### 추가 파일 (권장)

4. **README_MULTIGPU.md** - 멀티 GPU 사용 가이드
5. **MULTIGPU_UPGRADE_SUMMARY.md** - 개선사항 요약
6. **test_multigpu.py** - 시스템 테스트 스크립트
7. **INSTALLATION_GUIDE.md** - 이 파일

---

## 📋 상세 업그레이드 절차

### Step 1: 백업

```bash
# 기존 파일 백업
cp process_and_train.py process_and_train.py.backup
cp auto_labeler.py auto_labeler.py.backup
cp run_training_ui.py run_training_ui.py.backup
```

### Step 2: 파일 교체

```bash
# 개선된 파일로 교체
# Windows
copy /Y process_and_train_multigpu.py process_and_train.py
copy /Y auto_labeler_multigpu.py auto_labeler.py
copy /Y run_training_ui.py run_training_ui.py

# Linux/Mac
cp -f process_and_train_multigpu.py process_and_train.py
cp -f auto_labeler_multigpu.py auto_labeler.py
```

### Step 3: 테스트

```bash
# GPU 감지 테스트
python test_multigpu.py

# 프로그램 실행 테스트
python run_training_ui.py
```

### Step 4: 설정 확인

프로그램 실행 후:
1. "Training" 탭 열기
2. "Device" 드롭다운 확인
3. GPU 목록이 표시되는지 확인

---

## 🔧 시스템 요구사항 확인

### CUDA 설치 확인

```bash
# CUDA 버전 확인
nvidia-smi

# PyTorch CUDA 지원 확인
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.device_count())"
```

예상 출력:
```
True
4
```

### 필수 패키지 확인

```bash
pip list | grep -E "torch|ultralytics"
```

예상 출력:
```
torch              2.5.1+cu121
torchaudio         2.5.1+cu121
torchvision        0.20.1+cu121
ultralytics        8.3.204
```

---

## 🎯 기능 테스트 체크리스트

### UI 테스트
- [ ] 프로그램이 정상 실행되는가?
- [ ] GPU 개수가 올바르게 표시되는가?
- [ ] Device 드롭다운에 옵션이 있는가?
  - [ ] "Auto (All X GPUs)"
  - [ ] "GPU 0: [모델명]"
  - [ ] "GPU 1: [모델명]"
  - [ ] "Custom..."

### 학습 테스트 (간단)
- [ ] 소규모 데이터셋 준비 (이미지 10장)
- [ ] Device: "Auto" 선택
- [ ] Epochs: 3
- [ ] Batch Size: 8
- [ ] Start Training 클릭
- [ ] 로그에 "MULTI-GPU TRAINING ENABLED" 표시

### Auto-Labeling 테스트
- [ ] 모델 파일 (.pt) 선택
- [ ] 이미지 폴더 선택
- [ ] Start Auto-Labeling 클릭
- [ ] Progress Bar 동작 확인
- [ ] 라벨 파일 (.txt) 생성 확인

---

## 🐛 문제 해결

### 문제 1: "No GPU detected"

**증상:**
```
Device 드롭다운에 "CPU (No GPU detected)" 만 표시
```

**해결책:**
```bash
# NVIDIA 드라이버 재설치
# https://www.nvidia.com/Download/index.aspx

# PyTorch CUDA 버전 재설치
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 문제 2: "DDP 초기화 실패"

**증상:**
```
RuntimeError: Failed to initialize distributed backend
```

**해결책:**
1. 방화벽에서 Python 허용
2. 단일 GPU로 먼저 테스트
3. 가상 환경 재생성

### 문제 3: "CUDA Out of Memory"

**증상:**
```
RuntimeError: CUDA out of memory
```

**해결책:**
```
배치 크기 줄이기:
64 → 32 → 16 → 8

또는 더 작은 모델:
yolov8x → yolov8l → yolov8m
```

### 문제 4: "ImportError: cannot import parse_device_string"

**증상:**
```
test_multigpu.py 실행 시 import 오류
```

**해결책:**
```bash
# process_and_train.py가 현재 디렉토리에 있는지 확인
ls process_and_train.py

# 파일이 개선 버전인지 확인 (parse_device_string 함수 있어야 함)
grep "def parse_device_string" process_and_train.py
```

---

## 📊 성능 비교 방법

### 벤치마크 실행

```bash
# 1 GPU 테스트
Device: "GPU 0"
Batch: 16
Epochs: 10

# 4 GPU 테스트
Device: "Auto (All 4 GPUs)"
Batch: 64
Epochs: 10

# 시간 비교
# 로그에서 "Epoch X/10" 시간 측정
```

### 예상 결과

| GPU | Batch | Epoch 시간 | 속도 |
|-----|-------|-----------|------|
| 1   | 16    | 10분      | 1.0x |
| 4   | 64    | 3분       | 3.3x |

---

## 🔄 롤백 (이전 버전으로 되돌리기)

문제 발생 시:

```bash
# 백업 파일로 복원
cp process_and_train.py.backup process_and_train.py
cp auto_labeler.py.backup auto_labeler.py
cp run_training_ui.py.backup run_training_ui.py

# 또는 GitHub에서 다시 받기
git clone https://github.com/Jolrin-Saram/auto_label_trainer.git original_version
```

---

## 📞 지원

문제가 해결되지 않으면:

1. **GitHub Issues**: https://github.com/Jolrin-Saram/auto_label_trainer/issues
2. **로그 파일 첨부**: `log_YYYYMMDD_HHMMSS.txt`
3. **시스템 정보 제공**:
   ```bash
   nvidia-smi
   python --version
   pip list | grep torch
   ```

---

## ✅ 설치 완료 확인

모든 것이 정상이면:

```
✅ GPU가 감지됨
✅ Device 드롭다운에 GPU 목록 표시
✅ 테스트 학습 성공
✅ 로그에 "MULTI-GPU TRAINING ENABLED" 표시
✅ Auto-Labeling 작동
```

**축하합니다! 멀티 GPU 학습 준비 완료!** 🎉

---

## 📚 다음 단계

1. [README_MULTIGPU.md](README_MULTIGPU.md) - 사용 가이드
2. [MULTIGPU_UPGRADE_SUMMARY.md](MULTIGPU_UPGRADE_SUMMARY.md) - 기술 세부사항
3. 본격 학습 시작!
