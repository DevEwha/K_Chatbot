from huggingface_hub import snapshot_download

# 1. 저장할 모델 ID와 토큰 설정
model_id = "rinarina0429/7b_results"

# 2. 모델 다운로드 및 저장
# local_dir: 저장할 폴더 경로
try:
    path = snapshot_download(
        repo_id=model_id,
        # local_dir=f"/acpl-ssd30/{model_id}",
        local_dir="../models/7b_results",
        local_dir_use_symlinks=False,  # 심볼릭 링크 대신 실제 파일을 다운로드
    )
    print(f"모델 다운로드가 완료되었습니다. 저장 경로: {path}")
except Exception as e:
    print(f"오류 발생: {e}")