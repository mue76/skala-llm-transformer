"""김유정 단편 한국어 corpus 생성 스크립트.

위키문헌(ko.wikisource.org)에서 김유정 단편을 내려받아 정제·병합하여
minGPT 문자 시퀀스 실습용 input_kr.txt를 만듭니다.

- 저작권: 저자 사망(1937) 후 70년 경과, 퍼블릭 도메인
- 출처: https://ko.wikisource.org/wiki/저자:김유정
- 사용법: python prepare_kim_yujeong_corpus.py
  (실행 위치에 input_kr.txt 생성 → GitHub 저장소에 업로드하여 사용)
"""

import re
import json
import unicodedata
import urllib.request
import urllib.parse
from collections import Counter

WORKS = [
    "동백꽃",
    "봄봄",
    "산골 나그네",
    "소낙비",
    "금 따는 콩밭",
    "만무방",
    "땡볕",
    "따라지",
]

API = "https://ko.wikisource.org/w/api.php"
SEP = "\n\n<|작품끝|>\n\n"   # 작품 경계 구분자 (생성 결과 관찰 포인트)
OUTPUT = "input_kr.txt"


def fetch_work(title):
    """위키문헌 API에서 작품 본문(plain text)을 가져온다."""
    params = urllib.parse.urlencode({
        "action": "query", "prop": "extracts",
        "explaintext": 1, "redirects": 1,
        "format": "json", "titles": title,
    })
    req = urllib.request.Request(
        f"{API}?{params}",
        headers={"User-Agent": "SKALA-edu-corpus/1.0"}
    )
    with urllib.request.urlopen(req) as r:
        pages = json.loads(r.read())["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("extract", "")


def clean(text):
    """위키 텍스트 정제."""
    # 1) NFC 정규화 (필수: 자모 분해형이면 vocab 폭증)
    text = unicodedata.normalize("NFC", text)
    # 2) 라이선스 안내문 이후 제거
    for marker in ["이 저작물은", "== 라이선스 =="]:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
    # 3) 위키 섹션 제목 제거
    text = re.sub(r"^=+ .* =+$", "", text, flags=re.M)
    # 4) 각주 번호, 특수 공백 정리
    text = re.sub(r"\[\d+\]", "", text)
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    # 5) 3줄 이상 빈 줄 → 2줄
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def report(text):
    """corpus 통계 출력 (품질 확인용)."""
    chars = sorted(set(text))
    vocab_size = len(chars)
    print(f"\n전체 문자 수: {len(text):,}")
    print(f"고유 문자 수(vocab): {vocab_size:,}")

    cnt = Counter(text)
    rare = [c for c in chars if cnt[c] == 1]
    print(f"1회만 등장하는 문자: {len(rare)}개")
    print("예시:", "".join(rare[:40]))

    # gpt-mini 파라미터 증가 추정 (n_embd=192, 임베딩+출력층)
    added = vocab_size * 192 * 2
    base = 65 * 192 * 2
    print(f"vocab에 의한 파라미터: {added:,} "
          f"(Shakespeare 대비 +{added - base:,})")


def main():
    parts = []
    for title in WORKS:
        try:
            cleaned = clean(fetch_work(title))
            if len(cleaned) < 1000:
                print(f"⚠ {title}: {len(cleaned)}자 — 본문이 짧음, 제목 확인 필요")
            else:
                print(f"✔ {title}: {len(cleaned):,}자")
                parts.append(cleaned)
        except Exception as e:
            print(f"✘ {title}: 실패 ({e})")

    text = SEP.join(parts)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(text)

    report(text)
    print(f"\n저장 완료: {OUTPUT}")


if __name__ == "__main__":
    main()
