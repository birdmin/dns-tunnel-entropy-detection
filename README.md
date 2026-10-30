# 김경민 · 자기소개 사이트

실천하지 못한 일을 솔직하게 적고, 방법을 바꿔 다시 해 보는 사람.

로그인 없이 열리는 정적 사이트입니다. 첫 화면, 이야기, 숫자, 대표작, 이력서 입구로 이루어져 있습니다.

- 사이트: https://birdmin.github.io/birdmin-portfolio/
- 연락: birdmin9210@gmail.com
- 대표작(논문): https://github.com/birdmin/dns-tunnel-entropy-detection
- 대표작(앱): https://birdmin.github.io/dns-tunnel-explorer/

## 구성

```
index.html                       사이트 전체 (CSS 포함, 한 파일)
docs/김경민_이력서.pdf            이력서 (1쪽)
docs/김경민_지원서류.pdf          이력서·자기소개서·경력기술서 (3쪽)
README.md                        이 문서
```

## 배포 방법

1. 저장소 `birdmin-portfolio`를 Public으로 만들고 위 파일을 main 브랜치에 올립니다.
2. Settings → Pages → Source를 `Deploy from a branch`, 브랜치를 `main`, 폴더를 `/ (root)`로 저장합니다.
3. 1~2분 뒤 https://birdmin.github.io/birdmin-portfolio/ 가 열립니다. 시크릿 창에서 로그인 없이 열리는지 확인합니다.

## 새로 쓰는 방법

숫자 칸과 능력별 기록은 별도 장치(`site-updater`)가 새 기록에서 다시 만든 `index.html`로 바뀝니다. 장치를 돌린 뒤 나온 `out/index.html`을 이 저장소의 `index.html`로 교체해 올리세요.
