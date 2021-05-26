## 2021.05.26

#### [이미지 전처리 CV 연구하기](./1_cv_research.ipynb)

- `show_images`: 이미지가 있는 폴더 내의 이미지 시각화 함수
  - `divider`로 한 줄에 몇 장의 이미지를 시각화할지 지정 가능
- `load_images`: 이미지 폴더 내의 이미지를 불러와 np.array로 변환하여 리스트로 반환
- `show_transformed_images`: 이미지 시각화 및 영상처리 알고리즘 적용 함수
- `rgb_to_gray`: rgb에서 grayscale로 변환 함수
- `morph_ellipse`: 모폴로지 침식 알고리즘 함수
- `normalize`: 정규화 함수
- `threshold`: 오츠 알고리즘 적용하는 함수이며, `adaptive`로 adaptive threshold 지정 가능
- `clahe`: CLAHE 변환 적용하는 함수

#### References

- [OpenCV - 19. 모폴로지(Morphology) 연산 (침식, 팽창, 열림, 닫힘, 그레디언트, 탑햇, 블랙햇)](https://bkshin.tistory.com/entry/OpenCV-19-%EB%AA%A8%ED%8F%B4%EB%A1%9C%EC%A7%80Morphology-%EC%97%B0%EC%82%B0-%EC%B9%A8%EC%8B%9D-%ED%8C%BD%EC%B0%BD-%EC%97%B4%EB%A6%BC-%EB%8B%AB%ED%9E%98-%EA%B7%B8%EB%A0%88%EB%94%94%EC%96%B8%ED%8A%B8-%ED%83%91%ED%96%87-%EB%B8%94%EB%9E%99%ED%96%87)
- [OpenCV - 8. 스레시홀딩(Thresholding), 오츠의 알고리즘(Otsu's Method)](https://bkshin.tistory.com/entry/OpenCV-8-%EC%8A%A4%EB%A0%88%EC%8B%9C%ED%99%80%EB%94%A9Thresholding?category=1148027)
- [OpenCV - 10. 히스토그램과 정규화(Normalize), 평탄화(Equalization), CLAHE](https://bkshin.tistory.com/entry/OpenCV-10-%ED%9E%88%EC%8A%A4%ED%86%A0%EA%B7%B8%EB%9E%A8?category=1148027)
