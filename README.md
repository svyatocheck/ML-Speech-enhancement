
# CNN for speech enhancement (baseline model)

This project is part of my homework on [My first data project](https://ods.ai/tracks/my_first_data_project) course



## Requirements

- librosa
- numpy
- tensorflow

Note: there are a lot of places to improve in my code. I will do it later.


## Testing
[Example 1](https://drive.google.com/drive/folders/1-5wGQ1fpA0poQMsjk5xdNzo99HKL01lp?usp=sharing):

- STOI: 0.7485873492849783
- SDR: 11.094792287825626

![Screenshot from 2023-03-21 12-10-12](https://user-images.githubusercontent.com/63301430/226561548-743503a6-c1c2-42dc-a7f5-a0f7a2a508e5.png)
![Screenshot from 2023-03-21 12-10-23](https://user-images.githubusercontent.com/63301430/226561551-b33b70f7-f699-4879-a481-b6cf86470f31.png)
![Screenshot from 2023-03-21 12-10-28](https://user-images.githubusercontent.com/63301430/226561554-a6e60551-fac5-4cce-a79f-03731b4b90a5.png)

[Example 2](https://drive.google.com/drive/folders/1-5wGQ1fpA0poQMsjk5xdNzo99HKL01lp?usp=sharing):

- STOI: 0.5552426231950904
- SDR: 6.131914852996063

![Screenshot from 2023-03-21 12-15-12](https://user-images.githubusercontent.com/63301430/226562559-49c322a3-66df-4c48-886a-ddcc3c02cc1a.png)
![Screenshot from 2023-03-21 12-15-18](https://user-images.githubusercontent.com/63301430/226562563-de83326d-36c4-46af-aac9-bea6fdec9fee.png)
![Screenshot from 2023-03-21 12-15-23](https://user-images.githubusercontent.com/63301430/226562569-27601b24-8d6d-4be8-bc8c-f0099023370e.png)

## References
- https://paperswithcode.com/paper/a-fully-convolutional-neural-network-for
- https://habr.com/ru/post/668518/
- https://www.youtube.com/watch?v=ZqpSb5p1xQo&list=WL&index=8&t=1072s
- https://www.mathworks.com/help/deeplearning/ug/denoise-speech-using-deep-learning-networks.html
- https://www.kaggle.com/code/danielgraham1997/speech-denoising-analysis#Metrics-Analysis
- https://www.kaggle.com/code/carlolepelaars/bidirectional-lstm-for-audio-labeling-with-keras/notebook
- https://www.kaggle.com/datasets/chrisfilo/urbansound8k
- https://commonvoice.mozilla.org/
