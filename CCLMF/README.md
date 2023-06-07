## Cross-Course Learner Modeling Framework

CCLMF leverages meta-learning to establish connections for each student between courses. The experimental results on PTADisc verify the effectiveness of CCLMF in cold-start scenarios.



### Dependencies

- Python

- Pytorch

- EduCDM

- Numpy

- Pandas

  

### Directory Structure

```
CCLMF
├── README.md
├── data
│   └── README.md
├── logs
├── model
│   └── README.md
├── main.py
├── model.py
└── utils.py
```



### Run

Before training CC_NCD on java-30, download the dataset to ./data/ and the pretrained model to ./model/.

Then run main.py:

```
python main.py
```

