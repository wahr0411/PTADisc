# PTADisc

PTADisc is sourced from  [PTA](https://pintia.cn/) (Programming Teaching Assistant), a learner-based online system  for universities and society developed by Hangzhou PAT Education Technology Co., Ltd. PTADisc is a diverse, immense, student-centered dataset that emphasizes its sufficient cross-course information for personalized learning. It includes $68$ courses, $1,527,974$ students, $2,506$ concepts, $220,649$ problems, and over $680$ million student response logs.

We illustrate the characteristics of PTADisc from the following four aspects:

- Diverse: PTADisc contains rich concept-related information and fine-grained records of a large number of student behaviors.
- Immense: PTADisc is the largest dataset in personalized learning. It also includes different courses of different data scales which offer options for various studies.
- Student-centered: In PTADisc, problems and knowledge concepts are validated based on student response logs, leading to better consistency and well-maintainability in terms of diagnostic tasks.
- Cross-course: it covers a significant amount of students taking multiple courses as students are likely to take a series of courses according to their training program.



### Data

Data can be downloaded [here](http://124.70.199.175/).



```
data_baiteng/
├── scripts
├── structured_data
│   ├── global_data
│   ├── non_programming_data
│   └── programming_data
└── task_specific_dataset
    ├── non_programming_dataset
    │   ├── baseline_dataset_for_CD
    │   ├── baseline_dataset_for_KT
    │   ├── cross_course_datasets
    │   └── datasets_for_CCLMF
    └── programming_datasets
```



### Statistics

TODO: statictis3