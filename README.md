# PTADisc

PTADisc is sourced from  [PTA](https://pintia.cn/) (Programming Teaching Assistant), a learner-based online system  for universities and society developed by Hangzhou PAT Education Technology Co., Ltd.

 PTADisc is a diverse, immense, student-centered dataset that emphasizes its sufficient cross-course information for personalized learning. It includes $68$ courses, $1,527,974$ students, $2,506$ concepts, $220,649$ problems, and over $680$ million student response logs.

We illustrate the characteristics of PTADisc from the following four aspects:

- Diverse: PTADisc contains rich concept-related information and fine-grained records of a large number of student behaviors.
- Immense: PTADisc is the largest dataset in personalized learning. It also includes different courses of different data scales which offer options for various studies.
- Student-centered: In PTADisc, problems and knowledge concepts are validated based on student response logs, leading to better consistency and well-maintainability in terms of diagnostic tasks.
- Cross-course: PTADisc covers a significant amount of students taking multiple courses as students are likely to take a series of courses according to their training program.



### Data repository structure

Data can be downloaded [here](http://124.70.199.175/).

The directory is organized as:

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

| Leaf directory          | Content                                                      |
| ----------------------- | ------------------------------------------------------------ |
| scripts                 | Scripts which can conduct index mapping on structured data to generate final datasets for CD and KT. |
| global_data             | Global_data contains problem bank of all courses, including dictionaries of problem_to_difficulty, problem_to_knowledge, problem_to_reference_count, psp_to_full_score, psp_to_problem saves as .json files. |
| non_programming_data    | Intermediate structured data of $67$ courses, containing student list, concept list, problem list, Q-matrix file and processed response logs. |
| programming_data        | Intermediate structured data of $46$ courses, containing  processed response logs. |
| baseline_dataset_for_CD | $4$ selected courses' datasets for cognitive diagnosis.      |
| baseline_dataset_for_KT | $4$ selected courses' datasets for knowledge tracing.        |
| cross_course_datasets   | datasets of $5$ courses which are simultaneously taken by $29,454$ students. |
| datasets_for_CCLMF      | *Python Programming* and *Java Programming* datasets, which is used to conduct CCLMF experiments. |
| programming_datasets    | $4$ selected courses' datasets for knowledge tracing.        |



### Detailed information of structured data

#### Non-programming data

##### Description

Each course contains the following files:

| Filename              | Description                                                  |
| --------------------- | ------------------------------------------------------------ |
| info/info.json        | Record the number of the course's concepts, students, problems and logs. |
| info/concept_list.txt | Record the course's concept IDs.                             |
| info/student_list.txt | Record the course's student IDs.                             |
| info/problem_list.txt | Record the course's problem IDs.                             |
| problem_info.csv      | Record problems' `problem_set_problem_id`, `problem_id`, `knowledge_id`, `full_score` in the course. |
| response_log.csv      | Record logs' `submission_id`, `user_id`, `create_at`, `problem_type`, `score`, `problem_set_id`, `problem_set_problem_id`, `status` in the course. |

response_log.csv fields:

- `submission_id`: the 
- `problem_type`, 
- `score`,  
- `status`

##### Logs Example

| submission_id | user_id                                                      | create_at        | problem_type | score | problem_set_id | problem_set_problem_id | status       |
| ------------- | ------------------------------------------------------------ | ---------------- | ------------ | ----- | -------------- | ---------------------- | ------------ |
| 178060        | 0fa9733a132497dd515d426df206b67106364354b579231879e3c8f70630e431 | 2015/10/13 16:47 | 2            | 2     | 139            | 1746                   | ACCEPTED     |
| 178060        | 0fa9733a132497dd515d426df206b67106364354b579231879e3c8f70630e431 | 2015/10/13 16:47 | 2            | 0     | 139            | 1747                   | WRONG_ANSWER |
| 178060        | 0fa9733a132497dd515d426df206b67106364354b579231879e3c8f70630e431 | 2015/10/13 16:47 | 2            | 0     | 139            | 1748                   | WRONG_ANSWER |
| 178060        | 0fa9733a132497dd515d426df206b67106364354b579231879e3c8f70630e431 | 2015/10/13 16:47 | 2            | 2     | 139            | 1749                   | ACCEPTED     |
| 178060        | 0fa9733a132497dd515d426df206b67106364354b579231879e3c8f70630e431 | 2015/10/13 16:47 | 2            | 2     | 139            | 1750                   | ACCEPTED     |



#### Programming data

##### Description

Each course contains a data.csv with fields: 

- `submission_id`, 
- `problem_type`, 
- `score` , 
- `problem_set_id`, 
- `problem_set_problem_id`, 
- `status`, 
- `problem_id`, 
- `reference_count`, 
- `skill_id`, 
- `difficulty`.

##### Logs Example

| submission_id       | user_id                                                      | create_at           | problem_type | score | problem_set_id      | problem_set_problem_id | status | problem_id          | reference_count | skill_id | difficulty |
| ------------------- | ------------------------------------------------------------ | ------------------- | ------------ | ----- | ------------------- | ---------------------- | ------ | ------------------- | --------------- | -------- | ---------- |
| 1499254523189440512 | 0007dadd9dab752acf8a1363a7cccf292b154229df4b35156651fe48ce5faffc | 2022-03-03 13:25:07 | 1            | 1     | 1497371829860438016 | 1497372209131987013    | 1      | 1407160646878199808 | 89              | 598      | 1          |
| 1499254523189440512 | 0007dadd9dab752acf8a1363a7cccf292b154229df4b35156651fe48ce5faffc | 2022-03-03 13:25:07 | 1            | 1     | 1497371829860438016 | 1497372209127792778    | 1      | 1242959338549944320 | 383             | 584      | 1          |
| 1499254523189440512 | 0007dadd9dab752acf8a1363a7cccf292b154229df4b35156651fe48ce5faffc | 2022-03-03 13:25:07 | 1            | 1     | 1497371829860438016 | 1497372209131987123    | 1      | 1219454085610201088 | 87              | 654      | 1          |
| 1499254523189440512 | 0007dadd9dab752acf8a1363a7cccf292b154229df4b35156651fe48ce5faffc | 2022-03-03 13:25:07 | 1            | 1     | 1497371829860438016 | 1497372209127792668    | 1      | 1399889183840706560 | 81              | 576      | 1          |
| 1499254523189440512 | 0007dadd9dab752acf8a1363a7cccf292b154229df4b35156651fe48ce5faffc | 2022-03-03 13:25:07 | 1            | 0     | 1497371829860438016 | 1497372209131987069    | 0      | 1402820680818925568 | 52              | 616      | 1          |



### Statistics


![statistics](./img/statistics.png)

Detailed statistics of each course's students, problems, concepts, non_programming_logs and programming_logs can be found in [statistics.csv](https://github.com/wahr0411/PTADisc/blob/main/statistics.xlsx).

