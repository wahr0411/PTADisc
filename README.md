# PTADisc

PTADisc is sourced from  [PTA](https://pintia.cn/) (Programming Teaching Assistant), a learner-based online system  for universities and society developed by Hangzhou PAT Education Technology Co., Ltd.

 PTADisc is a diverse, immense, student-centered dataset that emphasizes its sufficient cross-course information for personalized learning. It includes $68$ courses, $1,527,974$ students, $2,506$ concepts, $220,649$ problems, and over $680$ million student response logs.

We illustrate the characteristics of PTADisc from the following four aspects:

- Diverse: PTADisc contains rich concept-related information and fine-grained records of a large number of student behaviors.
- Immense: PTADisc is the largest dataset in personalized learning. It also includes different courses of different data scales which offer options for various studies.
- Student-centered: In PTADisc, problems and knowledge concepts are validated based on student response logs, leading to better consistency and well-maintainability in terms of diagnostic tasks.
- Cross-course: PTADisc covers a significant amount of students taking multiple courses as students are likely to take a series of courses according to their training program.



### Data repository structure

Data can be downloaded [here](http://121.36.215.35/).

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
| datasets_for_CCLMF      | *Java Programming* datasets and pretrained NCD model on *Python Programming*, which is used to conduct CCLMF experiments. |
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

response_log.csv fields description:

- `submission_id`: the ID of one commit record of a problem set.
- `problem_type`: 1 represents true or false, 2 represents single-choice, 3 represents multiple-choice, and 4 represents fill-in-the-blank.
- `score`: the scoring ratio the student get on  this problem.
- `status`: auxiliary field to calculate student's score on this problem, contains *ACCEPTED*, *WRONG_ANSWER*, *PARTIAL_ACCEPTED*, etc.

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

Each course contains a response_log.csv with fields: `submission_id`, `user_id`,  `create_at`, `language` ,  `score`,  `problem_set_problem_id`,  `problem_id`,  `skill_id`,  `code`,  `response`, `time_consume`, `memory_consume`. Some  of the fields description:

- `submission_id`: the same as in non_programming data.
- `language`: the programming code 
- `score`: the same as in non_programming data.
- `code`: the code submitted by the student.
- `response`: the same as `status`in non_programming data.
- `time_consume`: time consumed by the submitted code.
- `memory_consume`: memory consumed by the submitted code.

##### Logs Example

| submission_id       | user_id                                                      | create_at           | language | score | problem_set_problem_id | problem_id          | skill_id | code                                                         | response     | time_consume | memory_consume |
| ------------------- | ------------------------------------------------------------ | ------------------- | -------- | ----- | ---------------------- | ------------------- | -------- | ------------------------------------------------------------ | ------------ | ------------ | -------------- |
| 1523963863660281856 | 82e2f8cab241392f2ffae8614297ed854b6113256c5c8d31cf2d14740a68706b | 2022-05-10 17:51:13 | Java     | 0.0   | 1523908485467762710    | 1013962033606774784 | 201      | import java.util.*;\npublic class Main {\n public static void  main(String[] args) {\n Scanner sc = new Scanner(System.in);\n int n =  sc.nextInt();\n List<Integer> list = new ArrayList<Integer>();\n  for (int i = 0; i < n; i++) {\n list.add(sc.nextInt());\n }\n  list.sort(Comparator.naturalOrder());\n  System.out.println(list.get(list.size()-2));\n }\n} | WRONG_ANSWER | 125          | 18404          |
| 1523963973706231808 | 82e2f8cab241392f2ffae8614297ed854b6113256c5c8d31cf2d14740a68706b | 2022-05-10 17:51:39 | Java     | 0.0   | 1523908485467762710    | 1013962033606774784 | 201      | import java.util.*;\npublic class Main {\n public static void  main(String[] args) {\n Scanner sc = new Scanner(System.in);\n int n =  sc.nextInt();\n List<Integer> list = new ArrayList<Integer>();\n  for (int i = 0; i < n; i++) {\n list.add(sc.nextInt());\n }\n  list.sort(Comparator.naturalOrder());\n  System.out.print(list.get(list.size()-2));\n }\n} | WRONG_ANSWER | 113          | 15232          |



### Statistics


![statistics](./img/statistics.png)

Detailed statistics of each course's students, problems, concepts, non_programming_logs and programming_logs can be found in [statistics.csv](https://github.com/wahr0411/PTADisc/blob/main/statistics.xlsx).





### CCLMF

Details can be found in [README.md](https://github.com/wahr0411/PTADisc/blob/main/CCLMF/README.md).
