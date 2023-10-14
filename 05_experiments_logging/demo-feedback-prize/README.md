[Competition url](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/overview)
# Description
## Goal of the Competition

Цель этоой соревы — оценить знание языка учащихся, изучающих английский язык на уровне 8-12 классов (English Language Learners, ELLs). Использование набора данных с эссе, написанных ELLs, поможет разработать модели оценки владения языком.

Ваша работа поможет ELLs получить более точную обратную связь по их развитию языка и ускорит цикл оценивания для учителей. Эти результаты могут позволить ELLs получать более подходящие задачи для обучения, которые помогут им улучшить свои знания английского языка.


## Context
Письмо - это фундаментальный навык. К сожалению, мало кому из студентов удается его отточить, часто из-за того, что письменные задания редко ставятся в школе. Быстро растущее количество студентов, изучающих английский как второй язык, особенно пострадали от недостатка практики. Хотя автоматизированные инструменты обратной связи облегчают учителям задачу поставить больше письменных заданий, они не предназначены для работы с ELLs.

Существующие инструменты не могут предоставить обратную связь, основанную на знании языка ученика, что приводит к итоговой оценке, которая может быть предвзятой против обучающегося. Наука мождет улучшить автоматизированные инструменты обратной связи, чтобы лучше поддерживать потребности учеников.


## Evaluation

Submissions are scored using MCRMSE, mean columnwise root mean squared error:

$$ MCRMSE = \frac{1}{N_t} \sum_{j=1}^{N_t} \sqrt{\frac{1}{n} \sum_{i=1}^n (y_{ij} - \hat{y}_{ij})^2} $$

where $N_t$ is the number of scored ground truth target columns, and $y$ and $\hat{y}$ are the actual and predicted values, respectively.



## Dataset Description

The dataset presented here (the ELLIPSE corpus) comprises argumentative essays written by 8th-12th grade English Language Learners (ELLs). The essays have been scored according to six analytic measures: **cohesion** (согласованность), **syntax** (синтаксис), **vocabulary** (словарный запас), **phraseology** (фразеология), **grammar** (грамматика), and **conventions** (конвенции).


Each measure represents a component of proficiency in essay writing, with greater scores corresponding to greater proficiency in that measure. The scores range from $1.0$ to $5.0$ in increments of $0.5$. Your task is to predict the score of each of the six measures for the essays given in the test set.


## Downloading 

```bash
cd data/
kaggle competitions download -c feedback-prize-english-language-learning
unzip feedback-prize-english-language-learning.zip 
rm -f feedback-prize-english-language-learning.zip sample_submission.csv test.csv 
```

