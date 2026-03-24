# Replication Data for: A Sleeping, Recovering Bandit Algorithm for OptimizingRecurring Notifications

## Introduction

This distribution consists of the datasets used in the paper title "A Sleeping, Recovering Bandit Algorithm for Optimizing Recurring Notifications" by Kevin Yancey and Burr Settles, published at KDD 2020.  It contains millions of practice reminder push notifications sent to Duolingo users over a 35 day period, including which template was used, whether the user converted within 2 hours, and other metadata.  It may be used to reproduce the offline experiments described in the KDD paper, or to test other recurring notification optimization algorithms for comparison.  All proprietary or personally identifying information has been removed.

If you use our dataset, please cite our paper:

```
@inproceedings{
  yancey2020sleeping,
  title={A Sleeping, Recovering Bandit Algorithm for Optimizing Recurring Notifications},
  booktitle={Proceedings of the 26th ACM SIGKDD international conference on Knowledge discovery and data mining},
  author={Kevin Yancey and Burr Settles},
  year={2020},
  doi = {10.1145/3394486.3403351},
  url = {https://doi.org/10.1145/3394486.3403351}
}
```

## Files

This distribution includes two datasets:
* `training` - 15 days of training data.  87,665,839 rows.
* `test` - 19 days of test data.  114,486,159 rows.

These are each distributed in multiple parts to keep the file sizes under 2.5 GB.  The files for each part should be unzipped to a single direcctory and read as a single dataset.

This split reflects the original train/test split that was used in the KDD paper.  However, their formats are identical, and they may be recombined and split to suit your needs.

## Format

The files are provided in parquet format, with the following columns:

* `datetime : float` - the date and time that the notification was sent, represented as a float that indicates the number of days since the start datetime of the dataset.  For example, a value of 4.25 is 4 days and 6 hours after the start datetime of the dataset.  These datetimes were all calculated from the UTC timezone, not each user's timezone.
* `ui_language : string` - the ISO 639-1 code of the language the user receives his/her notifications in.
* `eligible_templates : List[string]` - a list of the templates that were eligible for this notification.  Each template is identified by a single-letter code: A-L.
* `history : List[Tuple[string, float]]`- a list of previous templates sent to the user and when they were sent, represented as a float indicating the number of days prior to this notification.  For example, the element in the list `("A": 2.5)` means that template A was sent to the user 2 days and 12 hours prior to this notification.  
* `selected_template : string` - the template that was used for this notification.
* `session_end_completed : boolean` - indicates whether a session was completed by the user within 2 hours of the notification being sent.  This is used as the reward metric: 1 if True, 0 if False.

Parquet files can be processed with popular libraries such as Spark (https://spark.apache.org/), PySpark (https://pypi.org/project/pyspark/), and Pandas (https://pandas.pydata.org/).  For more information on the parquet format, please see (https://parquet.apache.org/).