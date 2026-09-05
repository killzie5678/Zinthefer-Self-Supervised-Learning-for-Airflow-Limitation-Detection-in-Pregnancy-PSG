# Zinthefer-Self-Supervised-Learning-for-Airflow-Limitation-Detection-in-Pregnancy-PSG

Pregnancy can alter respiratory physiology and increase susceptibility to sleep-disordered breathing and airflow limitation. Full-night polysomnography provides rich information about these respiratory dynamics, but expert annotation is expensive. In our dataset, each patient has approximately eight hours of multichannel PSG, while only 559 breaths from 52 patients have consensus labels.
The central goal is to exploit the large amount of unlabeled PSG signal using self-supervised learning (SSL), then adapt the learned representation to the small breath-level labeled dataset.
The project has five main goals:
Build a PSG processing pipeline that extracts, aligns, resamples, and normalizes the respiratory channels from the original recordings.
Establish supervised baselines for breath-level classification using the 559 labeled breaths.
Learn respiratory representations from unlabeled PSG using SSL.
Fine-tune the pretrained encoder to classify individual breaths as normal or abnormal airflow-limited breaths.
Determine whether SSL improves label efficiency, particularly when only a small fraction of the 559 labels is available.
The primary research question is:
Can self-supervised pretraining on unlabeled pregnancy PSG improve breath-level detection of abnormal airflow limitation when expert-labeled data is limited?

Project goals and outline by Ioanna Gemou
