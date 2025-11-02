# ITALIC: An Italian Intent Classification Dataset

This folder contains the ITALIC dataset containing 16,521 audio recordings collected by 70 different volunteers. The dataset is composed of:

- `recordings`: a folder containing the audio recordings in `.wav` format. It contains all the recordings composing the data collection.
- `[CONFIG_NAME]_[SPLIT_NAME].json`: the files containing metadata used for generating the configuration proposed in the paper and their corresponding splits:
    - `[CONFIG_NAME]` is the name of the configuration, e.g. `massive`, `hard_noisy`, or `hard_speaker`. For the description of the configurations, please refer to the paper.
    - `[SPLIT_NAME]` is the name of the split, e.g. `train`, `validation`, or `test`. Each split is different for each configuration.

The metadata files are in JSON format, one sample per line. Each sample is a JSON object with the following fields:

- `id`: the unique identifier of the sample.
- `age`: the age of the speaker (self-reported)
- `gender`: the gender of the speaker (self-reported)
- `region`: the region of origin of the speaker (self-reported)
- `nationality`: the nationality of the speaker (self-reported)
- `lisp`: the presence of a lisp in the speaker (self-reported)
- `education`: the education level of the speaker (self-reported)
- `speaker_id`: the unique identifier of the speaker
- `environment`: the environment in which the recording was made (self-reported)
- `device`: the device used for recording (self-reported)
- `scenario`, `field`, `intent`: the information parsed from [massive](https://github.com/alexa/massive) annotations and accompanying metadata.
- `utt`: the utterance to be spoken by the speaker. This information is also taken from [massive](https://github.com/alexa/massive).

## License

The ITALIC dataset is released under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). 
If you use the dataset in your work, please cite the ITALIC paper.