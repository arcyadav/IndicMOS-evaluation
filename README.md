# TTS Metric Evaluation

This repository contains the work done to explore objective evaluation metrics for
Text-to-Speech (TTS) audio quality and to check whether an automatic MOS
prediction metric agrees with human perception for the audio samples used in a
voice questionnaire.

The main goal was to identify metrics that can help compare TTS systems without
always depending on a full listening test. After reviewing recent TTS evaluation
papers, the practical focus became MOS prediction because the available audio
samples were standalone generated audios and did not have matching reference
recordings.

## Background

TTS systems are usually evaluated with subjective listening tests such as MOS
or MUSHRA. These are reliable, but they are slow, expensive, and difficult to
repeat during model development. Objective metrics are useful because they can
quickly score generated audio and help shortlist better TTS outputs.

The metric survey was based mainly on:

- `TTSDS: Text-to-Speech Distribution Score` by Christoph Minixhofer, Ondrej
  Klejch, and Peter Bell.
- `Rethinking MUSHRA: Addressing Modern Challenges in Text-to-Speech Evaluation`.
- `TTSDS2: Resources and Benchmark for Evaluating Human-Quality Text to Speech
  Systems` by Christoph Minixhofer, Ondrej Klejch, and Peter Bell.
- `IndicMOS: Multilingual MOS Prediction for 7 Indian Languages` by Sathvik
  Udupa, Soumi Maiti, and Prasanta Kumar Ghosh.

The metric notes were first organized visually in an SVG file. For the actual
audio evaluation task, MOS prediction was selected because reference-free
metrics were needed.

## What This Project Does

1. Runs IndicMOS on TTS audio files to predict MOS scores.
2. Stores per-file MOS scores and folder-level summaries.
3. Uses a Google Form response sheet containing human MOS ratings.
4. Compares human MOS with predicted MOS using correlation and error metrics.
5. Saves the correlation results as a CSV file in the `results/` folder.

The correlation step reports:

- Pearson correlation
- Spearman rank correlation
- Kendall rank correlation
- MAE
- RMSE

## Repository Structure

```text
.
|-- evaluate_correlation.py      # Compare human MOS and predicted MOS
|-- run_indicmos.py              # Score a folder of audio files with IndicMOS
|-- run_single_indicmos.py       # Score one audio file with IndicMOS
|-- forms/
|   |-- Responses.xlsx           # Google Form response sheet
|   `-- form_for_human_ratings.pdf
|-- results/
|   |-- indicmos_file_scores.csv
|   |-- indicmos_folder_summary.csv
|   |-- indicmos_report.xlsx
|   `-- mos_correlation_results.csv
|-- requirements.txt
`-- README.md
```

## Setup

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

`ffmpeg` must also be available on the system path because the audio loading
script uses it to decode and resample audio to 16 kHz mono.

The IndicMOS code/model files are expected to be available in an `IndicMOS/`
folder at the project root, because `run_single_indicmos.py` imports
`infer_indicmos.py` from that location.

## Running IndicMOS on One Audio File

```bash
python run_single_indicmos.py path/to/audio.wav
```

For the language-aware IndicMOS model, pass the language id:

```bash
python run_single_indicmos.py path/to/audio.wav --use-langid --langid hi
```

Supported language ids depend on the IndicMOS model mapping. Examples include
`hi`, `te`, `kn`, `mr`, `bn`, and `en`.

## Running IndicMOS on a Folder

```bash
python run_indicmos.py --input-dir path/to/audio_folder
```

By default, the script writes output files under `results/indicmos/`:

- `indicmos_file_scores.csv`
- `indicmos_folder_summary.csv`
- `indicmos_report.xlsx`

You can change the output directory:

```bash
python run_indicmos.py --input-dir path/to/audio_folder --output-dir results/my_run
```

## Evaluating Correlation With Human Ratings

After collecting human MOS ratings in a Google Form and exporting the responses
as an Excel file, run:

```bash
python evaluate_correlation.py path/to/Responses.xlsx
```

The script reads the `MOS` and `Predicted MOS` rows from the sheet, groups the
audio columns by language, and computes correlation/error metrics.

By default, the output CSV is saved here:

```text
results/mos_correlation_results.csv
```

To save somewhere else:

```bash
python evaluate_correlation.py path/to/Responses.xlsx --output-csv path/to/output.csv
```

## Notes

- The language order used by `evaluate_correlation.py` is controlled by the
  `LANGUAGES` list inside the script.
- The current language order is `English`, `Hindi`, `Kannada`, and `Telugu`.
- If the Google Form layout changes, update `LANGUAGES` so it matches the
  left-to-right order of audio blocks in the exported Excel sheet.
- IndicMOS scores are predicted MOS values, so the correlation analysis is used
  to check how well those predictions match the human ratings collected through
  the questionnaire.
