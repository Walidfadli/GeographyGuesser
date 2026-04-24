# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A live-assist tool for GeoGuessr. It screenshots the active screen, runs three parallel classifiers against the crop, and opens Google Maps windows on the best candidate locations. The three classifiers are:

1. A Keras CNN that predicts **continent** (6-way softmax).
2. A Keras CNN that predicts **country flag** (one class per GeoGuessr country).
3. A text pipeline: Google Vision OCR → Nominatim geocode on each OCR phrase → keep only addresses whose last comma-segment language matches the Vision-reported `locale` (done via spaCy `language_detector`).

Results from all three are printed; geocoded addresses (if any) are opened in Selenium-controlled Chrome windows pointed at `google.com/maps`.

TensorFlow is pinned at 2.7.0 in the source comments. No `requirements.txt` exists — dependencies are implicit.

## Running

Entry point: `python run_live_new.py` (runs indefinitely; use hotkeys).

Runtime hotkeys (global — `keyboard` module uses OS-level hooks, so the script must run with appropriate privileges on Windows):

- **`s`** — screenshot the middle 60% of a 2560×1440 screen (hard-coded in [run_live_new.py:94](run_live_new.py#L94) and [run_live_new.py:319](run_live_new.py#L319)), then opens a `cv2` window; click-drag to pick a crop (for the OCR phase), release to save to `live_images/imageN.jpg`.
- **`a`** — analyze everything in `live_images/`, print top continents/flags/geocoded addresses, and load each address into one of the pre-spawned Chrome windows (defaults to `max_locations_shown = 1`).
- **`d`** — wipe `live_images/`.

Prerequisites before running:
- `live_images/` must exist (not created by the script).
- Google Vision credentials path must be set at [specified_classification/text_classification/image_with_text_to_country_new.py:13](specified_classification/text_classification/image_with_text_to_country_new.py#L13) (`os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""` is empty by default — fill in before use).
- The two `.h5` model files at `models/continent_classifier_model/` and `models/flag_classifier_model/` must exist (tracked in the repo).
- `data/images_sorted_by_continent/*` and `data/images_country_flags_modified/*` subfolders must exist — at inference time their *folder names* are read via `glob` to label the model's output indices ([run_live_new.py:126](run_live_new.py#L126), [run_live_new.py:157](run_live_new.py#L157)). These image dirs are gitignored, so a fresh clone cannot run inference without rebuilding them (see Data pipeline below).
- `spacy` model: `python -m spacy download en_core_web_sm`.

## Training

No CLI wrapper — each training script calls `train()` at module bottom, so training runs on `python training_testing_models/train_continent_classifier.py` (or `train_flag_classifier.py`). Edit the file to switch to `test()` for confusion-matrix evaluation.

- Input size is actually **150×150** despite filenames saying `_250_250.h5` (see [training_testing_models/train_continent_classifier.py:33](training_testing_models/train_continent_classifier.py#L33)).
- Both scripts use `ModelCheckpoint(..., save_best_only=True, monitor='val_loss')` with `epochs=2000` — you are meant to interrupt manually.
- Class weights are computed from folder file counts via `get_class_weights()` and passed to `model.fit` to offset imbalance.

## Data pipeline

The `data/` directory is a mix of JSON reference data (checked in) and throwaway scraping/preprocessing scripts (checked in, but their outputs are gitignored — see `.gitignore`). Rebuild order if starting from scratch:

1. `data_gather.py` — Selenium scrapes Google Maps street-view thumbnails by sweeping lat/long, skipping ocean via `mpl_toolkits.basemap`. Saves to `data/images/<lat>_<long>.png` and tracks dedup in `urls_downloaded.json`. Requires a hard-coded `chromedriver_path` (no `webdriver_manager` here, unlike the flag scraper).
2. `sort_images_by_continent.py` — reverse-geocodes each `<lat>_<long>.png` filename → country → continent (via `countries_to_continents.json`), copies into `data/images_sorted_by_continent/<continent>/`. This is the continent-classifier training input.
3. `populate_images_country_flags.py` — Selenium+Google Images scrape of "<country> flag flying" photos (not icons) into `data/images_country_flags_original/<country-code>/`. A separate manual/offline step produces `images_country_flags_modified/` (used by the flag classifier); the transform step is not in the repo.
4. `balance_classes_continents.py` / `balance_classes_flags.py` — destructively `os.remove()` extra files past the min-class count to equalize. Run once; re-running on an already-balanced set is a no-op.
5. `visualize_*.py` — matplotlib class-distribution bar charts, useful after step 4.

Key JSON data (committed):
- `countries_in_geoguesser.json` — ISO-2 → full name. Source of truth for which countries exist in the output space. The flag model's folder names are ISO-2 codes and are mapped to display names via this file ([run_live_new.py:220](run_live_new.py#L220)).
- `countries_to_continents.json` — ISO-2 → one of `AF/AS/EU/NA/OC/SA`.
- `country_to_languages.json` — used in the text branch to convert a Vision-detected language back to candidate GeoGuessr countries ([run_live_new.py:208](run_live_new.py#L208)).
- `continents_in_geoguesser.json` — enum sentinel file (values all 0); not loaded at runtime.
- `urls_downloaded.json` — scraper dedup ledger; ~4.5 MB, committed.

## Things to know before editing

- Folder-label coupling: the order of classes in both `.h5` models is determined by Python's `glob` sort of the matching `data/images_{sorted_by_continent,country_flags_modified}/*` directories. Renaming, adding, or reordering those folders silently breaks inference label mapping — do not touch them without retraining.
- The `\\` splits in [run_live_new.py:147](run_live_new.py#L147) and [run_live_new.py:178](run_live_new.py#L178) are Windows path separators; these are not portable to POSIX as-is.
- `run_live_new.py` hardcodes 2560×1440 and a 60%-centered capture region. On other resolutions, edit `get_screen()` and the `get_screen(2560, 1440, ...)` call site together.
- `text()` in `train_continent_classifier.py` has a bug: it passes `num_countries=` to `ContinentClassifierModel` which only accepts `num_continents=`. Will `TypeError` if invoked — fix before using the `test()` path.
