# DA323-Assignment
Assignment (cum project-01) for the course DA323: Multimodal Data Processing and Learning 2.0

## Overview
This repository contains my submission for the Assignment-cum-Project-01 for the course DA323: Multimodal Data Analysis and Learning 2.0 at IIT Guwahati (Jan-May 2025). The project focuses on scalable data collection, multimodal analysis, and computational matching techniques across various data modalities.

## Tasks

### 1. Task1
Implementation of automated methods to collect, process, and analyze different types of data:

- [**Image Dataset Collection**](./PokeVision_dataset): Automated script to download 50 images for each of 20 selected categories
- [**Text Dataset Collection**](./AstroCorpus): Web crawler to extract content from websites for 20 categories
- [**Audio Dataset Collection**](./Audio_Dataset_Collection): Script to record and store audio streams from online radio stations
- [**Weather Dataset Collection**](./Weather_Dataset_Collection): Implementation of weather data collection from open APIs
- [**Analyzing India with Data**](./Analyzing_india_with_data/): Exploratory Data Analysis of a dataset from data.gov.in

### 2. [Task 2: Search for a Match](./Search_for_a_Match/)
Computational approach to match audio tracks with corresponding muted video clips by analyzing visual and acoustic features from a dataset of ball motion simulations.

### 3. Task 3
Analysis of visual characteristics of national flags and linguistic features of national anthems to identify potential correlations:

- [**Data Collection**](./Analyzing_Flags_and_Anthems/): Collection of flag images, anthem texts, and music files
- [**Visual Analysis**](./Analyzing_Flags_and_Anthems/): Analysis of flag images using data analysis techniques
- [**Textual Analysis**](./Analyzing_Flags_and_Anthems/): Processing and analysis of anthem translations
- [**Audio Analysis**](./Analyzing_Flags_and_Anthems/): Analysis of anthem music compositions
- [**Multimodal Correlation**](./Analyzing_Flags_and_Anthems/): Exploration of correlations between visual, textual, and audio modalities

## Repository Structure
```
├── Scalable_Data_Collection/
│   ├── Image_Dataset/
│   ├── Text_Dataset/
│   ├── Audio_Dataset/
│   ├── Weather_Dataset/
│   └── Analyzing_India_with_Data/
├── Search_for_a_Match/
└── Analyzing_Flags_and_Anthems/
├── Data_Collection/
├── Visual_Analysis/
├── Textual_Analysis/
├── Audio_Analysis/
└── Multimodal_Correlation/
```
## Technologies Used
- **Python libraries**: selenium, requests, BeautifulSoup, scrapy, ffmpeg, pydub, pandas, matplotlib, seaborn, plotly, and more
- **Data analysis techniques**: Feature extraction, correlation analysis, pattern recognition
- **Multimodal analysis**: Cross-modal feature matching, audio-visual correlation


## Requirements
Requirements are specified in the [requirements.txt](./requirements.txt) file.
