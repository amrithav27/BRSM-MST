# BRSM-MST: Boundary-Related Segmentation and Memory - Mnemonic Similarity Task

## Project Overview

This project investigates the effects of event boundaries on memory performance using the Mnemonic Similarity Task (MST). The study examines how task boundaries and item boundaries influence recognition memory and pattern separation.

**Author:** Korlapati Keerthana
**Date:** 05/3/2026

## Experiments

The project consists of three experimental conditions:

1. **Both Item and Task Boundaries** - Participants experience both types of boundaries during encoding
2. **Item Only Boundaries** - Only item-level boundaries are present
3. **Task Only Boundaries** - Only task-level boundaries are present

## Repository Structure

```
BRSM-MST/
├── src/
│   ├── phase1.Rmd                 # Main analysis notebook
│   ├── Phase_1_dataviz.Rmd       # Visualization with plot saving
│   └── Phase_1_Analysis.Rmd      # Statistical analysis
├── Both_item_task/
│   └── both_data/                # Data files for Experiment 1
├── item_only/
│   └── item_only_data/           # Data files for Experiment 2
├── task_only/
│   └── task_only_data/           # Data files for Experiment 3
└── README.md
```

## Requirements

### R Packages

```r
install.packages("tidyverse")
install.packages("afex")
```

## Data Structure

### Task Files
- Pattern: `*_MST_task_*.csv` or `task*.csv`
- Contains encoding phase data with reaction times and trial information
- Key columns: `image_path`, `trials.thisN`, `trials.key_resp_9.rt`, `trials.key_resp_8.rt`

### Test Files
- Pattern: `*_MST_test_*.csv` or `test*.csv`
- Contains recognition test data with participant responses
- Key columns: `image_path`, `trials.key_resp_3.keys`, `trials.key_resp_3.rt`, `position_of_stimuli`

## Metrics

### MST Performance Metrics

1. **Recognition Score (REC)**
   - Formula: `P(Old|Target) - P(Old|Foil)`
   - Measures overall recognition memory

2. **Lure Discrimination Index (LDI)**
   - Formula: `P(Similar|Lure) - P(Similar|Foil)`
   - Measures pattern separation ability

### Boundary Conditions

- **Pre-boundary**: Event position 7 (last item before boundary)
- **Post-boundary**: Event position 1 (first item after boundary)
- **Non-boundary**: Event positions 2-6 (control items)

## Analysis Workflow

### 1. Data Loading and Preprocessing

```r
# Set working directory to project root
setwd("/path/to/BRSM-MST")

# Run preprocessing
source("src/phase1.Rmd")
```

### 2. Generate Visualizations

```r
# Generate and save all plots
source("src/Phase_1_dataviz.Rmd")
```

Output plots are saved as PNG files with descriptive names:
- `both_Distribution_of_Responses.png`
- `item_MST_Metrics_Distribution.png`
- `task_Reaction_Time_by_Stimulus_Type.png`
- etc.

### 3. Statistical Analysis

```r
# Run statistical tests
source("src/Phase_1_Analysis.Rmd")
```

## Key Analyses

### Encoding Phase
- **Reaction time by boundary condition**: ANOVA testing RT differences across Pre/Post/Non-boundary positions
- **Boundary effect on encoding speed**: Comparing encoding times near event boundaries

### Recognition Phase
- **Stimulus × Response confusion matrices**: Analyzing response patterns for Targets, Lures, and Foils
- **Boundary effects on memory**: Testing whether items encoded near boundaries show different recognition patterns
- **Pattern separation by boundary**: Examining lure discrimination near vs. away from boundaries

### Cross-Experiment Comparisons
- **REC and LDI across experiments**: Comparing memory metrics across the three boundary conditions
- **Boundary × Experiment interactions**: Testing whether boundary effects differ by experimental manipulation

## Expected Results

### Typical Findings
- Higher reaction times at boundary positions during encoding
- Enhanced memory for items encoded at event boundaries (boundary advantage)
- Potential differences in pattern separation near boundaries
- Variation in effects across the three experimental conditions

## Output

### Console Output
- Participant counts and data summaries
- MST metrics (REC and LDI) with means and SDs
- ANOVA results and t-test statistics

### Visualizations
- Distribution plots for responses and reaction times
- Violin/box plots comparing conditions
- Confusion matrices for stimulus-response patterns
- Cross-experiment comparison plots

## Troubleshooting

### Common Issues

1. **File not found errors**
   - Ensure data directories match the structure above
   - Check that CSV files follow naming conventions

2. **Missing columns**
   - The code handles missing RT columns automatically
   - Verify your CSV files have required columns

3. **Participant ID extraction**
   - Participant IDs are extracted from the first 5 characters of filenames
   - Ensure consistent filename format



## Contact

For questions or issues, please contact Korlapati Keerthana.

