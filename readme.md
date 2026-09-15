# Data Science Project

## Project layout

```text
data-science/
|-- data/
|   |-- raw/          # Original, immutable data
|   |-- interim/      # Temporary data between processing steps
|   |-- processed/    # Clean data ready for analysis or modeling
|   `-- external/     # Data from external sources
|-- notebooks/
|   |-- exploratory/  # EDA and investigation
|   `-- modeling/     # Experiments and model evaluation
|-- src/
|   |-- data/         # Loading and cleaning utilities
|   |-- features/     # Feature engineering
|   |-- models/       # Training and prediction code
|   `-- visualization/ # Reusable plots and dashboards
|-- models/           # Saved model artifacts
|-- reports/
|   `-- figures/      # Generated charts and visual assets
|-- scripts/          # Repeatable command-line workflows
|-- tests/            # Automated tests
|-- docs/             # Project documentation
|-- requirements.txt  # Python dependencies
`-- venv/             # Local virtual environment, not committed
```

Keep raw data unchanged, put reusable logic in `src/`, and use notebooks for exploration rather than as the main home for production code.
