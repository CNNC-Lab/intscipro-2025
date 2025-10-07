# Comprehensive Data Analysis Pipeline: Behavioral Learning Experiment

## Project Overview
You are analyzing data from a behavioral learning experiment in genetically modified mice. This is a complex multi-session, multi-day experiment examining learning and memory across different stimulus modalities in wild-type (WT) and knockout (KO) mice.

## Dataset Description

### File: `complex_behavioral_data.csv`

**Experimental Design:**
- **Subjects**: 6 mice (3 WT, 3 KO genotypes)
- **Sessions**: Multiple sessions per day
- **Trials**: Multiple trials per session
- **Learning Phases**: Acquisition → Maintenance → Extinction
- **Stimulus Types**: Visual, auditory, tactile
- **Difficulty Levels**: Easy, medium, hard

### Column Definitions:

**Identity & Session Variables:**
- `subject_id`: Unique animal identifier (S001-S006)
- `session`: Session number within each day
- `trial`: Trial number within each session
- `block`: Block number for grouping trials
- `day`: Experimental day (1-3)
- `learning_phase`: Current phase (acquisition, maintenance, extinction)

**Performance Variables:**
- `response_time`: Time to respond to stimulus (seconds)
- `accuracy`: Correct response (1) or incorrect (0)
- `confidence`: Self-reported confidence measure (0-1 scale)
- `previous_accuracy`: Running accuracy up to current trial

**Stimulus Variables:**
- `stimulus_type`: Type of stimulus (visual, auditory, tactile)
- `difficulty`: Stimulus difficulty (easy, medium, hard)
- `stimulus_duration`: How long stimulus was presented (seconds)
- `reward_magnitude`: Size of reward (high, medium, low)

**Individual Differences:**
- `genotype`: Genetic modification status (WT = wild-type, KO = knockout)
- `age_weeks`: Age at testing (weeks)
- `weight_g`: Body weight (grams)
- `reaction_time_baseline`: Baseline reaction time for each animal

**State Variables:**
- `iti_duration`: Inter-trial interval duration (seconds)
- `stress_level`: Measured stress indicator (low, medium, high)
- `motivation_score`: Behavioral motivation measure (1-10 scale)

## Analysis Requirements

### 1. Data Quality and Preprocessing
- Load and inspect the dataset structure
- Check for missing values, outliers, and data consistency
- Create derived variables:
  - `learning_trial`: Trial number within each learning phase
  - `normalized_rt`: Response time normalized by baseline for each subject
  - `accuracy_rolling`: Rolling accuracy over last 5 trials
  - `session_progress`: Proportion of trials completed in current session

### 2. Descriptive Statistics and Data Exploration
- Generate comprehensive summary statistics by genotype and learning phase
- Create frequency tables for categorical variables
- Identify potential confounding variables (age, weight, motivation)
- Calculate learning curves (accuracy over trials) for each subject and condition

### 3. Learning Performance Analysis
- **Primary Question**: How does genotype affect learning across phases?
- Compare accuracy, response times, and confidence between WT and KO mice
- Analyze learning curves using exponential/logistic models
- Identify optimal performance levels and time-to-criterion for each genotype
- Test for differences in asymptotic performance levels

### 4. Stimulus Modality Effects
- Compare performance across visual, auditory, and tactile stimuli
- Test for genotype × stimulus interactions
- Analyze difficulty effects within each modality
- Create confusion matrices for error patterns by stimulus type

### 5. Temporal Dynamics and Adaptation
- Model within-session performance changes (fatigue, warm-up effects)
- Analyze between-session retention (memory consolidation)
- Track motivation and stress changes across days
- Investigate reward magnitude effects on sustained performance

### 6. Individual Differences and Clustering
- Perform clustering analysis to identify different learning strategies
- Correlate individual characteristics (weight, age, baseline RT) with performance
- Create individual subject profiles showing unique patterns
- Identify outlier subjects or sessions that deviate from group patterns

### 7. Statistical Modeling
- Fit mixed-effects models accounting for repeated measures
- Model accuracy using logistic regression with appropriate random effects
- Model response times using appropriate distributions (gamma, inverse gaussian)
- Include relevant covariates and interaction terms
- Perform model selection and validation

### 8. Advanced Analyses
- **Machine Learning**: Predict genotype from behavioral patterns using ensemble methods
- **Time Series**: Apply change-point detection to identify learning transitions
- **Network Analysis**: Create transition networks between accuracy states
- **Survival Analysis**: Time-to-first-success analysis for difficult trials

## Visualization Requirements

### 1. Overview Visualizations
- Dashboard-style summary showing key metrics by genotype
- Learning curves with confidence intervals for each condition
- Heatmaps showing performance across stimulus × difficulty combinations

### 2. Detailed Performance Plots
- Individual subject trajectories with group overlays
- Box plots and violin plots for distributions by key factors
- Scatter plots with regression lines for continuous relationships

### 3. Advanced Visualizations
- Interactive plots allowing filtering by different variables
- Animated plots showing temporal progression
- Network diagrams for behavioral transition patterns
- Correlation matrices with hierarchical clustering

### 4. Statistical Results Visualizations
- Forest plots for effect sizes with confidence intervals
- Model diagnostic plots (residuals, Q-Q plots, leverage)
- ROC curves for classification performance
- Partial effects plots for complex model terms

## Output Requirements

### 1. Processed Dataset
- Clean, validated dataset with derived variables
- Separate files for different analysis levels (trial, session, subject)
- Data dictionary documenting all variables and transformations

### 2. Analysis Report
- Executive summary with key findings
- Methods section describing statistical approaches
- Results section with interpretation of each analysis
- Discussion of limitations and future directions

### 3. Figures and Tables
- Publication-quality figures (300 DPI, appropriate fonts)
- Tables formatted for scientific publication
- Supplementary figures for detailed breakdowns
- Figure legends with complete statistical information

### 4. Reproducible Code
- Well-documented Python scripts organized by analysis type
- Jupyter notebook with narrative explanations
- Requirements file listing all dependencies
- README with instructions for reproduction

## Technical Specifications

### Code Style and Structure
- Use object-oriented programming where appropriate
- Implement error handling and data validation
- Follow PEP 8 style guidelines
- Include type hints for functions
- Use logging for tracking analysis progress

### Statistical Considerations
- Use appropriate multiple comparison corrections
- Report effect sizes alongside p-values
- Include confidence intervals for all estimates
- Check model assumptions and report diagnostics
- Use cross-validation for machine learning components

### Performance Optimization
- Use vectorized operations for large computations
- Implement parallel processing for computationally intensive analyses
- Cache intermediate results to avoid recomputation
- Profile code performance and optimize bottlenecks

## Expected Deliverables Timeline

### Phase 1: Data Preparation (30 minutes)
- Data loading, cleaning, and validation
- Derived variable creation
- Initial exploratory data analysis

### Phase 2: Core Analyses (60 minutes)
- Learning performance analysis
- Statistical modeling
- Stimulus modality effects

### Phase 3: Advanced Analyses (45 minutes)
- Machine learning components
- Advanced visualizations
- Individual differences analysis

### Phase 4: Reporting (30 minutes)
- Generate final report
- Create publication-ready figures
- Organize reproducible code package

## Success Criteria
The analysis will be considered successful if it:
1. Provides clear, statistically sound answers to the primary research questions
2. Identifies any unexpected patterns or interesting findings in the data
3. Generates reproducible, well-documented code
4. Creates publication-quality visualizations
5. Offers actionable insights for future experimental design

---
