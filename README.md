# Product Review Analysis for E-commerce Platforms
## Fundamentals of Data Engineering Project

### Project Overview
This project demonstrates a complete **Data Engineering Lifecycle** implementation for analyzing product reviews from e-commerce platforms. It covers all fundamental concepts from data generation to serving, showcasing modern data engineering practices and tools.

### Course Objectives Addressed
1. ✅ **Understand the Fundamentals of Data Engineering** - Complete lifecycle implementation
2. ✅ **Explore Data Architectures and Design Principles** - Scalable, modular architecture
3. ✅ **Identify and Classify Source Systems and Data Storage Solutions** - Multiple storage abstractions
4. ✅ **Implement Effective Data Ingestion Strategies** - Batch and streaming pipelines
5. ✅ **Apply Data Modeling and Transformation Techniques** - Review analysis and sentiment processing

### Data Engineering Lifecycle Implementation

#### 1. **Generation (Source Systems)**
- **Simulated E-commerce APIs**: Product catalog, user reviews, ratings
- **Database Sources**: Customer data, product metadata
- **File Sources**: Historical review data (CSV, JSON, Parquet)
- **Real-time Streams**: Live review submissions, user interactions

#### 2. **Storage Systems & Abstractions**
- **Relational Database**: PostgreSQL for structured data (users, products)
- **NoSQL Database**: MongoDB for semi-structured review data
- **Data Lake**: Local file system simulating cloud storage (S3-like structure)
- **Data Warehouse**: Dimensional modeling for analytics

#### 3. **Ingestion Strategies**
- **Batch Ingestion**: Daily ETL jobs for historical data
- **Streaming Ingestion**: Real-time review processing
- **API Integration**: RESTful services for data collection
- **Change Data Capture**: Database change tracking

#### 4. **Transformation & Modeling**
- **Data Cleaning**: Handling missing values, duplicates, format standardization
- **Sentiment Analysis**: NLP processing for review sentiment scoring
- **Feature Engineering**: Review quality metrics, user behavior patterns
- **Data Modeling**: Star schema for analytics, normalized for operations

#### 5. **Serving & Analytics**
- **Interactive Dashboard**: Comprehensive Streamlit-based analytics platform
- **Real-time Visualizations**: Sentiment analysis, rating trends, quality metrics
- **Business Intelligence**: KPI monitoring, performance dashboards
- **Data Export**: CSV/JSON export capabilities for further analysis
- **Pipeline Monitoring**: Health checks and status monitoring

### Architecture Principles Applied
- **Scalability**: Modular design supporting horizontal scaling
- **Reliability**: Error handling, data quality checks, monitoring
- **Maintainability**: Clean code, documentation, version control
- **Security**: Data governance, access controls, privacy protection
- **Cost Optimization**: Efficient storage and processing strategies

## 🛠️ Technology Stack

### **Core Languages & Frameworks**
- **Python 3.9+** - Primary development language
- **SQL** - Database queries and data modeling
- **YAML** - Configuration management

### **Data Storage & Databases**
- **SQLite** - Local relational database for development
- **PostgreSQL** - Production-ready relational database
- **MongoDB** - NoSQL document database for semi-structured data
- **File System** - Data lake simulation (CSV, JSON, Parquet)

### **Data Processing & Analytics**
- **Pandas 2.1.4** - Data manipulation and analysis
- **NumPy 1.24.3** - Numerical computing
- **SQLAlchemy 2.0.23** - Database ORM and connection management
- **SciPy 1.11.4** - Scientific computing and statistics

### **Machine Learning & NLP**
- **Scikit-learn 1.3.2** - Machine learning algorithms and pipelines
- **NLTK 3.8.1** - Natural language processing toolkit
- **TextBlob 0.17.1** - Text processing and sentiment analysis

### **Visualization & Dashboard**
- **Streamlit 1.28.2** - Interactive web dashboard framework
- **Plotly 5.17.0** - Interactive data visualizations
- **Matplotlib 3.8.2** - Static plotting library
- **Seaborn 0.13.0** - Statistical data visualization
- **WordCloud 1.9.2** - Text visualization

### **API & Web Services**
- **FastAPI 0.104.1** - Modern web API framework
- **Uvicorn 0.24.0** - ASGI web server
- **Requests 2.31.0** - HTTP client library

### **Data Quality & Testing**
- **pytest 7.4.3** - Testing framework
- **Great Expectations 0.18.5** - Data quality validation
- **Custom validation modules** - Domain-specific data checks

### **Development & Configuration**
- **PyYAML 6.0.1** - YAML configuration parsing
- **python-dotenv 1.0.0** - Environment variable management
- **Faker 20.1.0** - Synthetic data generation
- **Jupyter 1.0.0** - Interactive development notebooks
- **Black 23.11.0** - Code formatting
- **Flake8 6.1.0** - Code linting

### **Optional/Future Integrations**
- **Apache Spark (PySpark)** - Big data processing (commented in requirements)
- **Apache Airflow** - Workflow orchestration (commented in requirements)

## 🤖 Models & Algorithms

### **Sentiment Analysis Models**

#### **1. VADER Sentiment Analyzer**
- **Type**: Rule-based lexicon approach
- **Library**: NLTK's SentimentIntensityAnalyzer
- **Features**:
  - Compound sentiment scores (-1 to +1)
  - Handles negations, intensifiers, and punctuation
  - Real-time processing capability
  - No training data required
- **Use Case**: Primary sentiment analysis for real-time processing

#### **2. Machine Learning Sentiment Models**
- **Algorithm**: Logistic Regression with TF-IDF
- **Pipeline Components**:
  - **TfidfVectorizer**: Text feature extraction
  - **LogisticRegression**: Binary/multi-class classification
  - **Alternative Models**: Multinomial Naive Bayes, Random Forest
- **Features**:
  - N-gram analysis (1-2 grams)
  - Stop word removal
  - Maximum 5000 features
  - Cross-validation and model persistence
- **Use Case**: Custom domain-specific sentiment analysis

#### **3. Ensemble Sentiment Analysis**
- **Approach**: Weighted combination of VADER and ML models
- **Weights**: VADER (40%) + ML Model (60%)
- **Benefits**: Combines rule-based and learned patterns
- **Use Case**: High-accuracy sentiment analysis

### **Text Processing Models**

#### **1. TF-IDF Vectorization**
- **Purpose**: Convert text to numerical features
- **Parameters**: Max features (5000), N-grams (1-2), English stop words
- **Applications**: Feature extraction for ML models

#### **2. Emotion Detection**
- **Type**: Keyword-based emotion classification
- **Emotions**: Joy, Anger, Sadness, Fear, Surprise, Disgust
- **Method**: Lexicon matching with normalization
- **Output**: Emotion probability distribution

#### **3. Text Preprocessing Pipeline**
- **Components**:
  - **Tokenization**: NLTK word_tokenize
  - **Lemmatization**: WordNet lemmatizer
  - **Stop word removal**: NLTK English stopwords
  - **Text cleaning**: Regex-based normalization
  - **Keyword extraction**: TF-IDF based importance

### **Feature Engineering Models**

#### **1. Statistical Features**
- **Text Statistics**: Word count, sentence count, average word length
- **Rating Features**: Rating distributions, temporal trends
- **User Behavior**: Review frequency, rating patterns
- **Product Features**: Category analysis, popularity metrics

#### **2. Temporal Features**
- **Time-based Analysis**: Review trends over time
- **Seasonality Detection**: Periodic pattern identification
- **Recency Scoring**: Time-weighted importance

#### **3. Quality Metrics**
- **Helpfulness Ratio**: Helpful votes / total votes
- **Review Length Analysis**: Optimal length detection
- **Verified Purchase Impact**: Purchase verification effects

### **Data Quality Models**

#### **1. Completeness Checks**
- **Missing Value Detection**: Column-wise completeness scoring
- **Required Field Validation**: Critical field presence verification

#### **2. Validity Checks**
- **Data Type Validation**: Schema compliance verification
- **Range Validation**: Numerical bounds checking (ratings 1-5)
- **Format Validation**: Text format and encoding checks

#### **3. Consistency Checks**
- **Cross-field Validation**: Logical relationship verification
- **Duplicate Detection**: Record uniqueness validation
- **Referential Integrity**: Foreign key relationship checks

### **Visualization Models**

#### **1. Interactive Dashboards**
- **Plotly Graphs**: Dynamic, interactive visualizations
- **Real-time Updates**: Live data refresh capabilities
- **Multi-dimensional Analysis**: Correlation heatmaps, scatter plots

#### **2. Statistical Visualizations**
- **Distribution Analysis**: Histograms, box plots, violin plots
- **Trend Analysis**: Time series plots with confidence intervals
- **Comparative Analysis**: Side-by-side metric comparisons

### Project Structure
```
Product-Review-Analysis/
├── README.md
├── requirements.txt
├── config/
│   ├── database_config.py
│   └── pipeline_config.yaml
├── src/
│   ├── data_generation/
│   ├── ingestion/
│   ├── storage/
│   ├── transformation/
│   ├── analytics/
│   ├── serving/
│   └── monitoring/
├── data/
│   ├── raw/
│   ├── processed/
│   └── warehouse/
├── notebooks/
├── tests/
├── docs/
└── deployment/
```

### Getting Started

#### Quick Start (Recommended)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the analytics dashboard
python run_dashboard.py
```

#### Manual Setup
1. **Setup Environment**: `pip install -r requirements.txt`
2. **Generate Sample Data**: Use the dashboard's "Generate New Data" button
3. **Run Full Pipeline**: Use the dashboard's "Run Full Pipeline" button
4. **Explore Analytics**: Navigate through the dashboard tabs

#### Advanced Usage
```bash
# Generate custom dataset
python src/data_generation/run_data_generation.py --num-reviews 5000

# Run transformation pipeline
python src/transformation/transformation_orchestrator.py

# Check data quality
python src/transformation/data_quality.py
```

### Analytics Dashboard Features

#### 📊 **Overview Tab**
- **Real-time KPIs**: Total reviews, average ratings, unique products/users
- **Pipeline Health**: Storage, transformation, and ingestion status monitoring
- **Data Quality Metrics**: Overall quality scores and issue tracking

#### 😊 **Sentiment Analysis Tab**
- **Sentiment Distribution**: Pie chart showing positive/negative/neutral breakdown
- **Score Histogram**: Distribution of sentiment scores across reviews
- **Temporal Trends**: Sentiment changes over time with confidence bands
- **Rating Correlation**: Sentiment scores vs. star ratings analysis

#### ⭐ **Rating Analysis Tab**
- **Rating Distribution**: Bar chart with percentage annotations
- **Temporal Trends**: Average ratings and review volume over time
- **Category Analysis**: Average ratings by product category with error bars

#### 🔍 **Data Quality Tab**
- **Quality Score Gauge**: Overall data quality with color-coded indicators
- **Quality Radar Chart**: Scores by check type (completeness, validity, etc.)
- **Issue Breakdown**: Quality issues categorized by severity level

#### 🔧 **Feature Engineering Tab**
- **Correlation Heatmap**: Feature relationships and dependencies
- **Feature Importance**: Top contributing features for analysis
- **Distribution Comparison**: Statistical distributions of engineered features

#### 📊 **Data Explorer Tab**
- **Interactive Data Preview**: Searchable and filterable dataset view
- **Dataset Statistics**: Shape, memory usage, and column information
- **Export Capabilities**: Download processed data in CSV/JSON formats

### Key Features
- **End-to-End Pipeline**: Complete data flow from source to insights
- **Interactive Analytics**: Real-time dashboard with advanced visualizations
- **Multiple Data Sources**: Simulating real-world complexity
- **Quality Assurance**: Data validation and monitoring throughout
- **Scalable Design**: Ready for cloud deployment and scaling
- **Business Value**: Actionable insights for e-commerce optimization

### Learning Outcomes
This project demonstrates practical application of:
- Data engineering lifecycle management
- Modern data architecture design
- ETL/ELT pipeline development
- Data quality and governance
- Real-world problem solving with data

---
*This project fulfills the requirements for Fundamentals of Data Engineering course, demonstrating comprehensive understanding of data engineering principles and practices.*