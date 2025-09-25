# Technical Requirements Document (TRD)
## Cluster-Driven Product Development Pipeline

### **Document Information**
- **Version**: 1.0
- **Date**: September 2025
- **Technical Lead**: Alex Ruzu
- **Architecture**: Data-Driven Pipeline + AI Multi-Agent System

---

## **1. System Architecture Overview**

### **High-Level Architecture**
```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                                  │
├─────────────────┬───────────────────┬───────────────────────────┤
│   Data Sources  │   Configuration   │      Environment         │
│                 │                   │                           │
│ • viber.csv     │ • .env variables  │ • OpenAI API access      │
│ • ~10K reviews  │ • Pipeline params │ • Python 3.9+           │
│ • Text content  │ • Model settings  │ • Required packages      │
└─────────────────┴───────────────────┴───────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 PROCESSING LAYER                                │
├──────────────────┬──────────────────┬──────────────────────────┤
│  Data Processing │   AI Processing  │    Validation Layer      │
│                  │                  │                          │
│ • Text cleaning  │ • Feature gen    │ • Pydantic models       │
│ • TF-IDF vectors │ • OpenAI API     │ • Error handling        │
│ • K-means        │ • Agents SDK     │ • Retry logic           │
│ • Clustering     │ • Multi-agent    │ • Data validation       │
└──────────────────┴──────────────────┴──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  OUTPUT LAYER                                   │
├─────────────────┬───────────────────┬───────────────────────────┤
│ Generated Files │     Logs         │       Reports             │
│                 │                  │                           │
│ • personas.json │ • main.log       │ • Markdown summaries     │
│ • features.json │ • pipeline.log   │ • Board transcripts      │
│ • board_summary │ • Session data   │ • Cluster analysis       │
└─────────────────┴───────────────────┴───────────────────────────┘
```

### **Data Flow Architecture**
```
Input Data → Preprocessing → Clustering → Persona Extraction
    ↓
Feature Generation ← Pain Point Analysis ← Cluster Analysis
    ↓
Virtual Board Simulation → Validation → Output Generation
    ↓
Artifact Management → Logging → Session Tracking
```

---

## **2. Technology Stack**

### **Core Processing Engine**
- **Language**: Python 3.9+ (required for modern type hints and performance)
- **Data Processing**: pandas 2.0+ for DataFrame operations
- **Machine Learning**: scikit-learn 1.3+ for clustering and vectorization
- **Text Processing**: NLTK 3.8+ for preprocessing and sentiment analysis

### **AI Integration Layer**
- **Primary AI**: OpenAI API (GPT-4o-mini) for feature generation
- **Multi-Agent**: OpenAI Agents SDK for virtual board simulation
- **Data Validation**: Pydantic 2.0+ for type safety and validation
- **Configuration**: python-dotenv for environment management

### **Infrastructure Components**
- **Logging**: Python logging module with custom formatters
- **File I/O**: pathlib for cross-platform path handling
- **Serialization**: JSON for data exchange, Markdown for reports
- **Error Handling**: Custom exception classes with retry logic

### **Development Dependencies**
```python
# Core Processing
pandas>=2.0.0
scikit-learn>=1.3.0
nltk>=3.8.0
numpy>=1.24.0

# AI Integration
openai>=1.0.0
openai-agents-sdk>=1.0.0
pydantic>=2.0.0
python-dotenv>=1.0.0

# Utilities
pathlib2>=2.3.0  # Enhanced path handling
typing-extensions>=4.0.0  # Extended type hints
```

---

## **3. Component Specifications**

### **3.1 Data Processing Module** (`src/clustering.py`)

#### **Class: UserClusteringAnalyzer**
```python
class UserClusteringAnalyzer:
    def __init__(self, csv_path: str, output_dir: str = "docs/clustering_results")
    def preprocess_text(self, text: str) -> str
    def load_and_preprocess_data(self, limit: Optional[int] = None) -> pd.DataFrame
    def create_tfidf_features(self, texts: List[str]) -> Tuple[np.ndarray, TfidfVectorizer]
    def find_optimal_clusters(self, features: np.ndarray, min_k=3, max_k=15) -> int
    def perform_clustering(self, features: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray
    def analyze_clusters(self, df: pd.DataFrame, cluster_labels: np.ndarray, vectorizer: TfidfVectorizer) -> Dict[int, Dict]
    def generate_personas_from_clusters(self, cluster_analysis: Dict[int, Dict]) -> List[Dict]
    def save_results(self, cluster_analysis: Dict[int, Dict], df: pd.DataFrame) -> None
    def run_analysis(self, limit: Optional[int] = None) -> Dict[int, Dict]
```

#### **Performance Requirements**
- **Processing Speed**: Complete dataset processing in < 90 seconds for clustering
- **Memory Usage**: < 1GB RAM during clustering
- **Clustering Quality**: Silhouette score > 0.3
- **Output Format**: JSON and Markdown with complete traceability

#### **Algorithm Specifications**
- **TF-IDF Parameters**: max_features=1000, ngram_range=(1,2), min_df=3, max_df=0.85
- **K-means Parameters**: n_init=10, random_state=42, algorithm='lloyd'
- **Cluster Range**: 3-15 clusters with silhouette score optimization
- **Persona Limit**: Maximum 5 personas per analysis

### **3.2 Pipeline Orchestration** (`src/pipeline.py`)

#### **Function Specifications**
```python
def formulate_features(
    analysis_data: Dict[str, Any],
    personas: List[Persona],
    config: PipelineConfig,
    llm_client: LLMClient
) -> List[FeatureProposal]

def save_results(
    personas: List[Persona],
    features: List[FeatureProposal],
    config: PipelineConfig,
    artifact_manager: ArtifactManager
) -> None
```

#### **API Integration Requirements**
- **OpenAI Model**: gpt-4o-mini (cost-optimized)
- **Temperature**: 0.7 (balanced creativity/consistency)
- **Max Tokens**: 8000 per request
- **Retry Logic**: 3 attempts with exponential backoff
- **Rate Limiting**: Respect OpenAI API limits

### **3.3 Virtual Board Simulation** (`src/virtual_board.py`)

#### **Multi-Agent Architecture**
```python
class VirtualUserBoard:
    def __init__(self, output_dir: Path = Path("docs"))
    async def create_facilitator_agent(self) -> Agent
    async def create_persona_agents(self, personas: List[Persona]) -> List[Agent]
    async def run_structured_simulation(self, personas: List[Persona], features: List[FeatureProposal]) -> Dict[str, Any]
    async def run_board_simulation_from_files(self) -> Dict[str, Any]
```

#### **Simulation Requirements**
- **Agent Count**: 1 facilitator + up to 5 persona agents
- **Discussion Rounds**: 3 structured rounds per feature
- **Session Duration**: < 3 minutes total simulation time
- **Output Format**: Complete transcript + structured summary

### **3.4 Data Models** (`src/data_types.py`)

#### **Pydantic Models**
```python
class PipelineConfig(BaseModel):
    csv_path: str = Field(default="data/viber.csv")
    output_dir: str = Field(default="docs")
    min_features: int = Field(default=2, ge=1, le=5)
    max_features: int = Field(default=3, ge=1, le=5)
    max_cluster_personas: int = Field(default=5, ge=1, le=10)
    openai_model: str = Field(default="gpt-4o-mini")
    openai_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    virtual_board_rounds: int = Field(default=3, ge=1, le=5)

class Persona(BaseModel):
    name: str
    profile: Dict[str, Any]
    background: str
    pain_points: List[str]
    needs: List[str]
    usage_pattern: str
    cluster_info: Dict[str, Any]
    evidence: List[str]

class FeatureProposal(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=10, max_length=500)
    problem_addressed: str = Field(..., min_length=10, max_length=300)
    value_proposition: str = Field(..., min_length=10, max_length=300)
    priority_score: float = Field(default=0.5, ge=0.0, le=1.0)
```

---

## **4. Infrastructure Requirements**

### **4.1 System Requirements**

#### **Minimum Hardware**
- **CPU**: 2 cores, 2.0 GHz
- **RAM**: 4GB (2GB available for pipeline)
- **Storage**: 1GB free space for outputs and logs
- **Network**: Broadband internet for API access

#### **Recommended Hardware**
- **CPU**: 4 cores, 3.0 GHz
- **RAM**: 8GB (4GB available for pipeline)
- **Storage**: 5GB free space for multiple sessions
- **Network**: High-speed internet for optimal API performance

### **4.2 Software Dependencies**

#### **Operating System Support**
- **Primary**: Ubuntu 20.04+ LTS
- **Secondary**: macOS 11+, Windows 10+
- **Container**: Docker support for isolated deployment

#### **Python Environment**
- **Version**: Python 3.9+ (required)
- **Package Manager**: pip 21.0+ or conda 4.10+
- **Virtual Environment**: venv or conda environment recommended

### **4.3 External Service Dependencies**

#### **OpenAI API Requirements**
- **Account**: OpenAI API account with billing enabled
- **Credits**: Minimum $10 credit balance recommended
- **Rate Limits**: Tier 1+ for reliable processing
- **Models**: Access to gpt-4o-mini model

#### **Network Requirements**
- **Bandwidth**: 10 Mbps minimum for API calls
- **Latency**: < 500ms to OpenAI endpoints
- **Reliability**: 99% uptime for continuous operation

---

## **5. Performance Specifications**

### **5.1 Processing Performance**

#### **Clustering Performance**
```
Dataset Size    │ Processing Time │ Memory Usage │ Quality Score
─────────────────┼─────────────────┼──────────────┼──────────────
1,000 reviews   │ < 15 seconds    │ < 500MB      │ > 0.3
~5,000 reviews  │ < 90 seconds    │ < 1GB        │ > 0.3
~10,000 reviews │ < 3 minutes     │ < 2GB        │ > 0.25
```

#### **Feature Generation Performance**
```
Feature Count   │ API Calls       │ Processing Time │ Token Usage
─────────────────┼─────────────────┼─────────────────┼──────────────
2 features      │ 1-2 calls       │ < 15 seconds    │ < 4K tokens
3 features      │ 1-2 calls       │ < 30 seconds    │ < 6K tokens
5 features      │ 2-3 calls       │ < 45 seconds    │ < 8K tokens
```

#### **Virtual Board Performance**
```
Participants    │ Rounds          │ Simulation Time │ Output Size
─────────────────┼─────────────────┼─────────────────┼──────────────
3 personas      │ 3 rounds        │ < 2 minutes     │ < 50KB
5 personas      │ 3 rounds        │ < 3 minutes     │ < 100KB
5 personas      │ 5 rounds        │ < 5 minutes     │ < 150KB
```

### **5.2 Scalability Limits**

#### **Data Processing Limits**
- **Maximum Reviews**: 20,000 (with performance degradation)
- **Maximum Clusters**: 50 (algorithm stability limit)
- **Maximum Personas**: 10 (diminishing returns beyond 5)
- **Maximum Features**: 10 (API cost considerations)

#### **Concurrent Usage**
- **Single User**: Full performance guaranteed
- **Multiple Users**: Resource contention may occur
- **Container Deployment**: Recommended for multi-user scenarios

---

## **6. Security & Reliability**

### **6.1 Data Security**

#### **Input Data Protection**
- **Local Storage**: All data processed locally, not transmitted to third parties
- **API Security**: Only feature descriptions sent to OpenAI (no raw review data)
- **Key Management**: OpenAI API keys stored in environment variables
- **Access Control**: File system permissions for generated outputs

#### **Privacy Considerations**
- **Data Anonymization**: Review IDs used for traceability, not personal data
- **Content Filtering**: No PII transmitted to external APIs
- **Local Processing**: Clustering and analysis performed entirely locally

### **6.2 Error Handling & Recovery**

#### **Error Categories**
```python
class ClusteringError(Exception): # Data quality or algorithm failures
class ValidationError(Exception): # Pydantic model validation failures
class LLMError(Exception):       # OpenAI API errors and timeouts
class ConfigurationError(Exception): # Environment or parameter errors
```

#### **Recovery Strategies**
- **API Failures**: Exponential backoff retry (3 attempts)
- **Data Errors**: Graceful degradation with partial results
- **Memory Issues**: Automatic data chunking for large datasets
- **Configuration**: Comprehensive validation with helpful error messages

### **6.3 Monitoring & Logging**

#### **Logging Architecture**
```
logs/
├── main.log          # Business-level progress and results
└── pipeline.log      # Technical implementation details
```

#### **Log Content Specification**
- **main.log**: Configuration, phases, completion status, file outputs
- **pipeline.log**: Clustering details, API calls, processing steps, errors
- **Console**: User-friendly progress indicators only

#### **Monitoring Points**
- **Pipeline Success Rate**: Target 95% completion rate
- **Performance Metrics**: Processing time per dataset size
- **API Usage**: Token consumption and cost tracking
- **Error Patterns**: Common failure modes and frequencies

---

## **7. Configuration Management**

### **7.1 Environment Variables**

#### **Required Configuration**
```bash
# OpenAI Integration
OPENAI_API_KEY=sk-...                    # Required: API access key
OPENAI_MODEL=gpt-4o-mini                 # Default: cost-optimized model
OPENAI_TEMPERATURE=0.7                   # Default: balanced creativity

# Pipeline Parameters
MIN_FEATURES=2                           # Default: minimum feature count
MAX_FEATURES=3                           # Default: maximum feature count
MAX_CLUSTER_PERSONAS=5                   # Default: persona limit
VIRTUAL_BOARD_ROUNDS=3                   # Default: simulation depth
```

#### **Optional Configuration**
```bash
# Advanced Settings
CLUSTERING_MIN_K=3                       # Minimum cluster count
CLUSTERING_MAX_K=15                      # Maximum cluster count
TFIDF_MAX_FEATURES=1000                  # TF-IDF vocabulary size
SENTIMENT_THRESHOLD=0.1                  # Sentiment classification boundary

# Performance Tuning
MAX_REVIEWS_LIMIT=10000                  # Dataset size limit
API_TIMEOUT_SECONDS=120                  # OpenAI API timeout
RETRY_ATTEMPTS=3                         # Error retry count
LOG_LEVEL=INFO                           # Logging verbosity
```

### **7.2 Configuration Validation**

#### **Startup Validation**
- **API Key**: Test OpenAI API connectivity
- **File Paths**: Verify input data and output directory accessibility
- **Parameter Ranges**: Validate numeric ranges and constraints
- **Dependencies**: Check required package availability

#### **Runtime Validation**
- **Data Quality**: Review count, content length, format validation
- **Resource Availability**: Memory and disk space monitoring
- **API Quotas**: Rate limit and credit balance checking

---

## **8. Testing Requirements**

### **8.1 Unit Testing**

#### **Component Coverage**
- **Data Processing**: Text preprocessing, vectorization, clustering
- **Validation**: Pydantic model validation and error handling
- **Utilities**: Logging, configuration, file operations
- **Error Handling**: Exception scenarios and recovery

#### **Test Data Requirements**
- **Synthetic Dataset**: 1,000 generated reviews for consistent testing
- **Edge Cases**: Empty reviews, special characters, malformed data
- **Performance Tests**: Large dataset simulation (10,000 reviews)

### **8.2 Integration Testing**

#### **API Integration**
- **OpenAI Connectivity**: Feature generation with mock and real APIs
- **Agents SDK**: Virtual board simulation end-to-end
- **Error Scenarios**: API failures, rate limiting, timeout handling

#### **Pipeline Testing**
- **End-to-End**: Complete pipeline with known dataset
- **Configuration**: Different parameter combinations
- **Output Validation**: Generated file format and content verification

### **8.3 Performance Testing**

#### **Load Testing**
- **Dataset Scaling**: 1K, 5K, 10K review performance measurement
- **Memory Profiling**: Peak usage monitoring and optimization
- **API Usage**: Token consumption and cost analysis

#### **Stress Testing**
- **Resource Limits**: Maximum dataset size handling
- **Failure Recovery**: System behavior under error conditions
- **Concurrent Usage**: Multi-user scenario simulation

---

## **9. Deployment Specifications**

### **9.1 Local Development Setup**

#### **Environment Setup**
```bash
# Repository clone
git clone <repository-url>
cd lesson2

# Environment creation
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Dependency installation
pip install -r requirements.txt

# Configuration
cp .env.example .env
# Edit .env with OpenAI API key

# Verification
python src/main.py  # Should complete successfully
```

### **9.2 Production Deployment**

#### **Container Deployment** (Recommended)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/
COPY .env .env

CMD ["python", "src/main.py"]
```

#### **Resource Allocation**
- **CPU**: 2 cores minimum, 4 cores recommended
- **Memory**: 4GB minimum, 8GB recommended
- **Storage**: 10GB for logs and artifacts
- **Network**: Outbound HTTPS access to OpenAI APIs

### **9.3 Monitoring & Maintenance**

#### **Health Checks**
- **API Connectivity**: Periodic OpenAI endpoint testing
- **Resource Usage**: Memory and disk space monitoring
- **Log Analysis**: Error pattern detection and alerting
- **Performance Tracking**: Processing time trend analysis

#### **Maintenance Tasks**
- **Log Rotation**: Automated cleanup of old log files
- **Artifact Management**: Session-based output organization
- **Dependency Updates**: Regular security and feature updates
- **Performance Optimization**: Query and algorithm tuning

---

## **10. Appendix**

### **10.1 API Specifications**

#### **OpenAI API Usage Patterns**
```python
# Feature Generation Request
{
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 2000,
    "messages": [
        {"role": "system", "content": "Feature generation prompt"},
        {"role": "user", "content": "Cluster analysis data + personas"}
    ]
}

# Expected Response Format
{
    "features": [
        {
            "name": "Feature Name",
            "description": "Detailed description",
            "problem_addressed": "Pain point solution",
            "value_proposition": "User benefit"
        }
    ]
}
```

### **10.2 Data Schemas**

#### **Input Data Schema** (CSV)
```csv
reviewId,content,score,date
1,"Review text content...",4,2023-01-01
2,"Another review...",2,2023-01-02
```

#### **Output Data Schema** (JSON)
```json
{
    "personas": [
        {
            "name": "Persona Name",
            "profile": {"age": 30, "role": "Manager", "location": "City"},
            "background": "Background description",
            "pain_points": ["Pain point 1", "Pain point 2"],
            "needs": ["Need 1", "Need 2"],
            "usage_pattern": "Usage description",
            "cluster_info": {"cluster_id": 0, "size": 420, "urgency": 0.54},
            "evidence": ["review_id_1", "review_id_2"]
        }
    ],
    "features": [
        {
            "name": "Feature Name",
            "description": "Feature description",
            "problem_addressed": "Problem solved",
            "value_proposition": "User value",
            "priority_score": 0.8
        }
    ]
}
```

### **10.3 Algorithm Details**

#### **TF-IDF Configuration**
- **Vocabulary Size**: 1000 terms maximum
- **N-gram Range**: Unigrams and bigrams (1,2)
- **Document Frequency**: min_df=3, max_df=0.85
- **Normalization**: L2 normalization applied

#### **K-means Parameters**
- **Initialization**: k-means++ for stable centroids
- **Iterations**: Maximum 300 iterations
- **Tolerance**: 1e-4 convergence threshold
- **Random State**: 42 for reproducible results

---

*This Technical Requirements Document provides the complete technical specification for implementing a production-ready cluster-driven product development pipeline.*