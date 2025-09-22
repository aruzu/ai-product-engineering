# Homework #2: From User Reviews to Feature Pre-Test

## 🤖 Hybrid: Cluster-Driven Pipeline + OpenAI Agents SDK

This project implements a **hybrid architecture** combining:
1. **Cluster-based data pipeline**: Analyzes complete Viber dataset using k-means clustering
2. **Real user persona extraction**: Personas from actual user segments, not AI stereotypes
3. **AI feature generation**: Features addressing cluster-identified pain points
4. **OpenAI Agents SDK validation**: Multi-agent virtual board simulation for feature testing

**Key Architecture**: Direct pipeline for efficient data processing + OpenAI Agents SDK for complex multi-agent interactions.

## 🔧 Architecture: Hybrid Pipeline Design

### **Data-Driven Pipeline Flow**
```
viber.csv (complete dataset) →
  K-Means Clustering Analysis →
    Persona Extraction from Clusters →
      AI Feature Generation from Pain Points →
        OpenAI Agents SDK Multi-Agent Virtual Board
```

### **Key Architectural Benefits**
- ✅ **Real User Segments** - Personas based on actual user clusters, not AI hallucination
- ✅ **Full Dataset Processing** - Uses complete review dataset for comprehensive insights
- ✅ **Evidence-Based** - Every persona traces back to specific cluster data and reviews
- ✅ **Hybrid Efficiency** - Direct pipeline for data processing + Agents SDK for multi-agent simulation
- ✅ **OpenAI Agents SDK** - Sophisticated multi-agent virtual board validation
- ✅ **Configurable** - Behavior customizable via .env file

## 📁 Project Structure
```
Alex_Ruzu/lesson2/
├── README.md                               # This file - project documentation
├── CLAUDE.md                               # Guidance for Claude Code
├── .env                                    # Environment configuration
├── requirements.txt                        # Dependencies
├── src/
│   ├── __init__.py                         # Package initialization
│   ├── main.py                             # Single entry point
│   ├── pipeline.py                         # Core pipeline implementation
│   ├── clustering.py                       # K-means clustering and persona extraction
│   └── virtual_board.py                    # Multi-agent virtual board simulation
├── data/
│   └── viber.csv                           # Complete Viber app reviews dataset
├── docs/                                   # Generated outputs
│   ├── personas_and_features.md            # Cluster-based personas and features
│   ├── generated_personas.json             # Extracted personas from clusters
│   ├── generated_features.json             # AI-generated features
│   ├── virtual_user_board_summary.md       # Virtual board validation results
│   └── clustering_results/                 # Complete clustering analysis
│       ├── clustering_results.json         # Raw cluster data
│       ├── cluster_personas.json           # Source personas from clusters
│       └── feature_requests_prioritized.json # Prioritized pain points
└── logs/                                   # Operational logs
    ├── main.log                            # Application flow and business progress
    └── pipeline.log                        # Technical implementation details
```

## 🛠️ Core Pipeline Components

### 1. **Clustering Analysis** (`clustering.py`)
```python
class UserClusteringAnalyzer:
    def run_analysis(self, limit=None) -> Dict[int, Dict]:
        """Run k-means clustering analysis on the complete review dataset."""
```
- **Input**: Complete viber.csv dataset (all available reviews)
- **Processing**: TF-IDF vectorization, k-means clustering, silhouette score optimization
- **Output**: Cluster analysis with feature requests and pain points

### 2. **Persona Extraction** (`clustering.py`)
```python
def generate_personas_from_clusters(self, cluster_analysis: Dict[int, Dict]) -> List[Dict]:
    """Extract personas from real user clusters, not AI generation."""
```
- **Input**: Cluster analysis results
- **Processing**: Extract representative personas from top clusters by size and urgency
- **Output**: Up to 5 personas with cluster evidence and review traceability

### 3. **Feature Generation** (`pipeline.py`)
```python
def formulate_features(analysis_json: str, personas_json: str) -> str:
    """Generate features based on cluster pain points and extracted personas."""
```
- **Input**: Cluster-derived pain points and personas
- **Processing**: AI synthesizes features from real user pain points
- **Output**: 2-3 features addressing cluster-identified problems

### 4. **OpenAI Agents SDK Virtual Board** (`virtual_board.py`)
```python
from agents import Agent, Runner

class VirtualUserBoard:
    async def run_board_simulation_from_files(self):
        """Multi-agent simulation using OpenAI Agents SDK with cluster-based personas."""
```

## ⚙️ Configuration

### **Environment Variables** (`.env`)
```env
# Pipeline Configuration
MIN_FEATURES=2              # Minimum features to generate (2)
MAX_FEATURES=3              # Maximum features to generate (3)
MAX_CLUSTER_PERSONAS=5      # Maximum personas from clustering (5)

# OpenAI Configuration
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.7
```

**Benefits of Configuration:**
- **Flexible Behavior** - Adjust persona/feature counts without code changes
- **Environment-Specific** - Different settings for dev/prod
- **Team Coordination** - Consistent parameters across team members
- **Easy Experimentation** - Quick parameter tuning for optimization

## 🎪 Enhanced Virtual User Board

### **Multi-Agent Simulation**
- **Facilitator Agent** - Guides discussions and synthesizes feedback
- **Cluster-Based Persona Agents** - Created from actual user segments, not AI stereotypes
- **Structured Evaluation** - Systematic feature assessment process with authentic user perspectives

### **Best Practices Implemented**
- **Real User Representation** - Personas reflect actual user clusters from data
- **Evidence-Based Discussions** - Feedback grounded in real pain points
- **Comprehensive Logging** - Full conversation tracking and analysis
- **Multi-Round Validation** - Structured discussion flow with facilitator orchestration

## 🚀 Setup and Execution

### **Prerequisites**
```bash
# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate edu_ai_prod_eng2

# Install dependencies
pip install -r requirements.txt

# Set environment variables in .env file
OPENAI_API_KEY="your-api-key"
```

### **Running the Pipeline**
```bash
# Navigate to project directory
cd Alex_Ruzu/lesson2

# Run the complete cluster-driven pipeline
python src/main.py
```

### **Expected Output**
```
ALEX_RUZU LESSON2 SOLUTION
Cluster-Driven Product Development Pipeline
Using Complete Dataset Analysis + OpenAI Agents SDK

🔄 Analyzing Viber reviews...
📊 Running clustering algorithm to identify user segments...
✅ Analysis complete! Personas and features generated.

🎭 Starting virtual user board simulation...
✅ Virtual board simulation completed successfully!

🎉 Pipeline completed in 0:04:45

Generated Files:
   - docs/generated_personas.json
   - docs/generated_features.json
   - docs/personas_and_features.md
   - docs/virtual_user_board_summary.md

Logs available in:
   - logs/main.log (business progress)
   - logs/pipeline.log (technical details)
```

## 📊 Data Analysis Results
From complete `data/viber.csv` dataset, the clustering analysis identifies real user segments:
- **Cluster 0**: Activation issues (420 users, urgency: 0.54, avg rating: 1.9)
- **Cluster 4**: Ad frustration (3,295 users, urgency: 0.4, avg rating: 3.2)
- **Cluster 8**: Message/spam problems (374 users, urgency: 0.55, avg rating: 2.0)
- **Cluster 10**: Phone number issues (324 users, urgency: 0.54, avg rating: 1.7)
- **Cluster 5**: Technical problems (211 users, urgency: 0.45, avg rating: 2.5)

## 📁 Generated Outputs

### **Automated File Generation**
- ✅ `docs/generated_personas.json` - Cluster-extracted personas with evidence
- ✅ `docs/generated_features.json` - Features addressing real pain points
- ✅ `docs/personas_and_features.md` - Human-readable documentation
- ✅ `docs/clustering_results/` - Complete clustering analysis data
- ✅ `docs/virtual_user_board_summary.md` - Virtual board validation results

### **Quality Assurance**
- ✅ **Cluster Traceability** - Every persona links to specific cluster data and review IDs
- ✅ **Evidence-Based** - All outputs grounded in statistical user analysis
- ✅ **Real User Segments** - Personas represent actual user groups, not stereotypes
- ✅ **Comprehensive Data** - Uses complete dataset for maximum insights

## 🎯 Success Criteria
1. **Cluster-Based Analysis** - Uses statistical clustering to identify real user segments
2. **Data Grounding** - All personas extracted from actual user clusters with evidence
3. **Feature Relevance** - Generated features address pain points identified through clustering
4. **Complete Dataset** - Processes all available reviews for comprehensive insights
5. **Validation** - Virtual board uses cluster-derived personas for authentic testing
6. **Traceability** - Clear connection from cluster data to personas to features

## 🔧 Key Technologies
- **OpenAI Agents SDK** - Multi-agent simulation framework for virtual board
- **Scikit-learn** - K-means clustering and TF-IDF vectorization
- **NLTK** - Text preprocessing and sentiment analysis
- **OpenAI API** - GPT models for feature generation and agent coordination
- **Python** - Core implementation language with type hints
- **Pandas** - Data processing and CSV analysis
- **Environment Configuration** - .env file for flexible parameter management

## 🎖️ Architecture Benefits

### **Compared to AI-Only Approaches**
| Aspect | AI-Generated Personas | Cluster-Based Personas |
|--------|----------------------|----------------------|
| **Data Grounding** | Limited | Complete |
| **User Authenticity** | Stereotypes | Real Segments |
| **Evidence** | Weak | Strong |
| **Scalability** | Poor | Excellent |
| **Bias Risk** | High | Low |
| **Actionability** | Low | High |

### **Production Ready Features**
- ✅ **Statistical Validation** - Clustering analysis provides objective segmentation
- ✅ **Complete Data Usage** - Processes entire dataset for comprehensive insights
- ✅ **Evidence Traceability** - Every persona links to specific cluster and review data
- ✅ **Configurable Pipeline** - Easy parameter adjustment via environment variables
- ✅ **Efficient Processing** - Direct pipeline execution without unnecessary overhead

## 📖 Documentation & Logs
- **README.md** - This comprehensive project documentation (root directory)
- **CLAUDE.md** - Guidance for Claude Code interactions
- **logs/main.log** - Application flow and business-level progress
- **logs/pipeline.log** - Technical implementation details and processing steps
- **Generated Documentation** - Automatic persona and feature documentation in docs/
- **Clustering Results** - Complete statistical analysis and evidence data in docs/clustering_results/

This solution demonstrates **data-driven product development** and serves as a reference implementation for **cluster-based user analysis and feature generation**.

## 🔄 When to Consider Embeddings + UMAP Migration

**Current approach (TF-IDF + K-Means) is optimal for most product development use cases.**

**Consider migrating to Embeddings + UMAP only when:**
- ✅ **Multilingual reviews** - Multiple languages in your dataset
- ✅ **Highly varied expressions** - Users describe same issues with completely different vocabulary
- ✅ **Cross-lingual analysis needed** - Want to cluster content regardless of language
- ✅ **Research depth prioritized** over business speed and interpretability

**Keep TF-IDF when:**
- ✅ **Fast iteration required** (5 seconds vs 5 minutes)
- ✅ **Clear keyword patterns exist** (technical issues, specific feature complaints)
- ✅ **Stakeholder interpretability crucial** (product managers need to understand WHY clusters formed)
- ✅ **Resource constraints** (embeddings require 10-20x more memory and processing)
- ✅ **Monolingual dataset** (English-only reviews)