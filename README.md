# ✈️ Flight Analysis Dashboard

An interactive web dashboard analyzing airline performance, flight delays, and predictive modeling using comprehensive flight data visualization and machine learning techniques.

![Dashboard Preview](https://img.shields.io/badge/Status-Live%20Demo-brightgreen)
![Technologies](https://img.shields.io/badge/Tech-Python%7CPandas%7CScikit--learn%7CJavaScript-blue)
![Theme](https://img.shields.io/badge/Design-Jekyll%20Blue%20%26%20Gray-lightblue)

## 🎯 Project Overview

This project demonstrates advanced data analysis and visualization skills through comprehensive flight delay analysis, featuring:

- **Airline Performance Analysis** - Delay frequency comparisons across carriers
- **Interactive Visualizations** - Dynamic airline-specific insights
- **Statistical Analysis** - Delay distribution patterns and trends  
- **Machine Learning Models** - KNN vs SVM delay prediction comparison
- **Professional Web Interface** - Jekyll-inspired responsive design

## 🚀 Live Demo

[View Interactive Dashboard](https://yourusername.github.io/flight-analysis-dashboard/)

## 📊 Key Features

### Overview Analytics
- **Airline Performance Ranking** - Horizontal bar chart of delay frequencies
- **Flight Volume Distribution** - Pie chart showing flights per airline
- **Delay Contribution Analysis** - Pie chart of delayed flights by carrier

### Interactive Airline Analysis
- **Route Analysis** - Most common travel routes (direction-independent)
- **Hub Identification** - Largest originating airports per airline
- **Delay Hotspots** - Origins with highest average delays
- **Best Performers** - Origins with lowest average delays

### Statistical Distributions
- **Delay Histograms** - Distribution patterns for top 5 airlines
- **Airport Comparisons** - Best vs worst performing airports
- **Pattern Recognition** - Long-tail delay distributions

### Machine Learning Classification
- **Model Comparison** - KNN vs SVM for >20-minute delay prediction
- **Performance Metrics** - Cross-validation accuracy, precision, recall
- **Predictive Analytics** - Feature importance and model evaluation

## 🛠️ Technologies Used

- **Backend**: Python 3.8+, Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn with custom styling
- **Frontend**: HTML5, CSS3 (Jekyll-inspired theme), Vanilla JavaScript
- **Machine Learning**: Classification algorithms, cross-validation, performance metrics
- **Design**: Responsive grid layouts, interactive components

## 📁 Project Structure

```
flight-analysis-dashboard/
├── analysis.py                 # Main analysis script
├── index.html                 # Interactive dashboard
├── style.css                  # Jekyll blue/gray theme
├── script.js                  # Interactive functionality
├── requirements.txt           # Python dependencies
├── assets/                    # Generated visualizations
│   ├── airline_performance.png
│   ├── flights_by_airline.png
│   ├── delayed_flights_by_airline.png
│   ├── [airline]_routes.png
│   ├── [airline]_hubs.png
│   ├── [airline]_worst_origins.png
│   ├── [airline]_best_origins.png
│   ├── airline_delay_distributions.png
│   ├── airport_delay_distributions.png
│   ├── classification_comparison.png
│   ├── airline_list.json
│   └── classification_results.json
└── README.md
```

## 🏃‍♂️ Quick Start

### Prerequisites
- Python 3.8 or higher
- Modern web browser
- Git

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/flight-analysis-dashboard.git
   cd flight-analysis-dashboard
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate analysis and visualizations**
   ```bash
   python analysis.py
   ```
   
   This creates:
   - All visualization charts in `/assets`
   - Interactive data files (JSON)
   - Performance comparison metrics

4. **View the dashboard**
   - Open `index.html` in your web browser
   - Or serve locally:
     ```bash
     python -m http.server 8000
     # Visit http://localhost:8000
     ```

## 📈 Analysis Pipeline

### Data Generation
```python
# Creates synthetic flight dataset with realistic patterns
- 50,000+ flight records across 8 major airlines
- 20 airports with varying delay characteristics  
- Seasonal, hourly, and airline-specific delay patterns
- Route-based analysis with direction independence
```

### Statistical Analysis
```python
# Comprehensive delay pattern analysis
- Airline performance ranking by delay frequency
- Route popularity and hub identification
- Airport-specific delay characteristics
- Distribution analysis with seasonal effects
```

### Machine Learning Models
```python
# Binary classification for significant delays (>20 minutes)
- Feature engineering: airline, route, time, distance
- Model comparison: KNN (k=5) vs SVM (RBF kernel)
- 5-fold cross-validation with stratified sampling
- Performance metrics: accuracy, precision, recall
```

## 🎨 Design Features

### Jekyll-Inspired Theme
- **Color Palette**: Professional blue (#3498db) and gray (#6c757d) scheme
- **Typography**: Inter font family for modern readability
- **Layout**: Responsive grid system with hover animations
- **Components**: Card-based design with subtle shadows and borders

### Interactive Elements
- **Dropdown Selection**: Airline-specific analysis switching
- **Smooth Transitions**: CSS animations and loading states
- **Responsive Design**: Mobile-friendly breakpoints
- **Error Handling**: Graceful fallbacks for missing data

## 📊 Key Insights

### Airline Performance
- Delay rates vary from ~20% to 35% across carriers
- Hub airports show higher delay rates due to traffic complexity
- Seasonal patterns affect delay frequency and duration

### Predictive Modeling
- SVM outperforms KNN for delay prediction (86.2% vs 83.5% accuracy)
- Route, timing, and airline factors are key predictors
- Cross-validation ensures model generalizability

### Operational Patterns
- Major hubs experience higher delay variability
- Time-of-day effects show peak hour congestion impact
- Route popularity correlates with delay frequency

## 🔧 Customization

### Adding New Airlines
```python
# In analysis.py, modify:
airlines = ['AA', 'UA', 'DL', 'WN', 'B6', 'AS', 'NK', 'F9', 'NEW_AIRLINE']
```

### Adjusting Visualizations
```python
# Customize chart styling:
plt.rcParams['axes.facecolor'] = '#your_color'
colors = ['#custom_blue', '#custom_red', '#custom_green']
```

### Modifying ML Models
```python
# Add new algorithms:
rf = RandomForestClassifier(n_estimators=100)
models = {'KNN': knn, 'SVM': svm, 'RandomForest': rf}
```

## 📱 Responsive Design

- **Desktop**: Full 3-column layout with side-by-side comparisons
- **Tablet**: 2-column adaptive grid with stacked sections
- **Mobile**: Single-column layout with optimized touch interactions

## 🚀 Deployment

### GitHub Pages Setup

1. **Enable GitHub Pages**
   - Repository Settings → Pages
   - Source: Deploy from branch (main)
   - Folder: / (root)

2. **Auto-deployment**
   ```bash
   git add .
   git commit -m "Update flight analysis dashboard"
   git push origin main
   ```

3. **Live URL**
   ```
   https://yourusername.github.io/flight-analysis-dashboard/
   ```

## 🔍 Performance Optimizations

- **Image Optimization**: Charts saved at optimal DPI (150) for web display
- **File Size Management**: Limited to <25MB total for GitHub compatibility
- **Lazy Loading**: Progressive image loading with error handling
- **Caching**: Browser-friendly asset caching for repeat visits

## 🏆 Skills Demonstrated

**Data Science & Analytics**
- ✅ Data generation and simulation
- ✅ Statistical analysis and hypothesis testing
- ✅ Distribution analysis and pattern recognition
- ✅ Performance benchmarking and comparison

**Machine Learning**
- ✅ Feature engineering and preprocessing
- ✅ Classification algorithms (KNN, SVM)
- ✅ Model evaluation and cross-validation
- ✅ Performance metrics and interpretation

**Data Visualization**
- ✅ Multi-chart dashboard creation
- ✅ Interactive visualization design
- ✅ Custom styling and theming
- ✅ Responsive chart implementations

**Web Development**
- ✅ Modern HTML5/CSS3/JavaScript
- ✅ Responsive design principles
- ✅ Interactive user interfaces
- ✅ Professional UI/UX design

## 📞 Contact

**Jake Medeiros**
- LinkedIn: [linkedin.com/in/jakemedeiros](https://linkedin.com/in/jakemedeiros)
- GitHub: [github.com/jakemedeiros7](https://github.com/jakemedeiros7)
- Email: jakemedeiros7@gmail.com

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

*Built to demonstrate comprehensive data analysis, machine learning, and web development skills for aviation industry applications.*