#!/usr/bin/env python3
"""
Flight Analysis Dashboard
Comprehensive analysis of flight delays and airline performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
import warnings
import os

warnings.filterwarnings('ignore')

class FlightAnalysis:
    def __init__(self):
        self.df = None
        self.top_airlines = None
        self.airline_data = {}
        
        # Create assets directory
        os.makedirs('assets', exist_ok=True)
        
        # Set plotting style - blue/gray theme
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = '#f8f9fa'
        plt.rcParams['axes.facecolor'] = '#ffffff'
        plt.rcParams['axes.edgecolor'] = '#6c757d'
        plt.rcParams['axes.labelcolor'] = '#343a40'
        plt.rcParams['text.color'] = '#343a40'
        plt.rcParams['xtick.color'] = '#6c757d'
        plt.rcParams['ytick.color'] = '#6c757d'
        
    def load_data(self):
        """Load and prepare flight data"""
        try:
            # Try to load seaborn flights dataset
            flights = sns.load_dataset('flights')
            
            # Since seaborn flights doesn't have delay data, we'll simulate it
            np.random.seed(42)
            
            # Create a more comprehensive dataset
            airlines = ['AA', 'UA', 'DL', 'WN', 'B6', 'AS', 'NK', 'F9']  # Limit to 8 airlines
            airports = ['ATL', 'LAX', 'ORD', 'DFW', 'DEN', 'JFK', 'SFO', 'LAS', 'SEA', 'MCO',
                       'EWR', 'CLT', 'PHX', 'IAH', 'MIA', 'BOS', 'MSP', 'FLL', 'DTW', 'PHL']
            
            # Generate synthetic flight data
            n_flights = 50000
            data = {
                'airline': np.random.choice(airlines, n_flights, p=[0.18, 0.15, 0.14, 0.12, 0.08, 0.08, 0.12, 0.13]),
                'origin': np.random.choice(airports, n_flights),
                'dest': np.random.choice(airports, n_flights),
                'month': np.random.choice(range(1, 13), n_flights),
                'day': np.random.choice(range(1, 29), n_flights),
                'hour': np.random.choice(range(6, 23), n_flights),
                'distance': np.random.normal(1000, 500, n_flights).clip(200, 3000)
            }
            
            self.df = pd.DataFrame(data)
            
            # Remove flights with same origin and destination
            self.df = self.df[self.df['origin'] != self.df['dest']]
            
            # Generate realistic delay patterns
            base_delay = np.random.exponential(5, len(self.df))  # Most flights have small delays
            
            # Add seasonal effects
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * self.df['month'] / 12)
            
            # Add hour effects (rush hours have more delays)
            hour_factor = 1 + 0.4 * ((self.df['hour'] - 12) ** 2) / 144
            
            # Add airline-specific patterns
            airline_factors = {'AA': 1.2, 'UA': 1.1, 'DL': 0.9, 'WN': 1.0, 'B6': 1.3, 
                             'AS': 0.8, 'NK': 1.4, 'F9': 1.1}
            airline_multiplier = self.df['airline'].map(airline_factors)
            
            # Add airport-specific patterns
            busy_airports = {'ATL': 1.3, 'LAX': 1.2, 'ORD': 1.4, 'DFW': 1.1, 'JFK': 1.5}
            airport_multiplier = self.df['origin'].map(busy_airports).fillna(1.0)
            
            # Calculate final delays
            self.df['delay_minutes'] = (base_delay * seasonal_factor * hour_factor * 
                                      airline_multiplier * airport_multiplier).round(0)
            
            # Some flights are early (negative delay)
            early_mask = np.random.random(len(self.df)) < 0.3
            self.df.loc[early_mask, 'delay_minutes'] *= -0.5
            
            # Create binary delay indicator (>15 minutes)
            self.df['delayed'] = (self.df['delay_minutes'] > 15).astype(int)
            
            # Create route column (direction-independent)
            self.df['route'] = self.df.apply(lambda x: '-'.join(sorted([x['origin'], x['dest']])), axis=1)
            
            print(f"Dataset created: {len(self.df)} flights")
            print(f"Airlines: {sorted(self.df['airline'].unique())}")
            print(f"Delay rate: {self.df['delayed'].mean():.1%}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
        
        return True
    
    def analyze_overall_performance(self):
        """Generate overall airline performance charts"""
        
        # 1. Airline Performance (Delay Rate)
        airline_stats = self.df.groupby('airline').agg({
            'delayed': ['count', 'sum'],
            'delay_minutes': 'mean'
        }).round(3)
        
        airline_stats.columns = ['total_flights', 'delayed_flights', 'avg_delay']
        airline_stats['delay_rate'] = airline_stats['delayed_flights'] / airline_stats['total_flights']
        airline_stats = airline_stats.sort_values('delay_rate', ascending=True)
        
        # Store top airlines for later use
        self.top_airlines = airline_stats.index.tolist()
        
        # Chart 1: Delay Rate Bar Chart
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(airline_stats.index, airline_stats['delay_rate'], 
                      color='#4a90e2', alpha=0.8, edgecolor='#2c5aa0')
        ax.set_xlabel('Delay Rate (Proportion of Delayed Flights)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Airline', fontsize=12, fontweight='bold')
        ax.set_title('Airline Performance: Frequency of Delays', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                   f'{width:.1%}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('assets/airline_performance.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Chart 2: Total Flights by Airline (Pie Chart)
        fig, ax = plt.subplots(figsize=(10, 10))
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(airline_stats)))
        
        wedges, texts, autotexts = ax.pie(airline_stats['total_flights'], 
                                         labels=airline_stats.index,
                                         autopct='%1.1f%%',
                                         colors=colors,
                                         startangle=90,
                                         textprops={'fontsize': 14, 'fontweight': 'bold'})
        
        ax.set_title('Number of Flights by Airline', fontsize=16, fontweight='bold', pad=20)
        
        # Enhance text with larger fonts
        for text in texts:
            text.set_fontsize(16)
            text.set_fontweight('bold')
            
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(14)
        
        plt.tight_layout()
        plt.savefig('assets/flights_by_airline.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Chart 3: Delayed Flights Contribution (Pie Chart)
        fig, ax = plt.subplots(figsize=(10, 10))
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(airline_stats)))
        
        wedges, texts, autotexts = ax.pie(airline_stats['delayed_flights'], 
                                         labels=airline_stats.index,
                                         autopct='%1.1f%%',
                                         colors=colors,
                                         startangle=90)
        
        ax.set_title('Delayed Flight Contribution by Airline', fontsize=14, fontweight='bold', pad=20)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig('assets/delayed_flights_by_airline.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("Overall performance charts generated")
        
    def analyze_by_airline(self):
        """Generate airline-specific analysis"""
        
        for airline in self.top_airlines:
            airline_df = self.df[self.df['airline'] == airline]
            
            # 1. Most Common Routes
            route_counts = airline_df['route'].value_counts().head(10)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(range(len(route_counts)), route_counts.values, 
                          color='#4a90e2', alpha=0.8)
            ax.set_yticks(range(len(route_counts)))
            ax.set_yticklabels(route_counts.index)
            ax.set_xlabel('Number of Flights', fontsize=12, fontweight='bold')
            ax.set_ylabel('Route', fontsize=12, fontweight='bold')
            ax.set_title(f'{airline}: Most Common Travel Routes', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + max(route_counts.values) * 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{int(width)}', ha='left', va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'assets/{airline}_routes.png', dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # 2. Largest Hubs (Origin airports)
            hub_counts = airline_df['origin'].value_counts().head(10)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.bar(range(len(hub_counts)), hub_counts.values, 
                         color='#7fb069', alpha=0.8)
            ax.set_xticks(range(len(hub_counts)))
            ax.set_xticklabels(hub_counts.index, rotation=45, fontsize=14, fontweight='bold')
            ax.set_ylabel('Number of Originating Flights', fontsize=14, fontweight='bold')
            ax.set_xlabel('Airport', fontsize=14, fontweight='bold')
            ax.set_title(f'{airline}: Largest Hubs', fontsize=16, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Make y-axis labels larger
            ax.tick_params(axis='y', labelsize=12)
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + max(hub_counts.values) * 0.01,
                       f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f'assets/{airline}_hubs.png', dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # 3. Most Delayed Origins
            origin_delays = airline_df.groupby('origin')['delay_minutes'].mean().sort_values(ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.bar(range(len(origin_delays)), origin_delays.values, 
                         color='#e74c3c', alpha=0.8)
            ax.set_xticks(range(len(origin_delays)))
            ax.set_xticklabels(origin_delays.index, rotation=45)
            ax.set_ylabel('Average Delay (minutes)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Origin Airport', fontsize=12, fontweight='bold')
            ax.set_title(f'{airline}: Most Delayed Origins', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + max(origin_delays.values) * 0.01,
                       f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'assets/{airline}_worst_origins.png', dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # 4. Best Performing Origins
            origin_delays_best = airline_df.groupby('origin')['delay_minutes'].mean().sort_values().head(10)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.bar(range(len(origin_delays_best)), origin_delays_best.values, 
                         color='#27ae60', alpha=0.8)
            ax.set_xticks(range(len(origin_delays_best)))
            ax.set_xticklabels(origin_delays_best.index, rotation=45, fontsize=14, fontweight='bold')
            ax.set_ylabel('Average Delay (minutes)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Origin Airport', fontsize=14, fontweight='bold')
            ax.set_title(f'{airline}: Best Performing Origins', fontsize=16, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Make y-axis labels larger
            ax.tick_params(axis='y', labelsize=12)
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + abs(min(origin_delays_best.values)) * 0.1,
                       f'{height:.1f}', ha='center', va='bottom' if height >= 0 else 'top', 
                       fontweight='bold', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f'assets/{airline}_best_origins.png', dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"Charts generated for {airline}")
    
    def analyze_delay_distributions(self):
        """Generate delay distribution histograms"""
        
        # Top 5 airlines by flight count
        top5_airlines = self.df['airline'].value_counts().head(5).index
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, airline in enumerate(top5_airlines):
            airline_df = self.df[self.df['airline'] == airline]
            
            axes[i].hist(airline_df['delay_minutes'], bins=50, alpha=0.7, 
                        color='#4a90e2', edgecolor='black')
            axes[i].set_title(f'{airline}: Delay Distribution', fontweight='bold')
            axes[i].set_xlabel('Delay Minutes')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(axis='y', alpha=0.3)
            axes[i].axvline(airline_df['delay_minutes'].mean(), color='red', 
                           linestyle='--', label=f'Mean: {airline_df["delay_minutes"].mean():.1f}min')
            axes[i].legend()
        
        # Remove empty subplot
        axes[5].remove()
        
        plt.suptitle('Delay Time Distributions: Top 5 Airlines', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('assets/airline_delay_distributions.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Airport delay distributions - most and least delayed
        airport_delays = self.df.groupby('origin')['delay_minutes'].mean().sort_values()
        
        # Top 5 most delayed and 5 least delayed airports
        worst_airports = airport_delays.tail(5).index
        best_airports = airport_delays.head(5).index
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        
        # Most delayed airports
        for i, airport in enumerate(worst_airports):
            airport_df = self.df[self.df['origin'] == airport]
            axes[0, i].hist(airport_df['delay_minutes'], bins=30, alpha=0.7, 
                           color='#e74c3c', edgecolor='black')
            axes[0, i].set_title(f'{airport}: Most Delayed', fontweight='bold')
            axes[0, i].set_xlabel('Delay Minutes')
            axes[0, i].set_ylabel('Frequency')
            axes[0, i].grid(axis='y', alpha=0.3)
            mean_delay = airport_df['delay_minutes'].mean()
            axes[0, i].axvline(mean_delay, color='darkred', linestyle='--', 
                              label=f'Mean: {mean_delay:.1f}min')
            axes[0, i].legend()
        
        # Least delayed airports
        for i, airport in enumerate(best_airports):
            airport_df = self.df[self.df['origin'] == airport]
            axes[1, i].hist(airport_df['delay_minutes'], bins=30, alpha=0.7, 
                           color='#27ae60', edgecolor='black')
            axes[1, i].set_title(f'{airport}: Best Performing', fontweight='bold')
            axes[1, i].set_xlabel('Delay Minutes')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].grid(axis='y', alpha=0.3)
            mean_delay = airport_df['delay_minutes'].mean()
            axes[1, i].axvline(mean_delay, color='darkgreen', linestyle='--', 
                              label=f'Mean: {mean_delay:.1f}min')
            axes[1, i].legend()
        
        plt.suptitle('Airport Delay Distributions: Best vs Worst Performing', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('assets/airport_delay_distributions.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("Delay distribution histograms generated")
    
    def build_classification_models(self):
        """Build and compare KNN vs SVM for delay prediction"""
        
        # Prepare features for classification
        le_airline = LabelEncoder()
        le_origin = LabelEncoder()
        le_dest = LabelEncoder()
        
        # Create feature matrix
        X = pd.DataFrame({
            'airline': le_airline.fit_transform(self.df['airline']),
            'origin': le_origin.fit_transform(self.df['origin']),
            'dest': le_dest.fit_transform(self.df['dest']),
            'month': self.df['month'],
            'day': self.df['day'],
            'hour': self.df['hour'],
            'distance': self.df['distance']
        })
        
        # Target: delay > 20 minutes
        y = (self.df['delay_minutes'] > 20).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        knn = KNeighborsClassifier(n_neighbors=5)
        svm = SVC(kernel='rbf', probability=True, random_state=42)
        
        # Fit models
        knn.fit(X_train_scaled, y_train)
        svm.fit(X_train_scaled, y_train)
        
        # Cross-validation scores
        knn_cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
        svm_cv_scores = cross_val_score(svm, X_train_scaled, y_train, cv=5)
        
        # Test predictions
        knn_pred = knn.predict(X_test_scaled)
        svm_pred = svm.predict(X_test_scaled)
        
        # Test accuracies
        knn_accuracy = accuracy_score(y_test, knn_pred)
        svm_accuracy = accuracy_score(y_test, svm_pred)
        
        # Create comparison chart
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Accuracy comparison
        models = ['KNN', 'SVM']
        cv_means = [knn_cv_scores.mean(), svm_cv_scores.mean()]
        cv_stds = [knn_cv_scores.std(), svm_cv_scores.std()]
        test_accs = [knn_accuracy, svm_accuracy]
        
        axes[0].bar(models, cv_means, yerr=cv_stds, capsize=5, 
                   color=['#4a90e2', '#7fb069'], alpha=0.8)
        axes[0].set_title('Cross-Validation Accuracy', fontweight='bold')
        axes[0].set_ylabel('Accuracy')
        axes[0].grid(axis='y', alpha=0.3)
        
        for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
            axes[0].text(i, mean + std + 0.005, f'{mean:.3f} ± {std:.3f}', 
                        ha='center', va='bottom', fontweight='bold')
        
        # Test accuracy comparison
        bars = axes[1].bar(models, test_accs, color=['#4a90e2', '#7fb069'], alpha=0.8)
        axes[1].set_title('Test Set Accuracy', fontweight='bold')
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(axis='y', alpha=0.3)
        
        for i, (bar, acc) in enumerate(zip(bars, test_accs)):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Confusion matrices side by side
        from sklearn.metrics import confusion_matrix
        import itertools
        
        def plot_confusion_matrix(cm, classes, ax, title, cmap=plt.cm.Blues):
            ax.imshow(cm, interpolation='nearest', cmap=cmap)
            ax.set_title(title, fontweight='bold')
            tick_marks = np.arange(len(classes))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)
            
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                ax.text(j, i, format(cm[i, j], 'd'),
                       horizontalalignment="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontweight='bold')
            
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        # Create subplot for confusion matrices
        knn_cm = confusion_matrix(y_test, knn_pred)
        svm_cm = confusion_matrix(y_test, svm_pred)
        
        # We'll create this as a separate figure
        axes[2].axis('off')
        axes[2].text(0.5, 0.5, f'KNN Accuracy: {knn_accuracy:.3f}\nSVM Accuracy: {svm_accuracy:.3f}\n\nBest Model: {"SVM" if svm_accuracy > knn_accuracy else "KNN"}', 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.suptitle('Flight Delay Classification: KNN vs SVM (>20 min delays)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('assets/classification_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Generate detailed classification results
        results = {
            'knn': {
                'cv_mean': float(knn_cv_scores.mean()),
                'cv_std': float(knn_cv_scores.std()),
                'test_accuracy': float(knn_accuracy),
                'classification_report': classification_report(y_test, knn_pred, output_dict=True)
            },
            'svm': {
                'cv_mean': float(svm_cv_scores.mean()),
                'cv_std': float(svm_cv_scores.std()),
                'test_accuracy': float(svm_accuracy),
                'classification_report': classification_report(y_test, svm_pred, output_dict=True)
            }
        }
        
        # Save results to JSON
        with open('assets/classification_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Classification models completed")
        print(f"KNN Accuracy: {knn_accuracy:.3f}")
        print(f"SVM Accuracy: {svm_accuracy:.3f}")
        
    def generate_interactive_data(self):
        """Generate JSON data for interactive components"""
        
        # Airline list for dropdown
        airline_list = self.top_airlines
        
        # Save airline data
        with open('assets/airline_list.json', 'w') as f:
            json.dump(airline_list, f)
        
        print("Interactive data generated")
    
    def run_complete_analysis(self):
        """Execute the complete analysis pipeline"""
        print("Starting Flight Analysis Dashboard...")
        
        if not self.load_data():
            return
        
        print("Generating overall performance analysis...")
        self.analyze_overall_performance()
        
        print("Analyzing airline-specific data...")
        self.analyze_by_airline()
        
        print("Creating delay distribution histograms...")
        self.analyze_delay_distributions()
        
        print("Building classification models...")
        self.build_classification_models()
        
        print("Generating interactive data...")
        self.generate_interactive_data()
        
        print("\nAnalysis complete! All charts and data saved to 'assets' folder.")
        print("Open index.html to view the interactive dashboard.")

if __name__ == "__main__":
    analysis = FlightAnalysis()
    analysis.run_complete_analysis()